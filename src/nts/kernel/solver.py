"""CP-SAT timetabling model (proposal Section 8.3).

Physical rules (Tier 0: every session once, no room/faculty/group clash, a
compatible room) are always enforced. Every other hard constraint ``C_k`` is
guarded by an assumption literal ``a_k`` so it can be switched on or off
(``a_k => C_k``); that is what conflict analysis and negotiation work on.
Soft constraints and, in repair mode, the number of moved sessions form the
objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ortools.sat.python import cp_model

from .instance import Instance
from .schemas import Constraint, ConstraintType, Placement, Tier
from .semantics import (
    PLACEMENT_TYPES,
    consecutive_entities,
    occupied,
    room_compatible,
    sessions_in_scope,
    violates,
)

# Candidate start key: (day index, slot, room id)
Key = tuple[int, int, str]


class InstanceInfeasible(RuntimeError):
    """The instance cannot be timetabled even with no request constraints."""


class SolverTimeout(RuntimeError):
    """CP-SAT hit its time limit without proving feasibility either way."""


_STATUS = {
    cp_model.OPTIMAL: "optimal",
    cp_model.FEASIBLE: "feasible",
    cp_model.INFEASIBLE: "infeasible",
    cp_model.UNKNOWN: "unknown",
}


@dataclass
class SolveResult:
    status: str
    assignment: dict[str, Placement] | None = None
    enforced: list[str] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)
    soft_violations: dict[str, int] = field(default_factory=dict)
    moved: int | None = None
    objective: float | None = None
    wall_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "feasible")


# Replayed answers: the web app's start-up history solves the same models every time it
# starts, so ``use_solve_cache`` keeps their answers on disk. Evaluations never turn it on
# (a replayed answer takes no time, which would falsify solve timings).
_CACHE: Path | None = None


def use_solve_cache(folder: Path | str | None) -> None:
    """Keep CP-SAT answers in ``folder``, keyed by the model and solver parameters; None turns it off."""
    global _CACHE
    _CACHE = Path(folder) if folder else None
    if _CACHE:
        _CACHE.mkdir(parents=True, exist_ok=True)


class _Replay:
    """A cached CP-SAT answer, read like the solver that produced it."""

    def __init__(self, d: dict) -> None:
        self.solution: list[int] = d["solution"]
        self.objective_value: float = d["objective"]
        self._core: list[int] = d["core"]

    def boolean_value(self, lit) -> bool:
        i = lit.index
        return bool(self.solution[i]) if i >= 0 else not self.solution[-i - 1]

    def sufficient_assumptions_for_infeasibility(self) -> list[int]:
        return self._core


class _Model:
    """One CP-SAT model of an instance plus its active constraints."""

    def __init__(
        self,
        instance: Instance,
        hard: list[Constraint],
        soft: list[Constraint],
        baseline: Mapping[str, Placement] | None,
        disruption_weight: int,
        skip: frozenset[str] = frozenset(),
    ) -> None:
        self.instance = instance
        self.skip = skip  # sessions not held this week (cancelled): they get no placement
        self.model = cp_model.CpModel()
        self.x: dict[str, dict[Key, cp_model.IntVar]] = {}
        self._fac_cov: dict[tuple[str, int, int], list[cp_model.IntVar]] = defaultdict(list)
        self._grp_cov: dict[tuple[str, int, int], list[cp_model.IntVar]] = defaultdict(list)
        self._build_physical()

        self.lit: dict[str, cp_model.IntVar] = {}
        for c in hard:
            self.lit[c.id] = self.model.new_bool_var(f"a[{c.id}]")
            self._add_guarded(c, self.lit[c.id])

        self.soft: dict[str, tuple[int, list[cp_model.IntVar]]] = {
            c.id: (max(1, round((c.weight or 1.0) * 10)), self._bad(c)) for c in soft
        }

        self.baseline_size = len(baseline or {})
        self.keep: list[cp_model.IntVar] = []
        for sid, pl in (baseline or {}).items():
            key = (instance.day_index(pl.day), pl.slot, pl.room)
            var = self.x.get(sid, {}).get(key)
            if var is not None:
                self.keep.append(var)
        self.disruption_weight = disruption_weight

    def _build_physical(self) -> None:
        inst, m = self.instance, self.model
        n_days, spd = len(inst.calendar.days), inst.calendar.slots_per_day
        room_cov: dict[tuple[str, int, int], list[cp_model.IntVar]] = defaultdict(list)
        for s in inst.sessions:
            if s.id in self.skip:
                continue
            cands: dict[Key, cp_model.IntVar] = {}
            rooms = [r for r in inst.rooms if room_compatible(inst, s, r)]
            for d in range(n_days):
                for p in range(spd - s.duration + 1):
                    for r in rooms:
                        v = m.new_bool_var(f"x[{s.id},{d},{p},{r.id}]")
                        cands[(d, p, r.id)] = v
                        for dd, q in occupied(s, d, p):
                            room_cov[(r.id, dd, q)].append(v)
                            self._fac_cov[(s.faculty, dd, q)].append(v)
                            for g in s.groups:
                                self._grp_cov[(g, dd, q)].append(v)
            if not cands:
                raise InstanceInfeasible(f"session {s.id} has no compatible room")
            m.add_exactly_one(cands.values())
            self.x[s.id] = cands
        for cover in (room_cov, self._fac_cov, self._grp_cov):
            for vs in cover.values():
                if len(vs) > 1:
                    m.add_at_most_one(vs)

    def _bad(self, c: Constraint) -> list[cp_model.IntVar]:
        """Placement variables that would break placement constraint ``c``."""
        inst = self.instance
        return [
            v
            for s in sessions_in_scope(inst, c.scope)
            for (d, p, r), v in self.x.get(s.id, {}).items()
            if violates(inst, c, s, d, p, inst.room_by_id[r])
        ]

    def _add_guarded(self, c: Constraint, lit: cp_model.IntVar) -> None:
        if c.type in PLACEMENT_TYPES:
            bad = self._bad(c)
            if bad:
                self.model.add_bool_and([~b for b in bad]).only_enforce_if(lit)
            return
        if c.type == ConstraintType.MAX_CONSECUTIVE:
            assert c.limit is not None
            spd = self.instance.calendar.slots_per_day
            for kind, eid in consecutive_entities(self.instance, c.scope):
                cover = self._fac_cov if kind == "faculty" else self._grp_cov
                for d in range(len(self.instance.calendar.days)):
                    for w in range(spd - c.limit):
                        terms = [v for q in range(w, w + c.limit + 1) for v in cover.get((eid, d, q), [])]
                        if len(terms) > c.limit:
                            self.model.add(sum(terms) <= c.limit).only_enforce_if(lit)
            return
        raise ValueError(f"unsupported constraint type {c.type.value}")

    def penalty(self) -> cp_model.LinearExprT:
        soft = sum(coef * sum(bad) for coef, bad in self.soft.values() if bad)
        moved = self.baseline_size - sum(self.keep)
        return soft + self.disruption_weight * moved

    def hint_from(self, solver: cp_model.CpSolver | _Replay) -> None:
        self.model.clear_hints()
        for cands in self.x.values():
            for v in cands.values():
                self.model.add_hint(v, solver.boolean_value(v))


class TimetableSolver:
    """Builds, checks and repairs timetables for one instance and week."""

    def __init__(
        self,
        instance: Instance,
        constraints: Iterable[Constraint],
        *,
        week: int | None = None,
        baseline: Mapping[str, Placement] | None = None,
        time_limit: float = 30.0,
        workers: int = 8,
        seed: int = 0,
        disruption_weight: int = 1,
        presolve: bool = True,
        skip: Iterable[str] = (),
    ) -> None:
        active = [c for c in constraints if c.active_in(week)]
        ids = [c.id for c in active]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate constraint ids")
        self.instance = instance
        self.week = week
        self.baseline = dict(baseline or {})
        self.constraints = {c.id: c for c in active}
        self.hard_ids = [c.id for c in active if c.hard]
        self.time_limit = time_limit
        self.workers = workers
        self.seed = seed
        self.disruption_weight = disruption_weight
        # CP-SAT presolve dominates on department-size models (about 2 of 3 s
        # per feasibility check); interactive use turns it off.
        self.presolve = presolve
        self.skip = frozenset(skip)  # sessions cancelled this week (pipeline.orchestrator.closures)
        self._feas: _Model | None = None

    # -- building and running -------------------------------------------------

    def build(self) -> _Model:
        hard = [c for c in self.constraints.values() if c.hard]
        soft = [c for c in self.constraints.values() if not c.hard]
        return _Model(self.instance, hard, soft, self.baseline, self.disruption_weight, self.skip)

    def run(self, m: _Model) -> tuple[str, cp_model.CpSolver | _Replay]:
        path = None
        if _CACHE is not None:
            params = f"{self.time_limit}|{self.workers}|{self.seed}|{self.presolve}|"
            path = _CACHE / f"{hashlib.sha256((params + str(m.model.proto)).encode()).hexdigest()}.json"
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                return d["status"], _Replay(d)
        status, solver = self._solve(m)
        if path is not None and status != "unknown":  # a timeout may go the other way next time
            r = solver.response_proto
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"status": status, "solution": list(r.solution), "objective": r.objective_value,
                                       "core": list(r.sufficient_assumptions_for_infeasibility)}), encoding="utf-8")
            tmp.replace(path)
        return status, solver

    def _solve(self, m: _Model) -> tuple[str, cp_model.CpSolver]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_workers = self.workers
        solver.parameters.random_seed = self.seed
        solver.parameters.cp_model_presolve = self.presolve
        status = solver.solve(m.model)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(m.model.validate())
        return _STATUS[status], solver

    # -- feasibility and cores ----------------------------------------------

    def core(self, enabled: Iterable[str]) -> list[str] | None:
        """``None`` if ``enabled`` hard constraints are jointly feasible, else a
        subset of them that is already infeasible (not necessarily minimal)."""
        enabled = list(enabled)
        if self._feas is None:
            self._feas = self.build()
        m = self._feas
        m.model.clear_assumptions()
        m.model.add_assumptions([m.lit[i] for i in enabled])
        status, solver = self.run(m)
        if status in ("optimal", "feasible"):
            return None
        if status == "unknown":
            raise SolverTimeout(f"no answer within {self.time_limit}s")
        if not enabled:
            raise InstanceInfeasible("infeasible even without request constraints")
        by_index = {m.lit[i].index: i for i in enabled}
        found = [by_index[k] for k in solver.sufficient_assumptions_for_infeasibility() if k in by_index]
        return found or enabled

    def is_feasible(self, enabled: Iterable[str]) -> bool:
        return self.core(enabled) is None

    # -- optimisation -------------------------------------------------------

    def solve(self, enabled: Iterable[str] | None = None) -> SolveResult:
        """Enforce ``enabled`` hard constraints (default: all) and minimise soft
        violations plus disruption against the baseline."""
        on = set(self.hard_ids if enabled is None else enabled)
        m = self.build()
        for i, lit in m.lit.items():
            m.model.add(lit == int(i in on))
        m.model.minimize(m.penalty())
        return self._finish(m, *self._timed_run(m))

    def solve_lexicographic(self, scores: Mapping[str, int] | None = None) -> SolveResult:
        """Tiers first, then scores (proposal Section 8.4).

        Tiers 0-2 are enforced; then the satisfied Tier 3 score is maximised
        and fixed, then Tier 4, then soft preferences and disruption.
        """
        scores = scores or {}
        m = self.build()
        start = time.perf_counter()
        for i, lit in m.lit.items():
            if self.constraints[i].tier <= Tier.COMMITMENT:
                m.model.add(lit == 1)
        for tier in (Tier.VERIFIED_UNAVAILABILITY, Tier.OPERATIONAL):
            ids = [i for i in m.lit if self.constraints[i].tier == tier]
            if not ids:
                continue
            expr = sum(self._score(i, scores) * m.lit[i] for i in ids)
            m.model.maximize(expr)
            status, solver = self.run(m)
            if status not in ("optimal", "feasible"):
                return SolveResult(status=status, wall_time=time.perf_counter() - start)
            m.model.add(expr >= round(solver.objective_value))
            m.hint_from(solver)
        m.model.minimize(m.penalty())
        status, solver = self.run(m)
        result = self._finish(m, status, solver, 0.0)
        result.wall_time = time.perf_counter() - start
        return result

    def _score(self, cid: str, scores: Mapping[str, int]) -> int:
        if cid in scores:
            return scores[cid]
        weight = self.constraints[cid].weight
        return max(1, round(weight)) if weight else 1

    def _timed_run(self, m: _Model) -> tuple[str, cp_model.CpSolver, float]:
        start = time.perf_counter()
        status, solver = self.run(m)
        return status, solver, time.perf_counter() - start

    def _finish(self, m: _Model, status: str, solver: cp_model.CpSolver | _Replay, wall: float) -> SolveResult:
        if status not in ("optimal", "feasible"):
            return SolveResult(status=status, wall_time=wall)
        days = self.instance.calendar.days
        assignment = {
            sid: Placement(day=days[d], slot=p, room=r)
            for sid, cands in m.x.items()
            for (d, p, r), v in cands.items()
            if solver.boolean_value(v)
        }
        soft = {
            cid: n
            for cid, (_, bad) in m.soft.items()
            if (n := sum(solver.boolean_value(b) for b in bad)) > 0
        }
        kept = sum(solver.boolean_value(v) for v in m.keep)
        return SolveResult(
            status=status,
            assignment=assignment,
            enforced=[i for i, lit in m.lit.items() if solver.boolean_value(lit)],
            relaxed=[i for i, lit in m.lit.items() if not solver.boolean_value(lit)],
            soft_violations=soft,
            moved=m.baseline_size - kept if self.baseline else None,
            objective=solver.objective_value,
            wall_time=wall,
        )
