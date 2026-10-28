"""CP-SAT timetabling model (proposal Section 8.3).

Physical rules (Tier 0: every session once, no room/faculty/group clash, a
compatible room) are always enforced. Every other hard constraint ``C_k`` is
guarded by an assumption literal ``a_k`` so it can be switched on or off
(``a_k => C_k``); that is what conflict analysis and negotiation work on.
Soft constraints and, in repair mode, the number of moved sessions form the
objective.

A repair can be restricted to a neighbourhood (``free``): every other session
keeps its baseline placement, which shrinks the model to the part of the
department a conflict touches. Feasible within a neighbourhood is feasible
outright; infeasible within one is not proof, so callers check a conflict
found there against the whole timetable (the negotiation of core-rework).
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
    related,
    relation_holds,
    relation_key,
    room_compatible,
    satisfied,
    sessions_in_scope,
    touched,
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
        frozen: Mapping[str, Placement] | None = None,
    ) -> None:
        self.instance = instance
        self.skip = skip  # sessions not held this week (cancelled): they get no placement
        self.frozen = dict(frozen or {})  # sessions held where they are: one candidate each
        self.model = cp_model.CpModel()
        self.x: dict[str, dict[Key, cp_model.IntVar]] = {}
        self._fac_cov: dict[tuple[str, int, int], list[cp_model.IntVar]] = defaultdict(list)
        self._grp_cov: dict[tuple[str, int, int], list[cp_model.IntVar]] = defaultdict(list)
        self._build_physical()

        self.lit: dict[str, cp_model.IntVar] = {}
        for c in hard:
            self.lit[c.id] = self.model.new_bool_var(f"a[{c.id}]")
            self._add_guarded(c, self.lit[c.id])

        # a soft placement constraint costs per placement that breaks it; any other soft constraint costs once
        # for each of its ok literals that is false (a ranked choice has one per choice it falls short of)
        self.soft: dict[str, tuple[int, list[cp_model.IntVar]]] = {}
        self.soft_ok: dict[str, tuple[int, list[cp_model.IntVar]]] = {}
        for c in soft:
            coef = max(1, round((c.weight or 1.0) * 10))
            if c.type in PLACEMENT_TYPES:
                self.soft[c.id] = (coef, self._bad(c))
            else:
                self.soft_ok[c.id] = (coef, self._soft_terms(c))

        self.baseline_size = len(baseline or {})
        self.keep: list[cp_model.IntVar] = []
        for sid, pl in (baseline or {}).items():
            key = (instance.day_index(pl.day), pl.slot, pl.room)
            var = self.x.get(sid, {}).get(key)
            if var is not None:
                self.keep.append(var)
        # the search starts from the baseline: a repair usually leaves most of it in place, and without a
        # hint each check builds a whole timetable from nothing (seconds per check in a department)
        kept = set(map(id, self.keep))
        for sid in (baseline or {}):
            for v in self.x.get(sid, {}).values():
                self.model.add_hint(v, id(v) in kept)
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
            at = self.frozen.get(s.id)
            for d in range(n_days):
                for p in range(spd - s.duration + 1):
                    for r in rooms:
                        if at is not None and (d, p, r.id) != (inst.day_index(at.day), at.slot, at.room):
                            continue
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
        if c.members:  # a choice: at least one member holds
            self.model.add_bool_or([self._ok(m) for m in c.members]).only_enforce_if(lit)
            return
        if c.type == ConstraintType.LOAD:
            self._load(c, lit)
            return
        if c.type == ConstraintType.RELATE:
            self._relate(c, lit)
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

    def _ok(self, c: Constraint) -> cp_model.IntVar:
        """A literal that, when true, enforces ``c`` (a member of a choice, or a soft constraint)."""
        ok = self.model.new_bool_var(f"ok[{c.id}]")
        self._add_guarded(c, ok)
        return ok

    def _soft_terms(self, c: Constraint) -> list[cp_model.IntVar]:
        """The literals a soft constraint pays for when false: one for most; for a ranked choice, one per
        choice: "the first or better" true when the first holds, "the second or better" when either does, so
        the first choice costs nothing, the second one unit, none of them one per choice."""
        if c.type != ConstraintType.RANKED:
            return [self._ok(c)]
        oks, out = [self._ok(m) for m in c.members], []
        for k in range(1, len(oks) + 1):
            better = self.model.new_bool_var(f"ranked[{c.id},{k}]")
            self.model.add_bool_or(oks[:k]).only_enforce_if(better)
            out.append(better)
        return out

    def _load(self, c: Constraint, lit: cp_model.IntVar) -> None:
        """Classes a day, hours a day, days a week, or free hours between two classes on a day."""
        inst, m = self.instance, self.model
        day: dict[int, list[tuple[int, cp_model.IntVar]]] = defaultdict(list)  # (hours, var) by day
        start: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
        occ: dict[tuple[int, int], list[cp_model.IntVar]] = defaultdict(list)
        for s in sessions_in_scope(inst, c.scope):
            for (d, p, _r), v in self.x.get(s.id, {}).items():
                day[d].append((s.duration, v))
                start[(d, p)].append(v)
                for dd, q in occupied(s, d, p):
                    occ[(dd, q)].append(v)
        limit = c.limit or 0
        if c.measure == "classes_per_day":
            for vs in day.values():
                m.add(sum(v for _h, v in vs) <= limit).only_enforce_if(lit)
        elif c.measure == "hours_per_day":
            for vs in day.values():
                m.add(sum(h * v for h, v in vs) <= limit).only_enforce_if(lit)
        elif c.measure == "days_per_week":
            teaches = []
            for d, vs in day.items():
                t = m.new_bool_var(f"teaches[{c.id},{d}]")
                for _h, v in vs:
                    m.add_implication(v, t)
                teaches.append(t)
            m.add(sum(teaches) <= limit).only_enforce_if(lit)
        elif c.measure == "gap":  # a class starting at q: nothing of theirs in the ``limit`` hours before it
            for (d, q), starts in start.items():
                for p in range(max(0, q - limit), q):
                    if occ.get((d, p)):
                        m.add(sum(starts) + sum(occ[(d, p)]) <= 1).only_enforce_if(lit)
        else:
            raise ValueError(f"unknown measure {c.measure!r}")

    def _relate(self, c: Constraint, lit: cp_model.IntVar) -> None:
        """Each pair of placements (by the part the relation looks at) that breaks the relation is ruled out;
        against one of the others, a literal per other says which one it holds against."""
        m = self.model

        def by_key(sid: str) -> dict:
            out: dict = defaultdict(list)
            for (d, p, r), v in self.x.get(sid, {}).items():
                out[relation_key(c.relation, d, p, r)].append(v)
            return out

        for a, others in related(self.instance, c):
            others = [b for b in others if b.id in self.x]
            if a.id not in self.x or not others:
                continue
            ka_vars = by_key(a.id)
            holds = [m.new_bool_var(f"rel[{c.id},{a.id},{b.id}]") for b in others] if c.any_other else None
            if holds:
                m.add_bool_or(holds).only_enforce_if(lit)
            for i, b in enumerate(others):
                when = [lit, holds[i]] if holds else lit
                for kb, vb in by_key(b.id).items():
                    for ka, va in ka_vars.items():
                        if not relation_holds(c, ka, kb, b):
                            m.add(sum(va) + sum(vb) <= 1).only_enforce_if(when)

    def penalty(self) -> cp_model.LinearExprT:
        soft = sum(coef * sum(bad) for coef, bad in self.soft.values() if bad)
        soft += sum(coef * (len(oks) - sum(oks)) for coef, oks in self.soft_ok.values())
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
        free: Iterable[str] | None = None,
    ) -> None:
        """``free``: the sessions a repair may move (a neighbourhood); every other session with a baseline
        placement stays there. None: every session may move."""
        # a choice keeps the members that apply this week
        active = [c if not c.members else c.model_copy(update={"members": [m for m in c.members if m.active_in(week)]})
                  for c in constraints if c.active_in(week)]
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
        self.free = None if free is None else frozenset(free)
        self._feas: _Model | None = None

    # -- building and running -------------------------------------------------

    def build(self) -> _Model:
        hard = [c for c in self.constraints.values() if c.hard]
        soft = [c for c in self.constraints.values() if not c.hard]
        frozen = ({s: p for s, p in self.baseline.items() if s not in self.free} if self.free is not None else None)
        return _Model(self.instance, hard, soft, self.baseline, self.disruption_weight, self.skip, frozen)

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
        soft.update({cid: n for cid, (_, oks) in m.soft_ok.items()
                     if (n := sum(not solver.boolean_value(ok) for ok in oks)) > 0})
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


def neighbourhood(instance: Instance, constraints: Iterable[Constraint], baseline: Mapping[str, Placement],
                  week: int | None = None) -> set[str]:
    """The sessions a repair of ``baseline`` is likely to move: those it places against an active placement
    constraint (or does not place), and those that share a teacher or a section with them or sit in a room
    one of them could use. A repair restricted to these leaves the rest of the department where it is."""
    inst = instance
    seeds = {s.id for s in inst.sessions if s.id not in baseline}
    placed = {sid: (inst.day_index(p.day), p.slot, p.room) for sid, p in baseline.items() if sid in inst.session_by_id}
    for c in constraints:
        if not c.active_in(week):
            continue
        if c.type not in PLACEMENT_TYPES:
            # a load limit, a relation or a choice the baseline breaks: every class it is about may have to move
            if not satisfied(inst, c, placed, week):
                seeds |= touched(inst, c)
            continue
        for s in sessions_in_scope(inst, c.scope):
            p = baseline.get(s.id)
            if p is not None and violates(inst, c, s, inst.day_index(p.day), p.slot, inst.room_by_id[p.room]):
                seeds.add(s.id)
    out = set(seeds)
    for sid in seeds:
        s = inst.session_by_id[sid]
        rooms = {r.id for r in inst.rooms if room_compatible(inst, s, r)}
        for o in inst.sessions:
            p = baseline.get(o.id)
            if (o.faculty == s.faculty or set(o.groups) & set(s.groups)
                    or (p is not None and p.room in rooms)):
                out.add(o.id)
    return out
