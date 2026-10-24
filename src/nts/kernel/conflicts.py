"""Conflict analysis (proposal Section 8.5).

* ``find_mus`` shrinks an infeasible set of hard constraints to a minimal
  unsatisfiable subset: every member is needed for the conflict.
* ``enumerate_mcs`` lists minimal correction sets: minimal sets of relaxable
  constraints whose removal restores feasibility. Each comes with a witness
  timetable, so every option the negotiator offers is solver-verified.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from .schemas import Tier
from .solver import SolveResult, TimetableSolver

DEFAULT_RELAXABLE = frozenset({Tier.VERIFIED_UNAVAILABILITY, Tier.OPERATIONAL})


@dataclass
class CorrectionOption:
    drop: list[str]
    cost: int
    witness: SolveResult


def find_mus(solver: TimetableSolver, enabled: Iterable[str] | None = None) -> list[str]:
    """Deletion-based shrinking of the solver's infeasibility core.

    Raises ``ValueError`` if ``enabled`` is feasible.
    """
    enabled = list(solver.hard_ids if enabled is None else enabled)
    core = solver.core(enabled)
    if core is None:
        raise ValueError("constraints are feasible; there is no conflict")
    mus = sorted(core)
    i = 0
    while i < len(mus):
        trial = mus[:i] + mus[i + 1 :]
        smaller = solver.core(trial)
        if smaller is None:
            i += 1  # mus[i] is necessary
        else:
            # Necessary members appear in every infeasible subset, so filtering
            # keeps mus[:i] intact and in order.
            keep = set(smaller)
            mus = [c for c in trial if c in keep]
    return mus


def enumerate_mcs(
    solver: TimetableSolver,
    enabled: Iterable[str] | None = None,
    *,
    relaxable_tiers: Collection[Tier] = DEFAULT_RELAXABLE,
    costs: Mapping[str, int] | None = None,
    limit: int = 5,
    keep: Collection[str] = (),
) -> list[CorrectionOption]:
    """Up to ``limit`` minimal correction sets, cheapest first.

    Only constraints in ``relaxable_tiers`` may be dropped; tiers outside it
    stay enforced, and so do the constraints in ``keep`` (e.g. ones whose
    owner already refused to concede). ``costs`` (default 1 each) must be
    positive, which is what makes each cheapest remaining solution
    subset-minimal.
    """
    enabled = list(solver.hard_ids if enabled is None else enabled)
    kept = set(keep)
    relaxable = [i for i in enabled if solver.constraints[i].tier in relaxable_tiers and i not in kept]
    cost = {i: (costs or {}).get(i, 1) for i in relaxable}
    if any(v <= 0 for v in cost.values()):
        raise ValueError("MCS costs must be positive")

    m = solver.build()
    on = set(enabled)
    for i, lit in m.lit.items():
        if i not in on:
            m.model.add(lit == 0)
        elif i not in cost:
            m.model.add(lit == 1)
    if relaxable:
        m.model.minimize(sum(c * (1 - m.lit[i]) for i, c in cost.items()))

    options: list[CorrectionOption] = []
    while len(options) < limit:
        status, cp = solver.run(m)
        if status == "infeasible":
            break
        if status == "unknown":
            break
        drop = sorted(i for i in relaxable if not cp.boolean_value(m.lit[i]))
        if not drop:
            return []  # already feasible: nothing to correct
        if status == "feasible":  # timed out before proving optimality
            drop = _shrink(solver, on, drop)
        witness = solver.solve(on - set(drop))
        options.append(CorrectionOption(drop=drop, cost=sum(cost[i] for i in drop), witness=witness))
        m.model.add_bool_or([m.lit[i] for i in drop])  # block this set and its supersets
    return options


def _shrink(solver: TimetableSolver, on: set[str], drop: list[str]) -> list[str]:
    """Greedily re-enable dropped constraints that are not actually needed."""
    needed = list(drop)
    for i in list(drop):
        trial = [d for d in needed if d != i]
        if solver.is_feasible(on - set(trial)):
            needed = trial
    return needed
