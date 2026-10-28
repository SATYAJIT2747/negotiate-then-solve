"""Other times for a class whose move another teacher's rule blocks, checked by the solver (coordinator-v5).

When the owner of the blocking rule says no, or does not answer, ``alternatives`` finds up to two times for
the requester's class that work with that rule kept: the same day at the nearest hours first, then the
same hour on other days, then the rest. When the owner proposes a time instead (a counter-offer), ``works``
says whether the class can go there. Either way the solver decides, not the model or the person: a time is
offered only if the whole week (or term) can be repaired with it.
"""

from __future__ import annotations

from nts.kernel.schemas import Constraint, ConstraintType, When
from nts.kernel.solver import SolverTimeout, TimetableSolver
from nts.state.store import touched

from .names import FULL_DAY, hour
from .registry import Context

TIME_LIMIT = 5.0
TRIES = 8  # solves at most, per call
WEEKS = 3  # weeks of a change checked at most (a term-long change: the semester timetable and two weeks)


def wanted(ctx: Context) -> Constraint | None:
    """The requester's own move in the draft: a firm 'classes only then' rule on one class."""
    world = ctx.state
    ours = [cid for ids in ctx.scratch.get("changes", {}).values() for cid in ids]
    return next((world.constraints[c] for c in ours if c in world.constraints
                 and world.constraints[c].type == ConstraintType.PREFER and world.constraints[c].scope.session
                 and world.constraints[c].hard), None)


def alternatives(ctx: Context, c: Constraint, n: int = 2) -> list[dict]:
    inst = ctx.state.instance
    s = inst.session_by_id[c.scope.session]
    cal = inst.calendar
    days = c.when.days or []
    d0 = days[0] if days else None
    q0 = c.when.slots[0] if c.when.slots else None

    def fits(q: int) -> bool:
        return q + s.duration <= cal.slots_per_day and (cal.lunch_slot is None or cal.lunch_slot not in
                                                        range(q, q + s.duration))

    cands: list[tuple[str, int | None]] = []
    if q0 is None:
        cands = [(d, None) for d in cal.days if d != d0]
    else:
        near = sorted((q for q in range(cal.slots_per_day) if fits(q) and q != q0), key=lambda q: (abs(q - q0), q))
        cands += [(d0, q) for q in near[:3]] if d0 else []
        cands += [(d, q0) for d in cal.days if d != d0 and fits(q0)]
    out: list[dict] = []
    for d, q in cands[:TRIES]:
        if _solves(ctx, c, d, q):
            out.append({"day": FULL_DAY[d], **({"hour": hour(q)} if q is not None else {"hour": "any"})})
            if len(out) == n:
                break
    return out


def works(ctx: Context, c: Constraint, day: str, slot: int | None) -> bool:
    return _solves(ctx, c, day, slot)


def _solves(ctx: Context, c: Constraint, day: str, slot: int | None) -> bool:
    world, base = ctx.state, ctx.world
    s = world.instance.session_by_id[c.scope.session]
    moved = c.model_copy(update={"when": When(days=[day], slots=None if slot is None else
                                              list(range(slot, slot + s.duration)))})
    rules = [x for x in world.constraints.values() if x.id != c.id] + [moved]
    for w in sorted(touched(world, moved), key=lambda x: 0 if x is None else x)[:WEEKS]:
        now, skip = base.timetable(w), world.timetable(w)
        if now is None or skip is None:
            continue
        solver = TimetableSolver(world.instance, rules, week=w, baseline=now.assignment, skip=skip.cancelled,
                                 time_limit=TIME_LIMIT, presolve=False)
        try:
            if not solver.solve().ok:
                return False
        except SolverTimeout:
            return False
    return True
