"""A new semester timetable: solved from scratch with every standing rule and wish, then published.

    build = rebuild(store, from_week=5)        # a Draft (not committed) and what it changes
    build.stats   # classes moved, wishes met before and after, weeks re-repaired, weeks that kept their timetable
    store.commit(build.draft)                  # publishing: the kernel checks it like any change

During term, changes are repairs: each moves as few classes as it can, so wishes added one by one are
met only where a repair happened to allow it. A rebuild solves the semester timetable again from
nothing (no baseline to stay near), so it can meet more of them; the weeks from ``from_week`` on that
have a timetable of their own are then repaired on top of the new one, and earlier weeks stay as
they ran. Nothing changes until the office publishes the draft.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from nts.kernel.instance import Instance
from nts.kernel.schemas import Constraint, Placement
from nts.kernel.semantics import PLACEMENT_TYPES, sessions_in_scope, violates
from nts.kernel.solver import TimetableSolver
from nts.state import Draft, Store, TimetableSet
from nts.tools import words
from nts.tools.draft import OFFICE


@dataclass
class Rebuild:
    draft: Draft
    stats: dict
    unmet: list[str] = field(default_factory=list)  # the wishes the new timetable still does not meet, in words


def unmet(inst: Instance, assignment: Mapping[str, Placement], constraints: Iterable[Constraint],
          week: int | None = None) -> list[Constraint]:
    """The soft rules (wishes) in force in ``week`` that ``assignment`` breaks for some class."""
    out = []
    for c in constraints:
        if c.hard or not c.active_in(week) or c.type not in PLACEMENT_TYPES:
            continue
        for s in sessions_in_scope(inst, c.scope):
            p = assignment.get(s.id)
            if p is not None and violates(inst, c, s, inst.day_index(p.day), p.slot, inst.room_by_id[p.room]):
                out.append(c)
                break
    return out


def rebuild(store: Store, *, from_week: int | None = None, time_limit: float = 60.0, episode: str = "rebuild") -> Rebuild:
    world = store.world
    inst = world.instance
    cons = list(world.constraints.values())
    now = world.timetable(None)
    assert now is not None
    result = TimetableSolver(inst, cons, time_limit=time_limit).solve()
    if not result.ok:
        raise RuntimeError(f"no semester timetable: {result.status}")
    new = result.assignment
    assert new is not None
    draft = store.draft(OFFICE, episode)
    draft.add(TimetableSet(assignment=new))

    redone, kept = [], []
    for w, tt in sorted((w, t) for w, t in world.timetables.items() if w is not None):
        if from_week is not None and w < from_week:
            kept.append(w)  # it already ran: keep it as it was
            continue
        r = TimetableSolver(inst, cons, week=w, baseline=new, skip=tt.cancelled, time_limit=20, presolve=False).solve()
        if r.ok and r.assignment is not None:
            draft.add(TimetableSet(week=w, assignment=r.assignment, cancelled=sorted(tt.cancelled)))
            redone.append(w)
        else:
            kept.append(w)

    wishes = [c for c in cons if not c.hard and c.active_in(None) and c.type in PLACEMENT_TYPES]
    before, after = unmet(inst, now.assignment, cons), unmet(inst, new, cons)
    stats = {"classes": len(new), "moved": sum(now.assignment.get(s) != p for s, p in new.items()),
             "wishes": len(wishes), "met_before": len(wishes) - len(before), "met_after": len(wishes) - len(after),
             "weeks_repaired": redone, "weeks_kept": kept, "solver": result.status,
             "seconds": round(result.wall_time, 1)}
    return Rebuild(draft, stats, [words.constraint(inst, c) for c in after])
