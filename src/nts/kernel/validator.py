"""Deterministic checks that never call the solver or an LLM.

``validate_constraint`` rejects compiled constraints that reference unknown
entities or impossible times (proposal Section 8.2, step 4).
``verify_timetable`` independently checks a finished timetable; the
evaluation uses it to score hard-constraint validity for every system,
including the LLM-only baseline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from .instance import Instance
from .schemas import DAYS, Constraint, ConstraintType, Placement
from .semantics import (
    PLACEMENT_TYPES,
    consecutive_entities,
    occupied,
    room_compatible,
    sessions_in_scope,
    violates,
)


def validate_constraint(c: Constraint, instance: Instance) -> list[str]:
    errors: list[str] = []
    cal = instance.calendar
    lookups = {
        "faculty": instance.faculty_by_id,
        "group": instance.group_by_id,
        "session": instance.session_by_id,
        "room": instance.room_by_id,
        "course": {s.course for s in instance.sessions},
    }
    for kind, value in c.scope.model_dump().items():
        if value is not None and value not in lookups[kind]:
            errors.append(f"unknown {kind} {value!r}")
    if c.owner is not None and c.owner not in instance.faculty_by_id:
        errors.append(f"unknown owner {c.owner!r}")
    for d in c.when.days or []:
        if d not in cal.days:
            errors.append(f"day {d!r} is not a teaching day")
    for q in c.when.slots or []:
        if not 0 <= q < cal.slots_per_day:
            errors.append(f"slot {q} outside 0..{cal.slots_per_day - 1}")
    weeks = list(c.when.weeks or []) + [w for w in (c.valid.from_week, c.valid.to_week) if w is not None]
    for w in weeks:
        if not 1 <= w <= cal.weeks:
            errors.append(f"week {w} outside 1..{cal.weeks}")
    if c.room is not None:
        for r in c.room.rooms or []:
            if r not in instance.room_by_id:
                errors.append(f"unknown room {r!r}")
        known = {e for r in instance.rooms for e in r.equipment}
        for e in c.room.equipment:
            if e not in known:
                errors.append(f"no room has equipment {e!r}")
    return errors


def verify_timetable(
    instance: Instance,
    assignment: Mapping[str, Placement],
    constraints: Iterable[Constraint] = (),
    week: int | None = None,
) -> list[str]:
    """Every violated hard rule, as human-readable strings. Empty means valid."""
    out: list[str] = []
    cal = instance.calendar
    for sid in assignment:
        if sid not in instance.session_by_id:
            out.append(f"unknown session {sid}")

    room_use: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    fac_use: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    grp_use: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    placed: dict[str, tuple[int, int, str]] = {}
    for s in instance.sessions:
        pl = assignment.get(s.id)
        if pl is None:
            out.append(f"{s.id} not scheduled")
            continue
        if pl.day not in cal.days or pl.day not in DAYS:
            out.append(f"{s.id} on non-teaching day {pl.day}")
            continue
        if pl.slot < 0 or pl.slot + s.duration > cal.slots_per_day:
            out.append(f"{s.id} runs outside the teaching day")
            continue
        room = instance.room_by_id.get(pl.room)
        if room is None:
            out.append(f"{s.id} in unknown room {pl.room}")
            continue
        if not room_compatible(instance, s, room):
            out.append(f"{s.id} in incompatible room {room.id}")
        d = instance.day_index(pl.day)
        placed[s.id] = (d, pl.slot, room.id)
        for dd, q in occupied(s, d, pl.slot):
            room_use[(room.id, dd, q)].append(s.id)
            fac_use[(s.faculty, dd, q)].append(s.id)
            for g in s.groups:
                grp_use[(g, dd, q)].append(s.id)

    for label, use in (("room", room_use), ("faculty", fac_use), ("group", grp_use)):
        for (eid, d, q), sids in use.items():
            if len(sids) > 1:
                out.append(f"{label} {eid} double-booked {cal.days[d]} slot {q}: {', '.join(sids)}")

    for c in constraints:
        if not c.hard or not c.active_in(week):
            continue
        if c.type in PLACEMENT_TYPES:
            for s in sessions_in_scope(instance, c.scope):
                if s.id in placed:
                    d, p, r = placed[s.id]
                    if violates(instance, c, s, d, p, instance.room_by_id[r]):
                        out.append(f"{c.id} violated by {s.id}")
        elif c.type == ConstraintType.MAX_CONSECUTIVE:
            assert c.limit is not None
            for kind, eid in consecutive_entities(instance, c.scope):
                use = fac_use if kind == "faculty" else grp_use
                for d in range(len(cal.days)):
                    run = 0
                    for q in range(cal.slots_per_day):
                        run = run + 1 if use.get((eid, d, q)) else 0
                        if run == c.limit + 1:
                            out.append(f"{c.id} violated by {kind} {eid} on {cal.days[d]}")
    return out
