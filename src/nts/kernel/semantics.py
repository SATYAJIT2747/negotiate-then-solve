"""What a constraint means for a placement.

The solver and the independent validator both read constraints through these
functions, so a constraint can never mean one thing to the solver and another
to the checker that scores it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from .instance import Instance, Room, Session
from .schemas import Constraint, ConstraintType, Placement, RoomRequirement, Scope, When

# Constraint types decided placement by placement; max_consecutive is not.
PLACEMENT_TYPES = frozenset(
    {
        ConstraintType.UNAVAILABLE,
        ConstraintType.AVOID,
        ConstraintType.PREFER,
        ConstraintType.REQUIRE_ROOM,
    }
)


def sessions_in_scope(instance: Instance, scope: Scope) -> list[Session]:
    if scope.session is not None:
        return [s for s in instance.sessions if s.id == scope.session]
    if scope.faculty is not None:
        return [s for s in instance.sessions if s.faculty == scope.faculty]
    if scope.group is not None:
        return [s for s in instance.sessions if scope.group in s.groups]
    if scope.course is not None:
        return [s for s in instance.sessions if s.course == scope.course]
    return list(instance.sessions)  # global or room-scoped


def occupied(session: Session, day: int, slot: int) -> Iterator[tuple[int, int]]:
    """(day, slot) pairs a session occupies when it starts at ``slot``."""
    for q in range(slot, slot + session.duration):
        yield day, q


def in_when(instance: Instance, when: When, day: int, slot: int) -> bool:
    if when.days is not None and instance.calendar.days[day] not in when.days:
        return False
    return when.slots is None or slot in when.slots


def room_satisfies(req: RoomRequirement, room: Room) -> bool:
    if req.rooms is not None and room.id not in req.rooms:
        return False
    if req.room_type is not None and room.type.value != req.room_type:
        return False
    return set(req.equipment) <= set(room.equipment)


def room_compatible(instance: Instance, session: Session, room: Room) -> bool:
    """Tier 0: room type, capacity and equipment the session needs."""
    return (
        room.type == session.room_type
        and room.capacity >= instance.session_size(session)
        and set(session.equipment) <= set(room.equipment)
    )


def violates(
    instance: Instance, c: Constraint, session: Session, day: int, slot: int, room: Room
) -> bool:
    """Whether placing ``session`` at (day, slot, room) breaks placement constraint ``c``.

    Only meaningful for sessions in ``c``'s scope and for PLACEMENT_TYPES.
    """
    if c.type in (ConstraintType.UNAVAILABLE, ConstraintType.AVOID):
        if c.scope.room is not None and room.id != c.scope.room:
            return False
        return any(in_when(instance, c.when, d, q) for d, q in occupied(session, day, slot))
    if c.type == ConstraintType.PREFER:
        return not all(in_when(instance, c.when, d, q) for d, q in occupied(session, day, slot))
    if c.type == ConstraintType.REQUIRE_ROOM:
        assert c.room is not None
        return not room_satisfies(c.room, room)
    raise ValueError(f"{c.type.value} is not decided per placement")


def consecutive_entities(instance: Instance, scope: Scope) -> list[tuple[str, str]]:
    """(kind, id) pairs a max_consecutive constraint limits."""
    if scope.faculty is not None:
        return [("faculty", scope.faculty)]
    if scope.group is not None:
        return [("group", scope.group)]
    return [("faculty", f.id) for f in instance.faculty] + [("group", g.id) for g in instance.groups]


def placement_indices(instance: Instance, p: Placement) -> tuple[int, int]:
    return instance.day_index(p.day), p.slot


# -- load limits, relations between classes, and choices between constraints ---------------------------
# A placement is (day index, start slot, room id), as the validator keeps them.
Placed = Mapping[str, tuple[int, int, str]]


def relation_key(relation: str, day: int, slot: int, room: str):
    """The part of a placement a relation looks at: the solver compares placements only by it."""
    if relation == "same_room":
        return room
    if relation == "same_hour":
        return slot
    if relation in ("right_after", "after"):
        return day, slot
    return day


def relation_holds(c: Constraint, ka, kb, other: Session) -> bool:
    """Whether a class at key ``ka`` stands in ``c``'s relation to ``other`` at ``kb`` (``relation_key``):
    the same day, a different day, starting as ``other`` ends, after it on the same day, more than ``amount``
    days apart, at the same hour, in the same room; ``negate`` turns it round."""
    rel = c.relation
    if rel in ("same_day", "same_hour", "same_room"):
        ok = ka == kb
    elif rel == "different_day":
        ok = ka != kb
    elif rel == "day_gap":
        ok = abs(ka - kb) > (c.amount or 1)
    elif rel == "right_after":
        ok = ka[0] == kb[0] and ka[1] == kb[1] + other.duration
    elif rel == "after":
        ok = ka[0] == kb[0] and ka[1] >= kb[1] + other.duration
    else:
        raise ValueError(f"unknown relation {rel!r}")
    return ok != c.negate


def related(instance: Instance, c: Constraint) -> list[tuple[Session, list[Session]]]:
    """Each class a relate constraint places, with the classes it is placed against (each other, when it
    names none: "no two lectures of the course on the same day")."""
    by_id = instance.session_by_id
    mine = [by_id[s] for s in c.sessions or [] if s in by_id]
    others = [by_id[s] for s in c.others if s in by_id] if c.others is not None else mine
    return [(a, [b for b in others if b.id != a.id]) for a in mine]


def touched(instance: Instance, c: Constraint) -> set[str]:
    """The sessions ``c`` is about: what a repair must be free to move when ``c`` is broken."""
    if c.members:
        return set().union(*(touched(instance, m) for m in c.members))
    if c.type == ConstraintType.RELATE:
        return {a.id for a, bs in related(instance, c)} | {b.id for _a, bs in related(instance, c) for b in bs}
    if c.type == ConstraintType.MAX_CONSECUTIVE and c.scope.is_global:
        return set()
    return {s.id for s in sessions_in_scope(instance, c.scope)}


def load_value(instance: Instance, c: Constraint, placed: Placed) -> bool:
    """Whether the classes in ``c``'s scope keep its load limit."""
    days: dict[int, list[tuple[int, int]]] = {}
    for s in sessions_in_scope(instance, c.scope):
        if s.id in placed:
            d, q, _ = placed[s.id]
            days.setdefault(d, []).append((q, q + s.duration))
    limit = c.limit or 0
    if c.measure == "classes_per_day":
        return all(len(v) <= limit for v in days.values())
    if c.measure == "hours_per_day":
        return all(sum(e - b for b, e in v) <= limit for v in days.values())
    if c.measure == "days_per_week":
        return len(days) <= limit
    if c.measure == "gap":
        for v in days.values():
            v.sort()
            if any(b2 - e1 < limit for (_b1, e1), (b2, _e2) in zip(v, v[1:])):
                return False
        return True
    raise ValueError(f"unknown measure {c.measure!r}")


def satisfied(instance: Instance, c: Constraint, placed: Placed, week: int | None = None) -> bool:
    """Whether ``placed`` meets ``c``, whatever its type (the validator and the neighbourhood ask this; the
    solver encodes the same reading). Classes not placed are not judged."""
    if c.members:
        live = [m for m in c.members if m.active_in(week)] or c.members
        return any(satisfied(instance, m, placed, week) for m in live)
    if c.type in PLACEMENT_TYPES:
        return not any(violates(instance, c, s, *placed[s.id][:2], instance.room_by_id[placed[s.id][2]])
                       for s in sessions_in_scope(instance, c.scope) if s.id in placed)
    if c.type == ConstraintType.MAX_CONSECUTIVE:
        cover: dict[tuple[str, int], set[int]] = {}
        for kind, eid in consecutive_entities(instance, c.scope):
            for s in instance.sessions:
                if s.id in placed and ((kind == "faculty" and s.faculty == eid) or (kind == "group" and eid in s.groups)):
                    for d, q in occupied(s, *placed[s.id][:2]):
                        cover.setdefault((eid, d), set()).add(q)
        return all(not all(q + i in qs for i in range(c.limit + 1)) for qs in cover.values() for q in qs)
    if c.type == ConstraintType.LOAD:
        return load_value(instance, c, placed)
    if c.type == ConstraintType.RELATE:
        for a, bs in related(instance, c):
            if a.id not in placed:
                continue
            ka = relation_key(c.relation, *placed[a.id])
            held = [relation_holds(c, ka, relation_key(c.relation, *placed[b.id]), b) for b in bs if b.id in placed]
            if held and not (any(held) if c.any_other else all(held)):
                return False
        return True
    raise ValueError(f"unknown constraint type {c.type.value}")
