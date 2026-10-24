"""What a constraint means for a placement.

The solver and the independent validator both read constraints through these
functions, so a constraint can never mean one thing to the solver and another
to the checker that scores it.
"""

from __future__ import annotations

from collections.abc import Iterator

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
