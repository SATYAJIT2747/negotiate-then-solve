"""Constraints and times in words, for the model: "Dr. Iyer: not available on Tuesday, all day, week 7 (firm)"."""

from __future__ import annotations

from nts.kernel.instance import Instance
from nts.kernel.schemas import Constraint, ConstraintType, Scope

from .names import FULL_DAY, hour


def joined(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def hours(slots: list[int] | None) -> str:
    """[5, 6, 7] -> '2 pm to 5 pm'; runs that are not contiguous are listed one by one."""
    if not slots:
        return "all day"
    slots = sorted(slots)
    runs: list[list[int]] = [[slots[0]]]
    for q in slots[1:]:
        if q == runs[-1][-1] + 1:
            runs[-1].append(q)
        else:
            runs.append([q])
    return joined([f"{hour(r[0])} to {hour(r[-1] + 1)}" for r in runs])


def days(ds: list[str] | None) -> str:
    return "every day" if not ds else joined([FULL_DAY[d] for d in ds])


def weeks(c: Constraint) -> str:
    lo, hi = c.valid.from_week, c.valid.to_week
    if c.when.weeks:
        return "week " + joined([str(w) for w in c.when.weeks]) if len(c.when.weeks) == 1 else \
            "weeks " + joined([str(w) for w in c.when.weeks])
    if lo is None and hi is None:
        return "every week"
    if lo == hi:
        return f"week {lo}"
    if hi is None:
        return f"from week {lo}"
    if lo is None:
        return f"until week {hi}"
    return f"weeks {lo} to {hi}"


def who(inst: Instance, scope: Scope) -> str:
    if scope.faculty:
        return inst.faculty_by_id[scope.faculty].name
    if scope.group:
        return inst.group_by_id[scope.group].name
    if scope.room:
        return inst.room_by_id[scope.room].name
    if scope.course:
        return inst.course_title(scope.course)
    if scope.session:
        s = inst.session_by_id[scope.session]
        return f"the {inst.course_title(s.course)} {s.kind.value} of {inst.faculty_by_id[s.faculty].name}"
    return "everyone"


def constraint(inst: Instance, c: Constraint) -> str:
    when = f"{days(c.when.days)}, {hours(c.when.slots)}, {weeks(c)}"
    if c.type == ConstraintType.UNAVAILABLE:
        what = "closed" if c.scope.room else "not available"
        text = f"{who(inst, c.scope)}: {what} on {when}"
    elif c.type == ConstraintType.AVOID:
        text = f"{who(inst, c.scope)}: no classes on {when}"
    elif c.type == ConstraintType.PREFER:
        text = f"{who(inst, c.scope)}: classes only on {when}"
    elif c.type == ConstraintType.REQUIRE_ROOM:
        assert c.room is not None
        need = ([joined([inst.room_by_id[r].name for r in c.room.rooms])] if c.room.rooms else []) \
            + ([f"a {c.room.room_type} room"] if c.room.room_type else []) \
            + ([f"a room with {joined(c.room.equipment)}"] if c.room.equipment else [])
        text = f"{who(inst, c.scope)}: needs {' or '.join(need) or 'a room'}, {weeks(c)}"
    else:
        text = f"{who(inst, c.scope)}: at most {c.limit} hours in a row"
    if c.source.rule:
        return f"{text} (rule: {c.source.rule})"
    return f"{text} ({'firm' if c.hard else 'if possible'})"
