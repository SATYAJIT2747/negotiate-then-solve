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
    text = _weeks(c)
    if c.valid.except_weeks:
        text += " except week" + ("s " if len(c.valid.except_weeks) > 1 else " ") + joined(
            [str(w) for w in sorted(c.valid.except_weeks)])
    return text


def _weeks(c: Constraint) -> str:
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


def _sessions(inst: Instance, ids: list[str] | None) -> str:
    """The classes of a relation, by course and kind: 'the Database Systems lectures'."""
    groups: dict[tuple[str, str], int] = {}
    for sid in ids or []:
        s = inst.session_by_id[sid]
        key = (inst.course_title(s.course), s.kind.value)
        groups[key] = groups.get(key, 0) + 1
    return joined([f"the {c} {k}{'s' * (n > 1)}" for (c, k), n in groups.items()])


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
    elif c.type == ConstraintType.LOAD:
        text = f"{who(inst, c.scope)}: " + (f"at least {c.limit} free hour{'s' * (c.limit != 1)} between classes"
                                            if c.measure == "gap" else f"at most {c.limit} " + {
            "classes_per_day": "classes a day", "hours_per_day": "hours a day",
            "days_per_week": "teaching days a week"}[c.measure]) + f", {weeks(c)}"
    elif c.type == ConstraintType.RELATE:
        rel = {"same_day": "on the same day as", "different_day": "on a different day from",
               "right_after": "right after", "after": "later the same day than",
               "day_gap": "more than a day apart from"}[c.relation]
        others = "each other" if c.others is None else ("one of " if c.any_other else "") + _sessions(inst, c.others)
        text = f"{_sessions(inst, c.sessions)}: {'not ' * c.negate}{rel} {others}, {weeks(c)}"
    elif c.type in (ConstraintType.ANY_OF, ConstraintType.RANKED):
        parts = [constraint(inst, m).rsplit(" (", 1)[0] for m in c.members or []]
        text = ("in order of preference: " + "; else ".join(parts) if c.type == ConstraintType.RANKED
                else "one of: " + "; or ".join(parts))
    else:
        text = f"{who(inst, c.scope)}: at most {c.limit} hours in a row"
    if c.source.rule:
        return f"{text} (rule: {c.source.rule})"
    return f"{text} ({'firm' if c.hard else 'if possible'})"
