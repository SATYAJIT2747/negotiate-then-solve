"""Read tools: what the timetable says, in the department's names.

They read ``ctx.state`` (the episode's draft when there is one), take names, days and hours as people
write them, and answer the same way. A name they cannot place is refused with the closest names.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from nts.kernel.instance import Instance, Session
from nts.kernel.schemas import ConstraintType, Placement
from nts.kernel.semantics import in_when
from nts.state import World

from .names import FULL_DAY, Directory, hour
from .registry import Context, Refusal, Scope, tool

Week = Field(None, description="Week of term; leave out for the current week")


# -- what is where, in one week --------------------------------------------------


@dataclass
class _Placed:
    session: Session
    day: int
    slot: int
    room: str


class _Week:
    """Who and what is busy in one week: placed classes plus the hard unavailability in force."""

    def __init__(self, world: World, week: int | None) -> None:
        inst = world.instance
        tt = world.timetable(week)
        if tt is None:
            raise Refusal("no_timetable", "there is no timetable yet")
        self.instance, self.week = inst, week
        self.cancelled = sorted(tt.cancelled)
        self.placed = [_Placed(inst.session_by_id[sid], inst.day_index(p.day), p.slot, p.room)
                       for sid, p in tt.assignment.items() if sid not in tt.cancelled]
        self.busy: dict[tuple[str, str, int, int], str] = {}  # (kind, id, day, slot) -> session id
        for x in self.placed:
            for q in range(x.slot, x.slot + x.session.duration):
                self.busy[("teacher", x.session.faculty, x.day, q)] = x.session.id
                self.busy[("room", x.room, x.day, q)] = x.session.id
                for g in x.session.groups:
                    self.busy[("section", g, x.day, q)] = x.session.id
        self._away = defaultdict(list)  # (kind, id or None for everyone) -> hard unavailable constraints
        for c in world.active(week):
            if c.hard and c.type == ConstraintType.UNAVAILABLE:
                s = c.scope
                key = (("teacher", s.faculty) if s.faculty else ("section", s.group) if s.group
                       else ("room", s.room) if s.room else (None, None))
                self._away[key].append(c)

    def free(self, kind: str, id_: str, day: int, slot: int) -> bool:
        if (kind, id_, day, slot) in self.busy:
            return False
        rules = self._away[(None, None)] + self._away[(kind, id_)]
        return not any(in_when(self.instance, c.when, day, slot) for c in rules)

    def all_free(self, who: list[tuple[str, str]], day: int, start: int, length: int) -> bool:
        return all(self.free(k, i, day, q) for k, i in who for q in range(start, start + length))


def _week(ctx: Context, week: int | None) -> int | None:
    week = ctx.week if week is None else week
    n = ctx.state.instance.calendar.weeks
    if week is not None and not 1 <= week <= n:
        raise Refusal("unknown_week", f"week {week} is not in term", weeks=f"1-{n}")
    return week


def _days(d: Directory, days: list[str]) -> list[int]:
    inst = d.instance
    return [inst.day_index(d.day(x)) for x in days] if days else list(range(len(inst.calendar.days)))


def _class(inst: Instance, s: Session, day: str, slot: int, room: str) -> dict:
    return {"course": inst.course_title(s.course), "kind": s.kind.value,
            "teacher": inst.faculty_by_id[s.faculty].name,
            "sections": [inst.group_by_id[g].name for g in s.groups],
            "day": FULL_DAY[day], "from": hour(slot), "to": hour(slot + s.duration),
            "room": inst.room_by_id[room].name}


def _placed(inst: Instance, x: _Placed) -> dict:
    return _class(inst, x.session, inst.calendar.days[x.day], x.slot, x.room)


def _at(inst: Instance, s: Session, p: Placement) -> dict:
    return _class(inst, s, p.day, p.slot, p.room)


# -- the tools ------------------------------------------------------------------


class Find(BaseModel):
    text: str = Field(description="A name or part of one: 'Iyer', 'lab 2', 'machine learning'")


@tool(Scope.READ)
def find(ctx: Context, args: Find) -> list[dict]:
    """Which teachers, sections, rooms or courses a name could mean, with their names as this department writes them."""
    return Directory(ctx.state.instance).find(args.text)


class Describe(BaseModel):
    what: Literal["rooms", "sections", "teachers", "courses", "calendar"]


@tool(Scope.READ)
def describe(ctx: Context, args: Describe) -> list[dict] | dict:
    """Facts that do not change from week to week: the rooms (seats, equipment), the sections (sizes),
    the teachers and their courses, the courses and who teaches them, or the teaching days and hours."""
    inst = ctx.state.instance
    if args.what == "rooms":
        return [{"name": r.name, "kind": r.type.value, "seats": r.capacity, "equipment": r.equipment} for r in inst.rooms]
    if args.what == "sections":
        return [{"name": g.name, "students": g.size} for g in inst.groups]
    teaches: dict[str, set[str]] = defaultdict(set)
    for s in inst.sessions:
        teaches[s.faculty].add(s.course)
    if args.what == "teachers":
        return [{"name": f.name, "courses": sorted(inst.course_title(c) for c in teaches[f.id])} for f in inst.faculty]
    if args.what == "courses":
        by_course: dict[str, set[str]] = defaultdict(set)
        for s in inst.sessions:
            by_course[s.course].add(inst.faculty_by_id[s.faculty].name)
        return [{"name": inst.course_title(c), "code": c, "teachers": sorted(t)} for c, t in sorted(by_course.items())]
    cal = inst.calendar
    return {"days": [FULL_DAY[d] for d in cal.days], "hours": [hour(q) for q in range(cal.slots_per_day)],
            "lunch": hour(cal.lunch_slot) if inst.policy.lunch_break and cal.lunch_slot is not None else None,
            "weeks": cal.weeks, "current_week": ctx.week}


class ClassesOf(BaseModel):
    teacher: str | None = Field(None, description="A teacher's name, or 'me'")
    section: str | None = None
    room: str | None = None
    course: str | None = None
    days: list[str] = Field(default_factory=list, description="Only these days; empty for the whole week")
    week: int | None = Week


@tool(Scope.READ)
def classes_of(ctx: Context, args: ClassesOf) -> dict:
    """The classes of a teacher, a section, a room or a course (give one or more; all must match), in one week."""
    if not (args.teacher or args.section or args.room or args.course):
        raise Refusal("bad_arguments", "name a teacher, a section, a room or a course")
    d = Directory(ctx.state.instance)
    teacher = d.teacher(args.teacher, ctx.actor).id if args.teacher else None
    section = d.section(args.section).id if args.section else None
    room = d.room(args.room).id if args.room else None
    courses = d.courses(args.course) if args.course else None
    days = set(_days(d, args.days))
    w = _Week(ctx.state, _week(ctx, args.week))

    def wanted(s: Session) -> bool:
        return ((teacher is None or s.faculty == teacher) and (section is None or section in s.groups)
                and (courses is None or s.course in courses))

    held = sorted((x for x in w.placed if wanted(x.session) and x.day in days and room in (None, x.room)),
                  key=lambda x: (x.day, x.slot))
    out = {"week": w.week, "classes": [_placed(d.instance, x) for x in held]}
    if off := [d.instance.course_title(s.course) for sid in w.cancelled if wanted(s := d.instance.session_by_id[sid])]:
        out["not_held"] = off
    return out


class FreeHours(BaseModel):
    teachers: list[str] = Field(default_factory=list, description="Teachers' names, or 'me'")
    sections: list[str] = Field(default_factory=list)
    rooms: list[str] = Field(default_factory=list)
    days: list[str] = Field(default_factory=list, description="Only these days; empty for the whole week")
    length: int = Field(1, ge=1, description="Hours needed in one block")
    week: int | None = Week


@tool(Scope.READ)
def free_hours(ctx: Context, args: FreeHours) -> dict:
    """The hours when every given teacher, section and room is free (no class, not away, not lunch), as start times
    of blocks of ``length`` hours."""
    d = Directory(ctx.state.instance)
    who = ([("teacher", d.teacher(t, ctx.actor).id) for t in args.teachers]
           + [("section", d.section(s).id) for s in args.sections] + [("room", d.room(r).id) for r in args.rooms])
    if not who:
        raise Refusal("bad_arguments", "name at least one teacher, section or room")
    w = _Week(ctx.state, _week(ctx, args.week))
    cal = d.instance.calendar
    out = {}
    for day in _days(d, args.days):
        starts = [hour(q) for q in range(cal.slots_per_day - args.length + 1) if w.all_free(who, day, q, args.length)]
        out[FULL_DAY[cal.days[day]]] = starts
    return {"week": w.week, "length": args.length, "free": out}


class WhoIsFree(BaseModel):
    day: str
    hour: str = Field(description="'10 am', '2 pm'")
    of: Literal["teachers", "sections"] = "teachers"
    course: str | None = Field(None, description="Only teachers of this course (e.g. to cover a class)")
    week: int | None = Week


@tool(Scope.READ)
def who_is_free(ctx: Context, args: WhoIsFree) -> dict:
    """Which teachers (or sections) are free at one hour."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    day, slot = inst.day_index(d.day(args.day)), d.slot(args.hour)
    w = _Week(ctx.state, _week(ctx, args.week))
    if args.of == "sections":
        names = [g.name for g in inst.groups if w.free("section", g.id, day, slot)]
    else:
        courses = d.courses(args.course) if args.course else None
        pool = {s.faculty for s in inst.sessions if s.course in courses} if courses else set(inst.faculty_by_id)
        names = [f.name for f in inst.faculty if f.id in pool and w.free("teacher", f.id, day, slot)]
    return {"week": w.week, "day": FULL_DAY[inst.calendar.days[day]], "hour": hour(slot), "free": names}


class FreeRooms(BaseModel):
    day: str
    hour: str = Field(description="Start time: '10 am', '2 pm'")
    length: int = Field(1, ge=1, description="Hours needed")
    kind: Literal["lecture", "lab"] | None = None
    min_seats: int | None = None
    equipment: list[str] = Field(default_factory=list)
    week: int | None = Week


@tool(Scope.READ)
def free_rooms(ctx: Context, args: FreeRooms) -> dict:
    """Which rooms are free for ``length`` hours from one hour, optionally of one kind, size or equipment."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    day, slot = inst.day_index(d.day(args.day)), d.slot(args.hour)
    if slot + args.length > inst.calendar.slots_per_day:
        raise Refusal("bad_arguments", f"{args.length} hours from {hour(slot)} runs past the end of the day")
    known = {e for r in inst.rooms for e in r.equipment}
    if unknown := [e for e in args.equipment if e not in known]:
        raise Refusal("unknown_equipment", f"no room has {', '.join(unknown)}", equipment=sorted(known))
    w = _Week(ctx.state, _week(ctx, args.week))
    rooms = [r for r in inst.rooms
             if (args.kind is None or r.type.value == args.kind) and r.capacity >= (args.min_seats or 0)
             and set(args.equipment) <= set(r.equipment) and w.all_free([("room", r.id)], day, slot, args.length)]
    return {"week": w.week, "day": FULL_DAY[inst.calendar.days[day]], "from": hour(slot),
            "to": hour(slot + args.length), "rooms": [{"name": r.name, "seats": r.capacity} for r in rooms]}


class Changes(BaseModel):
    week: int | None = Week
    teacher: str | None = Field(None, description="Only this teacher's classes ('me' for the person asking)")
    section: str | None = None


@tool(Scope.READ)
def changes(ctx: Context, args: Changes) -> dict:
    """How one week differs from the usual timetable: classes moved and classes not held."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    week = _week(ctx, args.week)
    if week is None:
        raise Refusal("bad_arguments", "say which week")
    teacher = d.teacher(args.teacher, ctx.actor).id if args.teacher else None
    section = d.section(args.section).id if args.section else None
    usual, this = ctx.state.timetable(None), ctx.state.timetable(week)
    if usual is None or this is None:
        raise Refusal("no_timetable", "there is no timetable yet")

    def wanted(sid: str) -> bool:
        s = inst.session_by_id[sid]
        return (teacher is None or s.faculty == teacher) and (section is None or section in s.groups)

    moved = [{"class": _at(inst, inst.session_by_id[sid], usual.assignment[sid]),
              "now": _at(inst, inst.session_by_id[sid], p)}
             for sid, p in sorted(this.assignment.items())
             if wanted(sid) and sid not in this.cancelled and usual.assignment.get(sid) != p]
    off = [_at(inst, inst.session_by_id[sid], usual.assignment[sid]) for sid in sorted(this.cancelled) if wanted(sid)]
    return {"week": week, "moved": moved, "not_held": off}


READ_TOOLS = [find, describe, classes_of, free_hours, who_is_free, free_rooms, changes]
