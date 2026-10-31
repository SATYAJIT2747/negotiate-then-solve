"""Extend tools (from coordinator-v5): what a request can ask beyond rules and moves.

    cancel_class   a class not held in one week (the solver skips it there)
    add_class      a make-up or extra class in one week, at a time when the teacher, the sections and a
                   suitable room are all free and no firm rule is broken; nothing else moves
    limit_hours    at most so many hours of classes in a row, for a teacher or a section
    book_room      a room for a club or an event, checked like the club desk checks it

Each is one numbered change in the draft, checked by ``check_draft`` and approved like any other, so
nothing here reaches the timetable without the office. A make-up class is not a new session in the
department: it is firm blocks for that week (the teacher, each section and the room are taken then),
as the fine-tuned pipeline keeps them, so later repairs of the week keep clear of it.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from nts.kernel.instance import Session
from nts.kernel.schemas import (
    Constraint,
    ConstraintType,
    Placement,
    Source,
    Tier,
    Validity,
    When,
)
from nts.kernel.schemas import Scope as Who
from nts.kernel.semantics import (
    PLACEMENT_TYPES,
    consecutive_entities,
    occupied,
    room_compatible,
    violates,
)
from nts.state import ClassCancelled

from .draft import (
    CANCEL,
    OFFICE,
    _changes,
    _draft,
    _may,
    _pick,
    _record,
    _tier,
    _validity,
)
from .names import FULL_DAY, Directory, hour
from .read import _class
from .registry import Context, Refusal, Scope, tool

EXTRA = "extra class: "  # Source.rule of a make-up class's blocks


class CancelClass(BaseModel):
    course: str
    kind: Literal["lecture", "tutorial", "practical"] | None = None
    section: str | None = None
    on_day: str | None = Field(None, description="Where it is now (to pick one class)")
    at_hour: str | None = None
    week: int = Field(description="The week it is not held")


@tool(Scope.EXTEND)
def cancel_class(ctx: Context, args: CancelClass) -> dict:
    """Cancel one class for one week: it is not held then. A make-up can be added with add_class."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    _validity(ctx, args.week, args.week)
    tt = ctx.state.timetable(args.week)
    if tt is None:
        raise Refusal("no_timetable", "there is no timetable yet")
    s = _pick(ctx, d, tt, args.course, args.kind, args.section, None, args.on_day, args.at_hour)
    _may(ctx, Who(session=s.id))
    if s.id in tt.cancelled:
        raise Refusal("already_cancelled", f"that class is already not held in week {args.week}")
    _draft(ctx).add(ClassCancelled(week=args.week, session=s.id))
    changes = _changes(ctx)
    n = max(changes, default=0) + 1
    changes[n] = [f"{CANCEL}{args.week}:{s.id}"]
    p = tt.assignment[s.id]
    return {"change": n, "cancelled": f"the {inst.course_title(s.course)} {s.kind.value} on {FULL_DAY[p.day]} at "
                                      f"{hour(p.slot)}, week {args.week}"}


class AddClass(BaseModel):
    course: str
    kind: Literal["lecture", "tutorial", "practical"] | None = None
    section: str | None = None
    week: int
    day: str | None = Field(None, description="None: any day with a free hour")
    hour: str | None = Field(None, description="None: any free hour")
    reason: str | None = None


@tool(Scope.EXTEND)
def add_class(ctx: Context, args: AddClass) -> dict:
    """Add a make-up or extra class (like one of the teacher's weekly classes) in one week, at a time when the
    teacher, the sections and a room are free. Nothing else moves; if the time asked is not free, it says
    which times are."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    _validity(ctx, args.week, args.week)
    tt = ctx.state.timetable(args.week)
    if tt is None:
        raise Refusal("no_timetable", "there is no timetable yet")
    s = _template(ctx, d, args)
    _may(ctx, Who(session=s.id))
    days = [d.day(args.day)] if args.day else None
    slots = [d.slot(args.hour)] if args.hour else None
    found = free_placements(ctx.state, s, args.week, days, slots)
    if not found:  # the times that do work, the day asked for first, so the requester is offered real ones
        every = free_placements(ctx.state, s, args.week)
        that_day = [p for p in every if days and p.day in days]
        free = list(dict.fromkeys(f"{FULL_DAY[p.day]} {hour(p.slot)}" for p in that_day + every))[:5]
        raise Refusal("no_free_time", "the teacher, a section or every suitable room is taken then, or a firm rule "
                      "forbids it" + (f"; free for this class: {', '.join(free)}. Ask the requester which one" if free
                                      else "; no time is free for it this week"), free=free)
    p = found[0]
    owner = ctx.actor if ctx.actor != OFFICE else s.faculty
    rule = f"{EXTRA}{inst.course_title(s.course)} {s.kind.value}"
    when = When(days=[p.day], slots=list(range(p.slot, p.slot + s.duration)))
    blocks = [Constraint(id="new", type=ConstraintType.UNAVAILABLE, hard=True, tier=Tier.COMMITMENT, owner=owner,
                         scope=scope, when=when, valid=Validity(from_week=args.week, to_week=args.week),
                         source=Source(rule=rule))
              for scope in [Who(faculty=s.faculty), *(Who(group=g) for g in s.groups), Who(room=p.room)]]
    result = _record(ctx, blocks, args.reason)
    out = {"change": result["change"], "extra_class": {**_class(inst, s, p.day, p.slot, p.room), "week": args.week}}
    named = [x for x in inst.calendar.days if re.search(rf"\b{FULL_DAY[x]}", ctx.request or "", re.IGNORECASE)]
    if args.day is None and len(named) == 1 and named[0] != p.day:
        out["note"] = (f"the request says {FULL_DAY[named[0]]}, and this class is on {FULL_DAY[p.day]} (no day was "
                       f"given); retract it and give day={FULL_DAY[named[0]]} if they asked for that day")
    elif named and p.day not in named:  # a day of its own choosing: the requester must agree before it is submitted
        out["note"] = (f"the request asks for {' or '.join(FULL_DAY[x] for x in named)}, not {FULL_DAY[p.day]}: "
                       "ask the requester whether this day is all right before you submit")
    return out


def _template(ctx: Context, d: Directory, args: AddClass) -> Session:
    """The weekly class an extra one copies: any of the teacher's classes of that course, kind and section
    (they are alike); a question when the course has classes of more than one kind or section."""
    inst = d.instance
    courses = d.courses(args.course)
    who = ctx.actor if ctx.actor != OFFICE else None
    sec = d.section(args.section).id if args.section else None
    found = [s for s in inst.sessions if s.course in courses and (who is None or s.faculty == who)
             and (args.kind is None or s.kind.value == args.kind) and (sec is None or sec in s.groups)]
    if not found:
        raise Refusal("no_such_class", f"no {inst.course_title(next(iter(courses)))} class matches"
                      + (" among your classes" if who else ""))
    kinds = {(s.kind.value, tuple(s.groups), s.duration) for s in found}
    if len(kinds) > 1:  # name only what differs, so the question to the requester is the one that matters
        sections = {gs for _, gs, _ in kinds}
        where = lambda gs: ", ".join(inst.group_by_id[g].name for g in gs)  # noqa: E731
        options = sorted({f"the {h}-hour {k}" + (f" for {where(gs)}" if len(sections) > 1 else "")
                          for k, gs, h in kinds})
        only = f" (all for {where(next(iter(sections)))})" if len(sections) == 1 else ""
        raise Refusal("ambiguous_class", f"{inst.course_title(found[0].course)} has {' and '.join(options)}{only}: give "
                      + ("kind" if len(sections) == 1 else "kind and section") + ", or ask the requester which one",
                      candidates=options)
    return found[0]


def free_placements(world, s: Session, week: int, days: list[str] | None = None,
                    slots: list[int] | None = None) -> list[Placement]:
    """Where ``s`` (a copy of it) can be held in ``week`` without moving anything, best first: its usual room,
    then the earliest time. The teacher, its sections and the room must be free, and no firm rule broken."""
    inst, cal = world.instance, world.instance.calendar
    tt = world.timetable(week)
    busy: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for sid, p in tt.assignment.items():
        if sid in tt.cancelled:
            continue
        x = inst.session_by_id[sid]
        cells = set(occupied(x, inst.day_index(p.day), p.slot))
        for key in [("faculty", x.faculty), ("room", p.room), *(("group", g) for g in x.groups)]:
            busy.setdefault(key, set()).update(cells)
    hard = [c for c in world.constraints.values() if c.active_in(week) and c.hard and _covers(c.scope, s)]
    mine = [("faculty", s.faculty), *(("group", g) for g in s.groups)]
    usual = tt.assignment.get(s.id)
    rooms = sorted((r for r in inst.rooms if room_compatible(inst, s, r)),
                   key=lambda r: (usual is None or r.id != usual.room, r.capacity))
    out: list[Placement] = []
    for day in [x for x in cal.days if days is None or x in days]:
        di = inst.day_index(day)
        for q in [x for x in range(cal.slots_per_day) if slots is None or x in slots]:
            if q + s.duration > cal.slots_per_day:
                continue
            cells = set(occupied(s, di, q))
            if any(cells & busy.get(k, set()) for k in mine) or not _within_limits(inst, hard, s, di, cells, busy):
                continue
            for room in rooms:
                if cells & busy.get(("room", room.id), set()):
                    continue
                if any(c.type in PLACEMENT_TYPES and violates(inst, c, s, di, q, room) for c in hard):
                    continue
                out.append(Placement(day=day, slot=q, room=room.id))
                break
    return out


def _covers(scope: Who, s: Session) -> bool:
    if scope.faculty is not None:
        return scope.faculty == s.faculty
    if scope.group is not None:
        return scope.group in s.groups
    if scope.session is not None:
        return scope.session == s.id
    if scope.course is not None:
        return scope.course == s.course
    return True  # everyone, or a room (``violates`` checks the room)


def _within_limits(inst, hard: list[Constraint], s: Session, day: int, cells: set[tuple[int, int]],
                   busy: dict[tuple[str, str], set[tuple[int, int]]]) -> bool:
    """No limit on hours in a row is broken for the teacher or a section."""
    for c in hard:
        if c.type != ConstraintType.MAX_CONSECUTIVE or not c.limit:
            continue
        for ent in set(consecutive_entities(inst, c.scope)) & {("faculty", s.faculty), *(("group", g) for g in s.groups)}:
            taken = {q for dd, q in busy.get(ent, set()) | cells if dd == day}
            run = best = 0
            for q in range(inst.calendar.slots_per_day):
                run = run + 1 if q in taken else 0
                best = max(best, run)
            if best > c.limit:
                return False
    return True


class LimitHours(BaseModel):
    most_in_a_row: int = Field(ge=1, le=7, description="Hours of classes in a row, at most")
    teacher: str | None = Field(None, description="Name, or 'me'; none with no section: the person asking")
    section: str | None = None
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = None
    reason: str | None = None


@tool(Scope.EXTEND)
def limit_hours(ctx: Context, args: LimitHours) -> dict:
    """At most so many hours of classes in a row, for a teacher or a section (always firm)."""
    d = Directory(ctx.state.instance)
    if args.section:
        scope = Who(group=d.section(args.section).id)
    else:
        who = d.teacher(args.teacher, ctx.actor).id if args.teacher else ctx.actor
        if who == OFFICE:
            raise Refusal("bad_arguments", "say which teacher or section")
        scope = Who(faculty=who)
    _may(ctx, scope)
    tier, why = _tier(ctx, True, args.reason)
    c = Constraint(id="new", type=ConstraintType.MAX_CONSECUTIVE, hard=True, tier=tier, owner=ctx.actor, scope=scope,
                   limit=args.most_in_a_row, justification=why,
                   valid=_validity(ctx, args.from_week, args.to_week), source=Source())
    return _record(ctx, [c], args.reason)


class BookRoom(BaseModel):
    club: str = Field(description="The club or group it is for")
    purpose: str
    week: int
    day: str
    from_hour: str
    to_hour: str
    attendees: int = Field(ge=1)
    room: str | None = Field(None, description="None: the smallest free room that holds them")


@tool(Scope.EXTEND)
def book_room(ctx: Context, args: BookRoom) -> dict:
    """Book a room for a club or an event, in one week. It never moves a class: if no room is free then, it
    says why and what is free instead."""
    from nts.bookings import Booking, assess  # nts.bookings uses the tools package

    decision, c = assess(ctx.state, Booking(**args.model_dump()), by=ctx.actor)
    if c is None:
        failed = [x["detail"] for x in decision.checks if not x["ok"]]
        raise Refusal("booking_refused", "it cannot be booked: " + "; ".join(failed),
                      alternatives=decision.alternatives)
    result = _record(ctx, [c], f"{args.club}: {args.purpose}")
    result["room"] = decision.room
    if decision.status == "needs_approval":
        result["note"] = "a large event: the timetable office decides when it approves the draft"
    return result


EXTEND_TOOLS = [cancel_class, add_class, limit_hours, book_room]
