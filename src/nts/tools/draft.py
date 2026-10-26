"""Draft tools: change the episode's draft, one numbered change at a time.

A change is one or two kernel constraints written from names, days, hours and weeks. Nothing is solved
here: ``check_draft`` asks the solver what the draft does, and ``submit`` hands it on. Who may change
what is decided here too: a teacher changes only their own classes and times; the timetable office
(``actor == "coordinator"``) may set rules for anyone, a section or a room.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from nts.kernel.instance import Session
from nts.kernel.schemas import (
    Constraint,
    ConstraintType,
    Justification,
    RoomRequirement,
    Source,
    Tier,
    Validity,
    When,
)
from nts.kernel.schemas import (
    Scope as Who,
)
from nts.kernel.semantics import room_compatible
from nts.state import ConstraintAdded, ConstraintRemoved, Draft

from . import words
from .names import FULL_DAY, Directory, hour
from .read import _class
from .registry import Context, Refusal, Scope, tool

OFFICE = "coordinator"


def _draft(ctx: Context) -> Draft:
    if ctx.draft is None:
        raise Refusal("no_draft", "this episode has no draft to change")
    return ctx.draft


def _changes(ctx: Context) -> dict[int, list[str]]:
    """Change number -> the ids of the constraints it added (kept in the episode's scratch)."""
    return ctx.scratch.setdefault("changes", {})


def _record(ctx: Context, constraints: list[Constraint], request: str | None) -> dict:
    draft, changes = _draft(ctx), _changes(ctx)
    n = max(changes, default=0) + 1
    ids = []
    for i, c in enumerate(constraints):
        c = c.model_copy(update={"id": f"{draft.episode or 'draft'}-{n}{'abcdefgh'[i] if len(constraints) > 1 else ''}",
                                 "source": c.source.model_copy(update={"request": request})})
        try:
            draft.add(ConstraintAdded(constraint=c))
        except ValueError as e:
            raise Refusal("bad_change", str(e)) from None
        ids.append(c.id)
    changes[n] = ids
    inst = draft.world.instance
    return {"change": n, "added": [words.constraint(inst, draft.world.constraints[i]) for i in ids]}


def _validity(ctx: Context, from_week: int | None, to_week: int | None) -> Validity:
    n = ctx.state.instance.calendar.weeks
    for w in (from_week, to_week):
        if w is not None and not 1 <= w <= n:
            raise Refusal("unknown_week", f"week {w} is not in term", weeks=f"1-{n}")
    if from_week is not None and to_week is not None and from_week > to_week:
        raise Refusal("bad_arguments", f"week {from_week} comes after week {to_week}")
    return Validity(from_week=from_week, to_week=to_week)


def _slots(d: Directory, from_hour: str | None, to_hour: str | None) -> list[int] | None:
    if from_hour is None and to_hour is None:
        return None
    lo = d.slot(from_hour) if from_hour is not None else 0
    hi = d.slot(to_hour, end=True) if to_hour is not None else d.instance.calendar.slots_per_day
    if lo >= hi:
        raise Refusal("bad_arguments", f"{hour(lo)} is not before {hour(hi)}")
    return list(range(lo, hi))


def _tier(ctx: Context, hard: bool, reason: str | None) -> tuple[Tier, Justification]:
    if not hard:
        return Tier.PREFERENCE, Justification.NONE
    if ctx.actor == OFFICE:
        return Tier.COMMITMENT, Justification.NONE  # the office's own decisions bind
    if reason:
        return Tier.VERIFIED_UNAVAILABILITY, Justification.STATED
    return Tier.OPERATIONAL, Justification.NONE


def _may(ctx: Context, scope: Who) -> None:
    """A teacher may constrain only themselves, their courses and their classes."""
    if ctx.actor == OFFICE:
        return
    inst = ctx.state.instance
    taught = {s.course for s in inst.sessions if s.faculty == ctx.actor}
    mine = {c for c in taught if all(s.faculty == ctx.actor for s in inst.sessions if s.course == c)}  # not shared
    ok = (scope.faculty == ctx.actor or (scope.course in mine) or
          (scope.session is not None and inst.session_by_id[scope.session].faculty == ctx.actor))
    if not ok:
        raise Refusal("not_allowed", "a teacher can change only their own times and classes; "
                                     "the timetable office decides for other people, sections and rooms")


# -- the tools ------------------------------------------------------------------


class Constrain(BaseModel):
    rule: Literal["unavailable", "no_classes", "classes_only", "room_needed"] = Field(
        description="unavailable: cannot be there; no_classes: keep free; classes_only: only then; "
                    "room_needed: needs a kind of room")
    teacher: str | None = Field(None, description="Name, or 'me'")
    section: str | None = None
    room: str | None = Field(None, description="unavailable: the room is closed")
    course: str | None = None
    days: list[str] = Field(default_factory=list, description="Empty: every day")
    from_hour: str | None = Field(None, description="'2 pm'; none: from the start of the day")
    to_hour: str | None = Field(None, description="Until ('5 pm'); none: to the end of the day")
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = Field(None, description="= from_week for one week")
    firm: bool = Field(True, description="false for 'if possible', 'I'd prefer'")
    reason: str | None = Field(None, description="The person's reason, if given")
    room_kind: Literal["lecture", "lab"] | None = None
    equipment: list[str] = Field(default_factory=list)
    rooms: list[str] = Field(default_factory=list, description="room_needed: one of these")


@tool(Scope.DRAFT)
def constrain(ctx: Context, args: Constrain) -> dict:
    """Add a rule to the draft for one teacher, section, room or course: when they cannot or should not
    have classes, when classes must be, or what room they need."""
    d = Directory(ctx.state.instance)
    named = [x for x in (args.teacher, args.section, args.room, args.course) if x]
    if len(named) != 1:
        raise Refusal("bad_arguments", "name exactly one of teacher, section, room or course")
    scope = (Who(faculty=d.teacher(args.teacher, ctx.actor).id) if args.teacher else
             Who(group=d.section(args.section).id) if args.section else
             Who(room=d.room(args.room).id) if args.room else Who(course=d.course(args.course)))
    _may(ctx, scope)
    when = When(days=[d.day(x) for x in args.days] or None, slots=_slots(d, args.from_hour, args.to_hour))
    valid = _validity(ctx, args.from_week, args.to_week)
    owner = scope.faculty or (ctx.actor if ctx.actor != OFFICE else None)
    room = None
    if args.rule == "room_needed":
        if args.room:
            raise Refusal("bad_arguments", "for room_needed, give the room in 'rooms' and who needs it")
        if not (args.room_kind or args.equipment or args.rooms):
            raise Refusal("bad_arguments", "say which room, kind of room or equipment is needed")
        known = {e for r in d.instance.rooms for e in r.equipment}
        if unknown := [e for e in args.equipment if e not in known]:
            raise Refusal("unknown_equipment", f"no room has {', '.join(unknown)}", equipment=sorted(known))
        room = RoomRequirement(rooms=[d.room(r).id for r in args.rooms] or None, equipment=args.equipment,
                               room_type=args.room_kind)
        kind, hard = ConstraintType.REQUIRE_ROOM, args.firm
    elif args.rule == "unavailable":  # "unavailable, if possible" is a time kept free if possible
        kind, hard = (ConstraintType.UNAVAILABLE, True) if args.firm else (ConstraintType.AVOID, False)
    elif args.room:
        raise Refusal("bad_arguments", "for a room, only 'unavailable' (closed) applies")
    else:
        kind, hard = (ConstraintType.AVOID if args.rule == "no_classes" else ConstraintType.PREFER), args.firm
    tier, why = _tier(ctx, hard, args.reason)
    c = Constraint(id="new", type=kind, hard=hard, tier=tier, owner=owner, scope=scope, when=when, room=room,
                   justification=why, valid=valid, source=Source())
    return _record(ctx, [c], args.reason)


class MoveClass(BaseModel):
    course: str
    kind: Literal["lecture", "tutorial", "practical"] | None = None
    section: str | None = None
    teacher: str | None = None
    on_day: str | None = Field(None, description="Where it is now (to pick one class)")
    at_hour: str | None = None
    to_day: str
    to_hour: str | None = Field(None, description="None: any time that day")
    to_room: str | None = None
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = Field(None, description="= from_week for one week")
    firm: bool = True


@tool(Scope.DRAFT)
def move_class(ctx: Context, args: MoveClass) -> dict:
    """Move one class to another day (and hour, room) in the draft. Pick it by course and, if needed,
    kind, section, teacher or where it is now."""
    d = Directory(ctx.state.instance)
    inst = d.instance
    valid = _validity(ctx, args.from_week, args.to_week)
    week = args.from_week or ctx.week
    tt = ctx.state.timetable(week)
    if tt is None:
        raise Refusal("no_timetable", "there is no timetable yet")
    courses = d.courses(args.course)
    teacher = d.teacher(args.teacher, ctx.actor).id if args.teacher else (ctx.actor if ctx.actor != OFFICE else None)
    section = d.section(args.section).id if args.section else None
    on_day = d.day(args.on_day) if args.on_day else None
    at = d.slot(args.at_hour) if args.at_hour else None

    def fits(s: Session) -> bool:
        p = tt.assignment.get(s.id)
        return (s.course in courses and (args.kind is None or s.kind.value == args.kind)
                and (teacher is None or s.faculty == teacher) and (section is None or section in s.groups)
                and p is not None and (on_day is None or p.day == on_day)
                and (at is None or p.slot <= at < p.slot + s.duration))

    found = [s for s in inst.sessions if fits(s)]
    if not found:
        raise Refusal("no_such_class", f"no {inst.course_title(next(iter(courses)))} class matches"
                      + (" among your classes" if ctx.actor != OFFICE else ""))
    if len(found) > 1:
        rows = [_class(inst, s, tt.assignment[s.id].day, tt.assignment[s.id].slot, tt.assignment[s.id].room)
                for s in found]
        raise Refusal("ambiguous_class", f"{len(found)} classes match; say which (kind, section, day or hour)",
                      candidates=rows)
    s = found[0]
    scope = Who(session=s.id)
    _may(ctx, scope)
    day = d.day(args.to_day)
    slots = None
    if args.to_hour is not None:
        q = d.slot(args.to_hour)
        if q + s.duration > inst.calendar.slots_per_day:
            raise Refusal("bad_arguments", f"a {s.duration}-hour class from {hour(q)} runs past the end of the day")
        slots = list(range(q, q + s.duration))
    tier, why = _tier(ctx, args.firm, None)
    out = [Constraint(id="new", type=ConstraintType.PREFER, hard=args.firm, tier=tier, owner=s.faculty, scope=scope,
                      when=When(days=[day], slots=slots), justification=why, valid=valid)]
    if args.to_room:
        room = d.room(args.to_room)
        if not room_compatible(inst, s, room):
            raise Refusal("room_unsuitable", f"{room.name} cannot hold this class "
                          f"({inst.session_size(s)} students, a {s.room_type.value} room"
                          + (f" with {words.joined(s.equipment)}" if s.equipment else "") + ")")
        out.append(Constraint(id="new", type=ConstraintType.REQUIRE_ROOM, hard=args.firm, tier=tier, owner=s.faculty,
                              scope=scope, room=RoomRequirement(rooms=[room.id]), justification=why, valid=valid))
    result = _record(ctx, out, None)
    if args.to_hour is None and (asked := _hour_asked(ctx.request, day)):
        result["note"] = (f"the request says {FULL_DAY[day]} at {asked}, and this change has no to_hour, so the class "
                          f"may land at any hour that day; retract it and give to_hour if they asked for {asked}")
    return result


def _hour_asked(request: str, day: str) -> str | None:
    """The hour a request names right after ``day`` ("Tuesday at 12 pm"), if it names one."""
    m = re.search(rf"\b(?:{FULL_DAY[day]}|{day})\w*\b[^.;,]{{0,12}}?\b(\d{{1,2}}(?::\d\d)?\s*[ap]\.?m\b\.?|noon)",
                  request, re.IGNORECASE)
    return m.group(1) if m else None


class Retract(BaseModel):
    change: int = Field(description="The change number")


@tool(Scope.DRAFT)
def retract(ctx: Context, args: Retract) -> dict:
    """Take one of this draft's changes back out."""
    draft, changes = _draft(ctx), _changes(ctx)
    if args.change not in changes:
        raise Refusal("unknown_change", f"there is no change {args.change}", changes=sorted(changes))
    for cid in changes.pop(args.change):
        draft.add(ConstraintRemoved(id=cid))
    return {"retracted": args.change, "left": sorted(changes)}


class ListDraft(BaseModel):
    pass


@tool(Scope.DRAFT)
def list_draft(ctx: Context, args: ListDraft) -> list[dict]:
    """The changes in the draft so far, by number."""
    world = _draft(ctx).world
    return [{"change": n, "rules": [words.constraint(world.instance, world.constraints[i]) for i in ids]}
            for n, ids in sorted(_changes(ctx).items())]


DRAFT_TOOLS = [constrain, move_class, retract, list_draft]
