"""Request intents: what a message asks for, in a small closed vocabulary.

A parser (the fine-tuned model, Gemini or ``RuleParser``) only says what the
message means: which intent, which class or person, which times and weeks,
how strongly. It never does the arithmetic or the lookups. This module does
those, deterministically:

* ``TimeSpec`` -> slots: clock times ("before 10:00", "after 14:00",
  "morning") become slot indices of the calendar, so the parser never counts
  slots;
* ``Period`` -> weeks: explicit weeks, ranges, "from week 9", "next week"
  (against the current week);
* ``ClassRef`` -> a session ID: by ID, or by owner, course, kind, group and
  where the class is now in the timetable ("my Thursday 10 am lecture").

Anything that resolves to nothing, or to more than one thing where one is
needed, becomes a clarifying question rather than a guess. Intents that
compile to constraints go through the usual deterministic post-processing
(``parsing.postprocess``: tiers, authority, validation); swaps, extra
classes and cancellations come out as resolved references for their own
flows.

``lower`` turns an ``IntentOutput`` into the older ``ParseOutput`` wire
format, so parsers that emit intents feed everything built on that format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.instance import Instance
from core.schemas import (
    Constraint,
    ConstraintType,
    Justification,
    Placement,
    Request,
    Role,
    Scope,
    Source,
    Tier,
    Validity,
)
from core.validator import validate_constraint

from .corpus import FIRST_SLOT_HOUR, FULL_DAY, hour
from .parsing import (
    DraftConstraint,
    ParseOutput,
    ParseResult,
    check_authority,
    postprocess,
)

Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Clock = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", description="24-hour HH:MM")]
Kind = Literal["lecture", "tutorial", "practical"]
JustificationLit = Literal["none", "stated", "verified"]


# ---------------------------------------------------------------------------
# References: how a parser points at times, weeks, people and classes
# ---------------------------------------------------------------------------


class TimeSpec(BaseModel):
    """Days and times of day. Empty means all of them. ``after`` and
    ``before`` together are a range ("between 2 and 4 pm"); ``part`` and
    ``at`` stand alone."""

    days: list[Day] | None = Field(None, description="null means every teaching day")
    part: Literal["morning", "afternoon", "lunch"] | None = None
    before: Clock | None = Field(None, description="classes that start before this time")
    after: Clock | None = Field(None, description="classes that start at or after this time")
    at: list[Clock] | None = Field(None, description="classes that start at these times")

    @model_validator(mode="after")
    def _one_way(self) -> TimeSpec:
        ways = [self.part is not None, self.before is not None or self.after is not None, self.at is not None]
        if sum(ways) > 1:
            raise ValueError("give a part of the day, a before/after range or start times, not several")
        return self

    @property
    def has_times(self) -> bool:
        return any(v is not None for v in (self.part, self.before, self.after, self.at))

    @property
    def is_empty(self) -> bool:
        return self.days is None and not self.has_times


class Period(BaseModel):
    """Semester weeks. Empty means every week (a recurring request)."""

    weeks: list[int] | None = None
    from_week: int | None = None
    to_week: int | None = None
    relative: Literal["this_week", "next_week"] | None = None

    @model_validator(mode="after")
    def _one_way(self) -> Period:
        ways = [self.weeks is not None, self.from_week is not None or self.to_week is not None,
                self.relative is not None]
        if sum(ways) > 1:
            raise ValueError("give weeks, a from/to range or a relative week, not several")
        return self

    @property
    def is_recurring(self) -> bool:
        return self.weeks is None and self.from_week is None and self.to_week is None and self.relative is None


class Who(BaseModel):
    kind: Literal["faculty", "group", "course", "room", "session"]
    id: str


class ClassRef(BaseModel):
    """One class, by ID or by description. ``day`` and ``time`` say where it
    is now in the timetable, not where it should go."""

    session: str | None = Field(None, description="a session ID from the directory, when the message pins it down")
    owner: str | None = Field(None, description="faculty ID of whoever teaches it; null means the sender")
    course: str | None = None
    kind: Kind | None = None
    group: str | None = None
    day: Day | None = None
    time: Clock | None = None


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


def _kind_first(schema: dict) -> None:
    """``kind`` picks the intent, so a grammar built from the schema must write it, and write it first."""
    schema["required"] = ["kind", *[r for r in schema.get("required", []) if r != "kind"]]


class _Base(BaseModel):
    """Every intent: ``kind`` first, ``justification`` last (the order constrained decoding writes them in)."""

    model_config = ConfigDict(json_schema_extra=_kind_first)


JUSTIFICATION = Field("none", description="'stated' when a reason is given; never copy the reason")


class Unavailable(_Base):
    """Cannot teach (or a room cannot be used by them) at these times. Always hard."""

    kind: Literal["unavailable"] = "unavailable"
    who: Who | None = Field(None, description="null means the sender")
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    justification: JustificationLit = JUSTIFICATION


class Avoid(_Base):
    """Times they do not want ("no classes before 10", "keep Friday free")."""

    kind: Literal["avoid"] = "avoid"
    who: Who | None = None
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class Only(_Base):
    """The only times they want to teach ("only Mon-Thu"): every class in scope fits inside."""

    kind: Literal["only"] = "only"
    who: Who | None = None
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class Move(_Base):
    """Put one class at these times ("move my Thursday ML lecture before lunch"). Any weekly meeting of the
    class will do; with the timetable, the one already on the day asked for."""

    kind: Literal["move"] = "move"
    which: ClassRef
    to: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = True
    justification: JustificationLit = JUSTIFICATION


class Swap(_Base):
    """Exchange the slots of the sender's class and a colleague's."""

    kind: Literal["swap"] = "swap"
    mine: ClassRef
    theirs: ClassRef = Field(description="owner is the colleague's faculty ID")
    period: Period = Field(default_factory=Period)
    justification: JustificationLit = JUSTIFICATION


class ExtraClass(_Base):
    """A class on top of the weekly ones, copying one of the sender's classes."""

    kind: Literal["extra_class"] = "extra_class"
    which: ClassRef
    period: Period = Field(default_factory=Period)
    time: TimeSpec = Field(default_factory=TimeSpec)
    justification: JustificationLit = JUSTIFICATION


class Cancel(_Base):
    """A class not held in the weeks named."""

    kind: Literal["cancel"] = "cancel"
    which: ClassRef
    period: Period = Field(default_factory=Period)
    justification: JustificationLit = JUSTIFICATION


class RoomClosed(_Base):
    kind: Literal["room_closed"] = "room_closed"
    room: str
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    justification: JustificationLit = JUSTIFICATION


class NeedsRoom(_Base):
    """A class needs equipment or a particular room. Applies to every class the reference fits."""

    kind: Literal["needs_room"] = "needs_room"
    which: ClassRef
    equipment: list[str] = Field(default_factory=list)
    room: str | None = None
    justification: JustificationLit = JUSTIFICATION


class LoadLimit(_Base):
    """At most this many teaching hours in a row."""

    kind: Literal["load_limit"] = "load_limit"
    who: Who | None = None
    max_consecutive: int = Field(ge=1)
    justification: JustificationLit = JUSTIFICATION


DateText = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="YYYY-MM-DD")]


class ClubBooking(_Base):
    """A room for a club activity (the club desk checks and books it). Students may ask for one."""

    kind: Literal["club_booking"] = "club_booking"
    club: str | None = Field(None, description="the club's name as the message gives it")
    activity: str | None = None
    day: Day | None = None
    date: DateText | None = Field(None, description="only when the message gives a calendar date")
    period: Period = Field(default_factory=Period)
    start: Clock | None = None
    end: Clock | None = None
    attendees: int | None = Field(None, ge=1)
    cohorts: list[str] = Field(default_factory=list, description="programmes the members come from, as named")
    room: str | None = Field(None, description="the room as the message names it")
    justification: JustificationLit = JUSTIFICATION


class ClashReport(_Base):
    """Students report two of their classes at the same time: when, and which courses (with ``investigate``)."""

    kind: Literal["clash_report"] = "clash_report"
    time: TimeSpec = Field(default_factory=TimeSpec)
    courses: list[str] = Field(default_factory=list, description="course IDs from the directory")
    justification: JustificationLit = JUSTIFICATION


Intent = Annotated[Unavailable | Avoid | Only | Move | Swap | ExtraClass | Cancel | RoomClosed | NeedsRoom | LoadLimit
                   | ClubBooking | ClashReport, Field(discriminator="kind")]
IntentKind = Literal["unavailable", "avoid", "only", "move", "swap", "extra_class", "cancel", "room_closed",
                     "needs_room", "load_limit", "club_booking", "clash_report"]

REQUEST_TYPE: dict[str, str] = {
    "unavailable": "unavailability", "avoid": "preference", "only": "preference", "move": "preference",
    "swap": "swap", "extra_class": "preference", "cancel": "preference", "room_closed": "room_issue",
    "needs_room": "room_issue", "load_limit": "preference", "club_booking": "club_booking",
    "clash_report": "clash_report",
}


class IntentOutput(BaseModel):
    """What a parser returns. ``compile`` carries intents; ``clarify`` the
    missing fields and a question (``topic``: the intent it would be);
    the other actions are routed without intents."""

    action: Literal["compile", "clarify", "investigate", "answer", "out_of_scope", "withdraw"]
    intents: list[Intent] = Field(default_factory=list)
    topic: IntentKind | None = None
    missing: list[str] = Field(default_factory=list)
    question: str | None = None

    @property
    def request_type(self) -> str:
        if self.action == "investigate":
            return "clash_report"
        if self.action == "answer":
            return "policy_question"
        if self.action == "out_of_scope":
            return "out_of_scope"
        kind = self.intents[0].kind if self.intents else self.topic
        return REQUEST_TYPE.get(kind or "", "preference")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class Unresolved(ValueError):
    """A reference that fits nothing, or too much: ask the sender."""

    def __init__(self, question: str, missing: str) -> None:
        super().__init__(question)
        self.question = question
        self.missing = missing


def minutes(clock: str) -> int:
    h, m = clock.split(":")
    return int(h) * 60 + int(m)


def slot_start(slot: int) -> int:
    return (FIRST_SLOT_HOUR + slot) * 60


def _clock_text(clock: str) -> str:
    h, m = divmod(minutes(clock), 60)
    return f"{h % 12 or 12}{f':{m:02d}' if m else ''} {'am' if h < 12 else 'pm'}"


def slots(instance: Instance, spec: TimeSpec) -> list[int] | None:
    """The slots ``spec`` names; ``None`` when it names no times (all day)."""
    cal = instance.calendar
    every = range(cal.slots_per_day)
    lunch = cal.lunch_slot if cal.lunch_slot is not None else cal.slots_per_day // 2
    if spec.part == "morning":
        return list(range(lunch))
    if spec.part == "afternoon":
        return list(range(lunch + 1, cal.slots_per_day))
    if spec.part == "lunch":
        if cal.lunch_slot is None:
            raise Unresolved("This timetable has no lunch hour.", "slots")
        return [cal.lunch_slot]
    if spec.at is not None:
        starts = {slot_start(q): q for q in every}
        off = [t for t in spec.at if minutes(t) not in starts]
        if off:
            raise Unresolved(f"No class starts at {_clock_text(off[0])}; classes start on the hour from "
                             f"{hour(0)} to {hour(cal.slots_per_day - 1)}.", "slots")
        return sorted({starts[minutes(t)] for t in spec.at})
    if spec.before is None and spec.after is None:
        return None
    lo = minutes(spec.after) if spec.after else 0
    hi = minutes(spec.before) if spec.before else 24 * 60
    found = [q for q in every if lo <= slot_start(q) < hi]
    if not found:
        said = " and ".join(x for x in (spec.after and f"after {_clock_text(spec.after)}",
                                          spec.before and f"before {_clock_text(spec.before)}") if x)
        raise Unresolved(f"No class starts {said}; classes start on the hour from {hour(0)} to "
                         f"{hour(cal.slots_per_day - 1)}.", "slots")
    return found


def days(instance: Instance, spec: TimeSpec) -> list[str] | None:
    if spec.days is None:
        return None
    off = [d for d in spec.days if d not in instance.calendar.days]
    if off:
        raise Unresolved(f"There are no classes on {FULL_DAY[off[0]]}. Which day do you mean?", "days")
    order = instance.calendar.days
    return sorted(set(spec.days), key=order.index)


def weeks(instance: Instance, period: Period, current_week: int | None = None) -> list[int] | None:
    """The weeks ``period`` names; ``None`` for every week."""
    n = instance.calendar.weeks
    if period.relative is not None:
        if current_week is None:
            raise Unresolved("Which week do you mean?", "weeks")
        out = [current_week + (period.relative == "next_week")]
    elif period.weeks is not None:
        out = sorted(set(period.weeks))
    elif period.from_week is not None or period.to_week is not None:
        lo = period.from_week if period.from_week is not None else 1
        hi = period.to_week if period.to_week is not None else n
        out = list(range(min(lo, hi), max(lo, hi) + 1))
    else:
        return None
    if not out or out[0] < 1 or out[-1] > n:
        raise Unresolved(f"The semester has teaching weeks 1 to {n}. Which weeks do you mean?", "weeks")
    return out


def _describe(instance: Instance, sid: str, timetable: Mapping[str, Placement] | None) -> str:
    s = instance.session_by_id[sid]
    name = instance.course_title(s.course) + ("" if s.kind.value == "lecture" else f" {s.kind.value}")
    p = (timetable or {}).get(sid)
    return f"{name} on {FULL_DAY[p.day]} at {hour(p.slot)}" if p else name


def classes(instance: Instance, ref: ClassRef, sender: str,
            timetable: Mapping[str, Placement] | None = None) -> list[str]:
    """Every session ``ref`` fits (owner defaults to the sender)."""
    if ref.session is not None:
        if ref.session not in instance.session_by_id:
            raise Unresolved("I could not find that class. Which one do you mean?", "session")
        return [ref.session]
    owner = ref.owner or sender
    found = [s for s in instance.sessions if s.faculty == owner
             and (ref.course is None or s.course == ref.course)
             and (ref.kind is None or s.kind.value == ref.kind)
             and (ref.group is None or ref.group in s.groups)]
    if ref.day is not None or ref.time is not None:
        if timetable is None:
            raise Unresolved("Which class do you mean? Give the course and kind of class.", "session")
        found = [s for s in found if (p := timetable.get(s.id)) is not None
                 and (ref.day is None or p.day == ref.day)
                 and (ref.time is None or slot_start(p.slot) == minutes(ref.time))]
    if not found:
        whose = "your" if owner == sender else f"{instance.faculty_by_id[owner].name}'s" \
            if owner in instance.faculty_by_id else "those"
        raise Unresolved(f"I could not find that class among {whose} classes. Which one do you mean?", "session")
    return [s.id for s in found]


def one_class(instance: Instance, ref: ClassRef, sender: str,
              timetable: Mapping[str, Placement] | None = None) -> str:
    found = classes(instance, ref, sender, timetable)
    if len(found) == 1:
        return found[0]
    names = [_describe(instance, sid, timetable) for sid in found]
    if len(set(names)) == 1:  # the weekly meetings of one class, and no timetable to tell them apart
        raise Unresolved(f"You have {len(found)} {names[0]} classes a week. Which day's class do you mean?", "session")
    raise Unresolved(f"Which class do you mean: {'; '.join(names[:4])}?", "session")


def _one_kind(instance: Instance, found: list[str], question: str) -> list[str]:
    """``found`` when every session in it is a weekly meeting of one class (course, kind and groups)."""
    first: dict[tuple, str] = {}
    for sid in found:
        s = instance.session_by_id[sid]
        first.setdefault((s.course, s.kind, tuple(s.groups)), sid)
    if len(first) > 1:
        names = list(dict.fromkeys(_describe(instance, sid, None) for sid in first.values()))
        raise Unresolved(f"{question}: {'; '.join(names[:4])}?", "session")
    return found


def template(instance: Instance, ref: ClassRef, sender: str,
             timetable: Mapping[str, Placement] | None = None) -> str:
    """The class an extra one copies: any weekly meeting of it will do, so only different classes are ambiguous.
    An extra class is a lecture unless the message says otherwise."""
    found = classes(instance, ref, sender, timetable)
    if ref.kind is None and ref.session is None:
        found = [sid for sid in found if instance.session_by_id[sid].kind.value == "lecture"] or found
    return _one_kind(instance, found, "Which class is the extra one for")[0]


def meeting(instance: Instance, ref: ClassRef, sender: str, timetable: Mapping[str, Placement] | None,
            to_days: list[str] | None) -> str:
    """The meeting a move is for. "Move my ML lecture to Friday at 2 pm" names a class that meets several
    times a week, and any one of them will do: the one already on a day asked for (so the class does not
    meet twice that day), else the first. Different classes are ambiguous."""
    found = _one_kind(instance, classes(instance, ref, sender, timetable), "Which class do you mean")
    on_day = [sid for sid in found if to_days and (p := (timetable or {}).get(sid)) and p.day in to_days]
    return (on_day or found)[0]


def named_weeks(instance: Instance, period: Period, current_week: int | None, what: str) -> list[int]:
    """The weeks of a one-off request (an extra or a cancelled class), which must name them."""
    found = weeks(instance, period, current_week)
    if found is None:
        raise Unresolved(f"Which week {what}? Teaching weeks are 1 to {instance.calendar.weeks}.", "weeks")
    return found


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


@dataclass
class SwapRef:
    mine: str
    theirs: str
    weeks: list[int] | None


@dataclass
class ExtraRef:
    template: str
    weeks: list[int] | None
    days: list[str] | None
    slots: list[int] | None


@dataclass
class CancelRef:
    session: str
    weeks: list[int] | None


@dataclass
class ClubRef:
    """A club booking with everything the desk needs but the calendar date: ``date`` when the message gave
    one, else ``day`` in ``weeks`` (the week it falls in; None: the next one)."""

    club: str | None
    activity: str | None
    date: str | None
    day: str | None
    weeks: list[int] | None
    start: str
    end: str
    attendees: int
    cohorts: list[str]
    room: str | None


@dataclass
class ClashRef:
    """What a clash report names: days and start slots (None: not said) and course IDs."""

    days: list[str] | None
    slots: list[int] | None
    courses: list[str]


@dataclass
class Compiled:
    """An ``IntentOutput`` resolved against the instance and timetable.
    ``result`` is what the orchestrator takes for constraint intents; the
    refs feed the swap, extra-class and cancellation flows."""

    output: IntentOutput
    result: ParseResult
    swaps: list[SwapRef] = field(default_factory=list)
    extras: list[ExtraRef] = field(default_factory=list)
    cancels: list[CancelRef] = field(default_factory=list)
    clubs: list[ClubRef] = field(default_factory=list)
    clashes: list[ClashRef] = field(default_factory=list)
    question: str | None = None
    missing: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.question is not None


def _who(w: Who | None, sender: str) -> tuple[str, str]:
    return (w.kind, w.id) if w is not None else ("faculty", sender)


def _drafts(instance: Instance, intent, sender: str, timetable, current_week) -> list[DraftConstraint]:
    """Constraint drafts for the intents that are constraints (raises ``Unresolved``)."""
    j = intent.justification
    match intent:
        case Unavailable() | Avoid() | Only():
            kind, sid = _who(intent.who, sender)
            ctype = {"unavailable": "unavailable", "avoid": "avoid", "only": "prefer"}[intent.kind]
            hard = True if isinstance(intent, Unavailable) else intent.hard
            return [DraftConstraint(type=ctype, hard=hard, scope_kind=kind, scope_id=sid,
                                    days=days(instance, intent.time), slots=slots(instance, intent.time),
                                    weeks=weeks(instance, intent.period, current_week), justification=j)]
        case Move():
            if not intent.to.has_times and intent.to.days is None:
                raise Unresolved("When would you like the class to be?", "slots")
            sid = meeting(instance, intent.which, sender, timetable, days(instance, intent.to))
            return [DraftConstraint(type="prefer", hard=intent.hard, scope_kind="session", scope_id=sid,
                                    days=days(instance, intent.to), slots=slots(instance, intent.to),
                                    weeks=weeks(instance, intent.period, current_week), justification=j)]
        case RoomClosed():
            return [DraftConstraint(type="unavailable", hard=True, scope_kind="room", scope_id=intent.room,
                                    days=days(instance, intent.time), slots=slots(instance, intent.time),
                                    weeks=weeks(instance, intent.period, current_week), justification=j)]
        case NeedsRoom():
            if not intent.equipment and intent.room is None:
                raise Unresolved("What does the class need: which equipment or which room?", "equipment")
            return [DraftConstraint(type="require_room", hard=True, scope_kind="session", scope_id=sid,
                                    equipment=intent.equipment or None, justification=j)
                    for sid in classes(instance, intent.which, sender, timetable)]
    return []


def _load_limit(instance: Instance, intent: LoadLimit, request: Request, index: int) -> Constraint:
    kind, sid = _who(intent.who, request.sender_id)
    owner = request.sender_id if request.role in (Role.FACULTY, Role.HOD, Role.GUEST_FACULTY) else None
    return Constraint(id=f"C-{request.id[2:]}-{index}", type=ConstraintType.MAX_CONSECUTIVE, hard=True,
                      tier=Tier.COMMITMENT if request.role in (Role.COORDINATOR, Role.EXAM_CELL) else Tier.OPERATIONAL,
                      owner=owner, scope=Scope(**{kind: sid}), limit=intent.max_consecutive,
                      justification=Justification(intent.justification), source=Source(request=request.id),
                      valid=Validity())


def club_ref(intent: ClubBooking, current_week: int | None, instance: Instance) -> ClubRef:
    if intent.date is None and intent.day is None:
        raise Unresolved("On which date is the activity?", "date")
    if intent.start is None or intent.end is None:
        raise Unresolved("From what time to what time do you need the room?", "time")
    if minutes(intent.end) <= minutes(intent.start):
        raise Unresolved("The activity has to end after it starts. From what time to what time?", "time")
    if intent.attendees is None:
        raise Unresolved("How many people will come?", "attendees")
    wk = None if intent.date else weeks(instance, intent.period, current_week)
    return ClubRef(intent.club, intent.activity, intent.date, intent.day, wk, intent.start, intent.end,
                   intent.attendees, list(intent.cohorts), intent.room)


def clash_ref(intent: ClashReport, instance: Instance) -> ClashRef:
    """Whatever of the report resolves: a report goes on even when its times are off."""
    try:
        d, q = days(instance, intent.time), slots(instance, intent.time)
    except Unresolved:
        d, q = None, None
    known = {s.course for s in instance.sessions}
    return ClashRef(d, q, [c for c in intent.courses if c in known])


def compile_intents(instance: Instance, request: Request, out: IntentOutput, *,
                    timetable: Mapping[str, Placement] | None = None, current_week: int | None = None) -> Compiled:
    """Resolve every intent; the first reference that cannot be resolved turns the whole request into a question."""
    sender = request.sender_id
    drafts: list[DraftConstraint] = []
    compiled = Compiled(output=out, result=ParseResult(request=request, output=None))
    limits: list[LoadLimit] = []
    if out.action == "investigate":
        compiled.clashes = [clash_ref(i, instance) for i in out.intents if isinstance(i, ClashReport)]
    try:
        for intent in out.intents if out.action == "compile" else []:
            match intent:
                case ClubBooking():
                    compiled.clubs.append(club_ref(intent, current_week, instance))
                case ClashReport():
                    pass  # only read with ``investigate``
                case Swap():
                    if intent.theirs.owner is None and intent.theirs.session is None:
                        raise Unresolved("Who would you like to swap with?", "faculty")
                    mine = one_class(instance, intent.mine, sender, timetable)
                    theirs = one_class(instance, intent.theirs, sender, timetable)
                    compiled.swaps.append(SwapRef(mine, theirs, weeks(instance, intent.period, current_week)))
                case ExtraClass():
                    compiled.extras.append(ExtraRef(template(instance, intent.which, sender, timetable),
                                                    named_weeks(instance, intent.period, current_week,
                                                                "should the extra class be in"),
                                                    days(instance, intent.time), slots(instance, intent.time)))
                case Cancel():
                    compiled.cancels.append(CancelRef(one_class(instance, intent.which, sender, timetable),
                                                      named_weeks(instance, intent.period, current_week,
                                                                  "should the class be cancelled")))
                case LoadLimit():
                    limits.append(intent)
                case _:
                    drafts.extend(_drafts(instance, intent, sender, timetable, current_week))
    except Unresolved as e:
        compiled.question, compiled.missing = e.question, [e.missing]
        drafts, limits = [], []
        compiled.swaps, compiled.extras, compiled.cancels, compiled.clubs = [], [], [], []

    if out.action == "compile" and out.intents and all(isinstance(i, ClubBooking) for i in out.intents):
        # a club booking goes to the club desk, not the solver: anyone may ask, students included
        compiled.result = ParseResult(request=request, output=None, club=True)
        return compiled
    if out.action == "withdraw":
        compiled.result = ParseResult(request=request, output=None, withdraw=True)
        if request.role == Role.STUDENT:
            compiled.result.refusal = "students can report issues but cannot change the timetable"
        return compiled
    # swaps and extra classes have their own flows, which ask their own questions: the old format routes them there
    own_flow = any(i.kind in ("swap", "extra_class") for i in out.intents)
    legacy = lower(out, drafts) if compiled.question is None or own_flow else ParseOutput(
        request_type=out.request_type, action="clarify", missing=compiled.missing,
        clarifying_question=compiled.question)
    compiled.result = postprocess(instance, request, legacy)
    if compiled.result.refusal is None:
        for i, intent in enumerate(limits, len(drafts) + 1):
            c = _load_limit(instance, intent, request, i)
            if errors := validate_constraint(c, instance):
                compiled.result.errors.extend(f"{c.id}: {e}" for e in errors)
            elif reason := check_authority(instance, request, c):
                compiled.result.refusal = reason
                break
            else:
                compiled.result.constraints.append(c)
    return compiled


def lower(out: IntentOutput, drafts: list[DraftConstraint]) -> ParseOutput:
    """The ``ParseOutput`` wire format for ``out``, with its constraint intents already turned into ``drafts``.
    Swaps keep the old shape (compile, no constraints) and extra classes the old ``investigate``."""
    if out.action == "withdraw":
        raise ValueError("ParseOutput has no withdraw action")
    action = out.action
    if action == "compile" and any(i.kind == "extra_class" for i in out.intents):
        action = "investigate"
    rtype = out.request_type
    if rtype == "club_booking":  # a club booking has no place in the old format: the first other intent's type
        rtype = next((REQUEST_TYPE[i.kind] for i in out.intents if i.kind != "club_booking"), "out_of_scope")
    return ParseOutput(request_type=rtype, action=action, constraints=drafts, missing=out.missing,
                       clarifying_question=out.question)
