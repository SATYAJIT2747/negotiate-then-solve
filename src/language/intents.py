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
  where the class is now in the timetable ("my Thursday 10 am lecture");
* IDs are checked against the message (``ground_references``): an ID must
  exist and name what the message names. Each section has its own course of
  a title, and a parser that picks the right title in the wrong section, a
  look-alike title, an ID that does not exist or the wrong colleague is
  corrected by what the message names and who sent it.

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

import re
from collections.abc import Iterable, Mapping
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
    STUDENTS_REPORT,
    DraftConstraint,
    ParseOutput,
    ParseResult,
    check_authority,
    postprocess,
)

# The version of the intent vocabulary. A stored reading (training rows, mined failures, labelled messages)
# carries the version it was written in; ``language.migrations`` brings it to this one. Bump it when a
# reading's meaning changes (a default, a field's sense) and add the migration; an added optional field
# needs a bump too, so that older code is not handed rows it cannot read.
#   1  the first intents
#   2  a move without ``hard`` is a wish (it was a requirement)
#   3  periods of several weeks from now (``count``, ``rest_of_semester``); a kind of room for a club
#   4  courses, colleagues, sections and rooms by name as the message gives them (``language.names``), and
#      no directory in the prompt; an ID still reads as itself
#   5  weeks left out of a period (``except_weeks``) and odd or even weeks (``parity``): the reader says them
#      and this module does the arithmetic it got wrong ("from now on, except week 16" read as every week left)
#   6  a message as a small program: items labelled by ``id``, a class named as another item's (``same``),
#      limits on a week (``limit``), classes placed against others (``relate``), and choices between items
#      (``any_of``, ``ranked``, ``fallback``)
SCHEMA_VERSION = 6

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
    """Semester weeks. Empty means every week (a recurring request). Weeks counted from now are
    ``relative`` with ``count``: "the next three weeks" is next_week with count 3, "this week and next"
    this_week with count 2, "for the rest of the semester" rest_of_semester. ``except_weeks`` and ``parity``
    narrow whatever the rest names (every week when nothing else is given): "from now on, except week 16" is
    rest_of_semester with except_weeks [16], "on even weeks" parity even, "every other week from week 5" from
    week 5 with parity odd."""

    weeks: list[int] | None = None
    from_week: int | None = None
    to_week: int | None = None
    relative: Literal["this_week", "next_week", "rest_of_semester"] | None = None
    count: int | None = Field(None, ge=1, description="with this_week/next_week: how many weeks from it")
    except_weeks: list[int] | None = Field(None, description="weeks left out: 'except week 16'")
    parity: Literal["odd", "even"] | None = Field(None, description="only odd or only even weeks")

    @model_validator(mode="after")
    def _one_way(self) -> Period:
        ways = [self.weeks is not None, self.from_week is not None or self.to_week is not None,
                self.relative is not None]
        if sum(ways) > 1:
            raise ValueError("give weeks, a from/to range or a relative week, not several")
        if self.count is not None and self.relative not in ("this_week", "next_week"):
            raise ValueError("count goes with this_week or next_week")
        return self

    @property
    def is_recurring(self) -> bool:
        return (self.weeks is None and self.from_week is None and self.to_week is None and self.relative is None
                and not self.except_weeks and self.parity is None)


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
    same: str | None = Field(None, description="the id of another item: the class that item is about")


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
ID = Field(None, description="a label, when another item refers to this one")


class Unavailable(_Base):
    """Cannot teach (or a room cannot be used by them) at these times. Always hard."""

    kind: Literal["unavailable"] = "unavailable"
    id: str | None = ID
    who: Who | None = Field(None, description="null means the sender")
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    justification: JustificationLit = JUSTIFICATION


class Avoid(_Base):
    """Times they do not want ("no classes before 10", "keep Friday free")."""

    kind: Literal["avoid"] = "avoid"
    id: str | None = ID
    who: Who | None = None
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class Only(_Base):
    """The only times they want to teach ("only Mon-Thu"): every class in scope fits inside."""

    kind: Literal["only"] = "only"
    id: str | None = ID
    who: Who | None = None
    time: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class Move(_Base):
    """Put one class at these times ("move my Thursday ML lecture before lunch"). Any weekly meeting of the
    class will do; with the timetable, the one already on the day asked for."""

    kind: Literal["move"] = "move"
    id: str | None = ID
    which: ClassRef
    to: TimeSpec = Field(default_factory=TimeSpec)
    period: Period = Field(default_factory=Period)
    hard: bool = False  # a wish unless they say it must be
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
    id: str | None = ID
    which: ClassRef
    period: Period = Field(default_factory=Period)
    time: TimeSpec = Field(default_factory=TimeSpec)
    justification: JustificationLit = JUSTIFICATION


class Cancel(_Base):
    """A class not held in the weeks named."""

    kind: Literal["cancel"] = "cancel"
    id: str | None = ID
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
    """At most this many teaching hours in a row, for the sender: a limit on a course or a room means nothing
    to the solver, which counts hours per teacher, so the reader is not asked whose it is."""

    kind: Literal["load_limit"] = "load_limit"
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
    room_kind: str | None = Field(None, description="a kind of room asked for ('a seminar room'), not a named one")
    justification: JustificationLit = JUSTIFICATION


class ClashReport(_Base):
    """Students report two of their classes at the same time: when, and which courses (with ``investigate``)."""

    kind: Literal["clash_report"] = "clash_report"
    time: TimeSpec = Field(default_factory=TimeSpec)
    courses: list[str] = Field(default_factory=list, description="course IDs from the directory")
    justification: JustificationLit = JUSTIFICATION


Measure = Literal["classes_per_day", "hours_per_day", "days_per_week", "gap"]
Relation = Literal["same_day", "different_day", "right_after", "after", "day_gap", "same_hour", "same_room"]


class Limit(_Base):
    """A limit on the sender's week: at most ``n`` classes a day, hours of teaching a day or teaching days a
    week, or at least ``n`` free hours between two classes on a day (``gap``). Hours in a row are load_limit;
    hours a week are not the timetable's to change (out_of_scope)."""

    kind: Literal["limit"] = "limit"
    id: str | None = ID
    measure: Measure
    n: int = Field(ge=1)
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class Relate(_Base):
    """Classes placed against others: ``which`` against ``other`` (null: the classes ``which`` fits against
    each other, "no two Algorithms lectures on the same day"), against one of them (``any``: "right after one of
    its lectures") or each, the relation kept or ruled out (``negate``: "not right after a practical")."""

    kind: Literal["relate"] = "relate"
    id: str | None = ID
    which: ClassRef
    relation: Relation
    other: ClassRef | None = Field(None, description="null: which's classes against each other")
    any: bool = Field(False, description="against one of the other classes, not each of them")
    negate: bool = Field(False, description="the relation must not hold")
    days: int | None = Field(None, ge=1, description="day_gap: free days between")
    hard: bool = False
    justification: JustificationLit = JUSTIFICATION


class AnyOf(_Base):
    """At least one of the items ``of`` names (by id) holds: "Tuesday morning or Thursday afternoon"."""

    kind: Literal["any_of"] = "any_of"
    of: list[str] = Field(min_length=2)
    hard: bool = False


class Ranked(_Base):
    """The items ``of`` names in order of preference: the first if it can be, else the next ("ideally Friday,
    otherwise Thursday")."""

    kind: Literal["ranked"] = "ranked"
    of: list[str] = Field(min_length=2)


class Fallback(_Base):
    """The first item ``of`` names; if it cannot be met without overriding others, the second instead ("move
    it to Thursday; if that is not possible, cancel it")."""

    kind: Literal["fallback"] = "fallback"
    of: list[str] = Field(min_length=2, max_length=2)


Intent = Annotated[Unavailable | Avoid | Only | Move | Swap | ExtraClass | Cancel | RoomClosed | NeedsRoom | LoadLimit
                   | ClubBooking | ClashReport | Limit | Relate | AnyOf | Ranked | Fallback, Field(discriminator="kind")]
IntentKind = Literal["unavailable", "avoid", "only", "move", "swap", "extra_class", "cancel", "room_closed",
                     "needs_room", "load_limit", "club_booking", "clash_report", "limit", "relate", "any_of",
                     "ranked", "fallback"]
CHOICES = (AnyOf, Ranked, Fallback)

REQUEST_TYPE: dict[str, str] = {
    "unavailable": "unavailability", "avoid": "preference", "only": "preference", "move": "preference",
    "swap": "swap", "extra_class": "preference", "cancel": "preference", "room_closed": "room_issue",
    "needs_room": "room_issue", "load_limit": "preference", "club_booking": "club_booking",
    "clash_report": "clash_report", "limit": "preference", "relate": "preference", "any_of": "preference",
    "ranked": "preference", "fallback": "preference",
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
    """The weeks ``period`` names; ``None`` for every week. Weeks left out (``except_weeks``) and a parity
    narrow what the rest names; when it names no weeks, the weeks from now on ("on even weeks" is a standing
    request, like every week, and starts now), or every week when the current week is not known."""
    n = instance.calendar.weeks
    named = _named_weeks(instance, period, current_week)
    if not period.except_weeks and period.parity is None:
        return named
    left_out = set(period.except_weeks or [])
    out = [w for w in (named or range(current_week or 1, n + 1)) if w not in left_out
           and (period.parity is None or w % 2 == (period.parity == "odd"))]
    if not out:
        raise Unresolved("No teaching week is left once those weeks are taken out. Which weeks do you mean?",
                         "weeks")
    return out


def _named_weeks(instance: Instance, period: Period, current_week: int | None) -> list[int] | None:
    n = instance.calendar.weeks
    if period.relative is not None:
        if current_week is None:
            raise Unresolved("Which week do you mean?", "weeks")
        start = current_week + (period.relative == "next_week")
        # "the next three weeks" near the end of term: the weeks of it the semester has
        end = n if period.relative == "rest_of_semester" else min(n, start + (period.count or 1) - 1)
        out = list(range(start, end + 1))
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


def _describe(instance: Instance, sid: str, timetable: Mapping[str, Placement] | None,
              section: bool = False) -> str:
    s = instance.session_by_id[sid]
    name = instance.course_title(s.course) + ("" if s.kind.value == "lecture" else f" {s.kind.value}")
    if section:
        name += " for " + ", ".join(instance.group_by_id[g].name if g in instance.group_by_id else g for g in s.groups)
    p = (timetable or {}).get(sid)
    return f"{name} on {FULL_DAY[p.day]} at {hour(p.slot)}" if p else name


def _names(instance: Instance, sids: list[str], timetable: Mapping[str, Placement] | None) -> list[str]:
    """How a question names ``sids``: by course, kind and time, and by section where that alone tells them
    apart (the same course taught to two sections)."""
    names = list(dict.fromkeys(_describe(instance, sid, timetable) for sid in sids))
    sections = list(dict.fromkeys(_describe(instance, sid, timetable, True) for sid in sids))
    return sections if len(sections) > len(names) else names


def _titled(instance: Instance, text: str, course: str) -> bool:
    """Whether ``text`` (lower case) names ``course``'s title, allowing a typo in a long one."""
    from .lexicon import mentions

    title = instance.course_title(course).lower()
    return re.search(rf"\b{re.escape(title)}\b", text) is not None or mentions(text, title)


def _sections(instance: Instance) -> dict[str, set[str]]:
    """The groups that take each course."""
    out: dict[str, set[str]] = {}
    for s in instance.sessions:
        out.setdefault(s.course, set()).update(s.groups)
    return out


def _named(instance: Instance, course: str, text: str, sections: dict[str, set[str]],
           taken: frozenset[str] = frozenset()) -> str:
    """``course``, or the one course in the same section whose title the message names when it does not name
    ``course``'s ("Data Structures" read as Database Systems). An abbreviation ("ML") names no title: kept.
    A title in ``taken`` is another reference's in the same reading ("the Discrete Maths practical" beside "the
    Software Engineering lecture"): it is accounted for, so it does not stand in for this one."""
    if course not in sections or _titled(instance, text, course):
        return course
    hits = {c for c in sections if sections[c] & sections[course] and _titled(instance, text, c)
            and instance.course_title(c) not in taken}
    return hits.pop() if len(hits) == 1 else course


# what ties a course to a section after its title: "(Section 9", "(with Section 7", "with Section 4" (a typo
# allowed: "wiht", "witth"), "of Section 2", "for the Section 3"
_TIE = r"\s*\(?\s*(?:(?:with|wiht|witth|wtih|wth|whit|wit|of|for|in)\s+)?(?:the\s+)?"


def _section_of(instance: Instance, course: str, text: str, sender: str, sections: dict[str, set[str]]) -> str:
    """In a clash report, the course of ``course``'s title in the section the message ties it to ("Information
    Security (Section 9)", "... with Section 4", "... of Section 2"), else in a student sender's own section,
    else in the one section the message names ("19 students in Section 3 are enrolled in both ...")."""
    title = instance.course_title(course).lower()
    named = [g.id for g in instance.groups if re.search(rf"\b{re.escape(g.name.lower())}\b", text)]
    group = next((g.id for g in instance.groups if re.search(
        rf"\b{re.escape(title)}\b{_TIE}{re.escape(g.name.lower())}\b", text)),
        sender[3:] if sender.startswith("ST-") else named[0] if len(named) == 1 else None)
    same = [c for c in sections if group in sections[c] and instance.course_title(c).lower() == title]
    return same[0] if len(same) == 1 else course


def _group_named(instance: Instance, course: str | None, text: str, sections: dict[str, set[str]]) -> str | None:
    """The one section the message names by its name ("... for B.Tech I year CSE+ECE - Section A", as an answer
    to "which class do you mean" gives it) among those taking a course of ``course``'s title, else None. A name
    inside another named one ("Section A" in "... - Section A") is not counted twice."""
    title = instance.course_title(course) if course in sections else None
    taking = {g for c, gs in sections.items() if title is None or instance.course_title(c) == title for g in gs}
    named = [g for g in instance.groups if g.id in taking
             and re.search(rf"(?<![a-z0-9]){re.escape(g.name.lower())}(?![a-z0-9])", text)]
    named = [g for g in named if not any(g.name.lower() in h.name.lower() and g is not h for h in named)]
    return named[0].id if len(named) == 1 else None


def _named_titles(instance: Instance, text: str, sections: dict[str, set[str]]) -> list[str]:
    """The course titles ``text`` (lower case) names, longest first, so "Advanced Algorithms" is not also
    read as "Algorithms"."""
    titles = sorted({instance.course_title(c) for c in sections}, key=len, reverse=True)
    found: list[str] = []
    for t in titles:
        if _titled(instance, text, next(c for c in sections if instance.course_title(c) == t))                 and not any(t.lower() in f.lower() for f in found):
            found.append(t)
    return found


def _course(instance: Instance, course: str, text: str, sections: dict[str, set[str]],
            taken: frozenset[str] = frozenset()) -> str | None:
    """``course`` checked against the message (``_named``); an ID that does not exist is the one course whose
    title the message names, or nothing."""
    if course in sections:
        return _named(instance, course, text, sections, taken)
    titles = _named_titles(instance, text, sections)
    hits = [c for c in sections if instance.course_title(c) in titles]
    return hits[0] if len(titles) == 1 and hits else None


def _surname(name: str) -> str:
    return name.split()[-1].lower()


def _colleague(instance: Instance, fid: str, text: str, sender: str) -> str:
    """``fid``, unless the message does not name them and names exactly one other colleague: that one."""
    from .lexicon import _close, words

    ws = words(text)

    def named(f) -> bool:
        s = _surname(f.name)
        return any(w == s or (len(s) >= 5 and _close(w, s)) for w in ws)

    if fid in instance.faculty_by_id and named(instance.faculty_by_id[fid]):
        return fid
    others = [f.id for f in instance.faculty if f.id != sender and named(f)]
    return others[0] if len(others) == 1 else fid


def ground_references(instance: Instance, request: Request, out: IntentOutput) -> IntentOutput:
    """``out`` with its IDs checked against the message, so that every ID exists and names what the message
    names. A course whose title the message does not name becomes the one it does in the same section, and an
    unknown course ID the one course the message names (or none: the rest of the reference decides). In a
    clash report each course is taken in the section the message ties it to, or the student's own (or the one
    section the message names), and a course a student's report names but left out is added. A colleague (a class's owner, a
    ``who``) the message does not name becomes the one colleague it does."""
    text, sender = request.raw_text.lower(), request.sender_id
    sections = _sections(instance)
    refs = [(i, name, ref.course) for i, intent in enumerate(out.intents) for name in ("which", "mine", "theirs", "other")
            if isinstance(ref := getattr(intent, name, None), ClassRef) and ref.course in sections]
    intents = []
    for i, intent in enumerate(out.intents):
        update = {}
        for name in ("which", "mine", "theirs", "other"):
            ref = getattr(intent, name, None)
            if isinstance(ref, ClassRef):
                fix = {}
                if ref.course is not None:
                    # the titles the reading's other class references name, which this one cannot take
                    taken = frozenset(instance.course_title(c) for j, n, c in refs if (j, n) != (i, name)
                                      and c != ref.course)
                    fix["course"] = _course(instance, ref.course, text, sections, taken)
                if ref.owner is not None:
                    fix["owner"] = _colleague(instance, ref.owner, text, sender)
                if ref.group is None and ref.session is None:
                    fix["group"] = _group_named(instance, fix.get("course", ref.course), text, sections)
                if any(fix[k] != getattr(ref, k) for k in fix):
                    update[name] = ref.model_copy(update=fix)
        who = getattr(intent, "who", None)
        if isinstance(who, Who) and who.kind == "faculty" and who.id != sender:
            fid = _colleague(instance, who.id, text, sender)
            if fid != who.id:
                update["who"] = who.model_copy(update={"id": fid})
        if isinstance(intent, ClashReport):
            courses = [c for c in (_course(instance, c, text, sections) for c in intent.courses) if c]
            if sender.startswith("ST-"):
                have = {instance.course_title(c) for c in courses}
                courses += [next(c for c in sorted(sections) if instance.course_title(c) == t)
                            for t in _named_titles(instance, text, sections) if t not in have]
            courses = [_section_of(instance, c, text, sender, sections) for c in courses]
            update["courses"] = list(dict.fromkeys(courses))
        intents.append(intent.model_copy(update=update) if update else intent)
    if len(intents) > 1:  # an intent that names nothing, tacked on after what was asked: the reader's slip, dropped
        # ("move my DS lecture on Monday to Thursday afternoon" read as the move and a clash report, which became a
        # request of its own, "clash report.", waiting for approval with no change); alone it stays (asked about)
        intents = [i for i in intents if not _names_nothing(i)] or intents
    return out.model_copy(update={"intents": intents})


def _names_nothing(intent) -> bool:
    """A clash report without two courses (a clash is between two classes: students' reports name both), or a swap
    with nobody's class on the other side."""
    if isinstance(intent, ClashReport):
        return len(intent.courses) < 2
    return isinstance(intent, Swap) and intent.theirs == ClassRef()


def classes(instance: Instance, ref: ClassRef, sender: str,
            timetable: Mapping[str, Placement] | None = None) -> list[str]:
    """Every session ``ref`` fits (owner defaults to the sender). Each section has its own course of a title,
    and a message names the title: a course stands for every course of its title the owner teaches, and a
    group the reference names picks the section (the course's ID may be the title's in another section).
    Which section is meant is otherwise up to the rest of the reference (day and time), or a question."""
    if ref.session is not None:
        if ref.session not in instance.session_by_id:
            raise Unresolved("I could not find that class. Which one do you mean?", "session")
        return [ref.session]
    owner = ref.owner or sender
    # someone who teaches no classes (the timetable office, the lab in-charge) means a class by what it is,
    # whoever teaches it: "move the Section 3 Algorithms lecture on Monday at 9 am"
    anyone = ref.owner is None and not any(s.faculty == sender for s in instance.sessions)
    courses = {ref.course} if ref.course is not None else None
    if ref.course is not None and ref.course in instance.course_titles:
        title = instance.course_title(ref.course)
        courses = {s.course for s in instance.sessions if (anyone or s.faculty == owner)
                   and instance.course_title(s.course) == title} or courses
    found = [s for s in instance.sessions if (anyone or s.faculty == owner)
             and (courses is None or s.course in courses)
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
    names = _names(instance, found, timetable)
    if len(names) == 1:  # the weekly meetings of one class, and no timetable to tell them apart
        raise Unresolved(f"You have {len(found)} {names[0]} classes a week. Which day's class do you mean?", "session")
    raise Unresolved(f"Which class do you mean: {'; '.join(names[:4])}?", "session")


_ORDINAL = {"first": 0, "1": 0, "1st": 0, "second": 1, "2": 1, "2nd": 1, "third": 2, "3": 2, "3rd": 2,
            "fourth": 3, "4": 3, "4th": 3, "last": -1}
_OPTIONS = re.compile(r"^Which class [^:]*:\s*(.+?)\??$")


def picked_option(question: str | None, answer: str) -> str | None:
    """The option an answer picks when the question listed classes to choose from ("Which class do you mean:
    X - Section A; X - Section B?"): the one whose own words, those not in every option, the answer says alone
    ("section a", "A", "the Monday one"), or the one it counts to ("the second"). The reader is given the
    option in full, since a short answer ("section a") names no section by its name. None when the answer
    picks none or several."""
    m = _OPTIONS.match((question or "").strip())
    if not m:
        return None
    options = [o.strip() for o in m[1].split(";") if o.strip()]
    if len(options) < 2:
        return None
    said = re.findall(r"[a-z0-9]+", answer.lower())
    # an option's own words: what is left once the words every option starts and ends with are taken off
    # ("... - Section A" and "... - Section B": "a" and "b"; the "b" of "B.Tech" is in every option's start)
    words = [re.findall(r"[a-z0-9]+", o.lower()) for o in options]
    head = 0
    while all(len(w) > head and w[head] == words[0][head] for w in words):
        head += 1
    tail = 0
    while all(len(w) > head + tail and w[-1 - tail] == words[0][-1 - tail] for w in words):
        tail += 1
    own = [set(w[head:len(w) - tail]) for w in words]
    hits = [o for o, w in zip(options, own, strict=True) if w & set(said)]
    if hits:  # "section 2" is Section 2, not the second option
        return hits[0] if len(hits) == 1 else None
    ordinal = [_ORDINAL[w] for w in said if w in _ORDINAL]
    if len(said) <= 3 and len(ordinal) == 1 and ordinal[0] < len(options):
        return options[ordinal[0]]
    return None


_PLURAL = re.compile(r"\b(all|every|classes|lectures|labs|practicals|tutorials|sessions)\b", re.IGNORECASE)


def _cancelled_classes(instance: Instance, ref: ClassRef, sender: str, timetable: Mapping[str, Placement] | None,
                       text: str) -> list[str]:
    """The classes a cancellation is for. "Cancel my classes in weeks 15-16" and "cancel my Computer Graphics
    lectures" mean every class they fit, when they name no day or time; one class is asked about as before."""
    if ref.session is None and ref.day is None and ref.time is None and _PLURAL.search(text):
        return classes(instance, ref, sender, timetable)
    return [one_class(instance, ref, sender, timetable)]


def _one_kind(instance: Instance, found: list[str], question: str) -> list[str]:
    """``found`` when every session in it is a weekly meeting of one class (course, kind and groups)."""
    first: dict[tuple, str] = {}
    for sid in found:
        s = instance.session_by_id[sid]
        first.setdefault((s.course, s.kind, tuple(s.groups)), sid)
    if len(first) > 1:
        names = _names(instance, list(first.values()), None)
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


# ---------------------------------------------------------------------------
# Intents back into words
# ---------------------------------------------------------------------------


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _say_ref(instance: Instance, ref: ClassRef, sender: str) -> str:
    owner = ref.owner or sender
    whose = "my" if owner == sender else (
        f"{instance.faculty_by_id[owner].name}'s" if owner in instance.faculty_by_id else "the")
    if ref.session is not None and ref.session in instance.session_by_id:
        s = instance.session_by_id[ref.session]
        name = f"{instance.course_title(s.course)} {s.kind.value}"
    else:
        title = instance.course_title(ref.course) if ref.course in instance.course_titles else (ref.course or "")
        name = " ".join(x for x in (title, ref.kind or "class") if x)
    where = (f" on {FULL_DAY[ref.day]}" if ref.day else "") + (f" at {_clock_text(ref.time)}" if ref.time else "")
    return f"{whose} {name}{where}"


def _say_time(spec: TimeSpec) -> str:
    out = []
    if spec.days:
        out.append("on " + _and([FULL_DAY[d] for d in spec.days]))
    if spec.part:
        out.append({"morning": "in the morning", "afternoon": "in the afternoon", "lunch": "at lunch"}[spec.part])
    if spec.after and spec.before:
        out.append(f"between {_clock_text(spec.after)} and {_clock_text(spec.before)}")
    elif spec.before:
        out.append(f"before {_clock_text(spec.before)}")
    elif spec.after:
        out.append(f"after {_clock_text(spec.after)}")
    if spec.at:
        out.append("at " + _and([_clock_text(t) for t in spec.at]))
    return " ".join(out)


def _say_period(period: Period) -> str:
    said = _say_named_weeks(period)
    if period.parity:
        said = f"{said} in {period.parity} weeks" if said else f"in {period.parity} weeks"
    if period.except_weeks:
        w = sorted(period.except_weeks)
        said += (" " if said else "") + "except " + ("week " if len(w) == 1 else "weeks ") + _and([str(x) for x in w])
    return said


def _say_named_weeks(period: Period) -> str:
    if period.weeks:
        w = sorted(period.weeks)
        return f"in week {w[0]}" if len(w) == 1 else "in weeks " + _and([str(x) for x in w])
    if period.from_week is not None or period.to_week is not None:
        if period.to_week is None:
            return f"from week {period.from_week} on"
        return f"from week {period.from_week or 1} to week {period.to_week}"
    if period.relative == "rest_of_semester":
        return "for the rest of the semester"
    if period.relative is not None:
        first = "this week" if period.relative == "this_week" else "next week"
        return first if (period.count or 1) == 1 else f"for {period.count} weeks from {first}"
    return ""


def say(instance: Instance, intent, sender: str) -> str:
    """One intent as a sentence the sender could have written ("Cancel my Computer Graphics lecture on
    Friday in week 15."): what a part of a message split into requests is read as, when its sender answers
    its question."""
    t = _say_time(getattr(intent, "time", None) or TimeSpec())
    p = _say_period(getattr(intent, "period", None) or Period())
    match intent:
        case Unavailable():
            core = f"I am not available {t}" if t else "I am not available"
        case Avoid():
            core = f"{'Do not' if intent.hard else 'If possible, do not'} schedule my classes {t}".rstrip()
        case Only():
            core = f"{'I can only' if intent.hard else 'I would prefer to only'} teach {t}".rstrip()
        case Move():
            to = _say_time(intent.to)
            core = f"Move {_say_ref(instance, intent.which, sender)}" + (f" to {to.removeprefix('on ')}" if to else "")
        case Swap():
            core = (f"Swap {_say_ref(instance, intent.mine, sender)} with "
                    f"{_say_ref(instance, intent.theirs, sender)}")
        case ExtraClass():
            core = f"Add an extra class of {_say_ref(instance, intent.which, sender)} {t}".rstrip()
        case Cancel():
            core = f"Cancel {_say_ref(instance, intent.which, sender)}"
        case RoomClosed():
            room = instance.room_by_id[intent.room].name if intent.room in instance.room_by_id else intent.room
            core = f"{room} is closed {t}".rstrip()
        case NeedsRoom():
            needs = " and ".join(intent.equipment) or "a suitable room"
            core = f"{_say_ref(instance, intent.which, sender)} needs a room with {needs}"
            core = core[0].upper() + core[1:]
        case LoadLimit():
            core = f"No more than {intent.max_consecutive} classes in a row for me"
        case Limit():
            n = intent.n
            core = (f"At least {n} free hour{'s' * (n != 1)} between two of my classes" if intent.measure == "gap" else
                    f"No more than {n} " + {"classes_per_day": "classes a day", "hours_per_day": "hours of teaching a day",
                                            "days_per_week": "teaching days a week"}[intent.measure] + " for me")
        case Relate():
            rel = {"same_day": "on the same day as", "different_day": "on a different day from",
                   "right_after": "right after", "after": "after, on the same day,", "same_hour": "at the same hour as",
                   "same_room": "in the same room as",
                   "day_gap": f"at least {intent.days or 1} free day{'s' * ((intent.days or 1) != 1)} apart from"}
            each = {"same_day": "on the same day", "different_day": "on different days", "same_hour": "at the same hour",
                    "same_room": "in the same room",
                    "day_gap": f"at least {intent.days or 1} free day{'s' * ((intent.days or 1) != 1)} apart"}
            mine = _say_ref(instance, intent.which, sender)
            if intent.other is None:
                core = f"Keep {mine}s {'not ' * intent.negate}{each.get(intent.relation, intent.relation)}"
            else:
                core = (f"Keep {mine} {'not ' * intent.negate}{rel[intent.relation]} "
                        f"{'one of ' * intent.any}{_say_ref(instance, intent.other, sender)}")
        case AnyOf():
            core = "Any one of the choices above will do"
        case Ranked():
            core = "The choices above in that order of preference"
        case Fallback():
            core = "If the first of those cannot be done, the second instead"
        case ClubBooking():
            core = f"Book a room for {intent.club or 'our club'}" + (f" on {FULL_DAY[intent.day]}" if intent.day else "")
        case _:
            core = intent.kind.replace("_", " ")
    return " ".join(x for x in (core, p) if x) + "."


# the whole name first: "wed(?:day)?" never matched "Wednesday", nor "tue(?:day)?" "Tuesdays"
_SAID_DAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|mon|tues|tue|wed|thurs|thur|thu|fri)s?\b",
                       re.IGNORECASE)
_SAID_TIME = re.compile(r"\b(\d{1,2})(?::(\d\d))?\s*(am|pm)\b", re.IGNORECASE)
_SAID_WEEKS = re.compile(r"\bweeks?\s+(\d{1,2})(?:\s*(?:-|–|to|through|until|and)\s*(?:weeks?\s+)?(\d{1,2}))?",
                         re.IGNORECASE)
_SAID_FIRM = re.compile(r"\b(must|only|cannot|can't|can not|have to|has to|no way)\b", re.IGNORECASE)


def _used(out: IntentOutput) -> tuple[set[str], set[int], set[int], set[str]]:
    """The days, clock hours, weeks and parts of the day the intents use."""
    days: set[str] = set()
    hours: set[int] = set()
    weeks: set[int] = set()
    parts: set[str] = set()

    def clocks(*cs):
        hours.update(minutes(c) // 60 for c in cs if c)

    for i in out.intents:
        # "only on Tuesday and Thursday" settles every day: the days it leaves out ("I can't teach on Mondays,
        # Wednesdays or Fridays") are used, as the days it rules out
        if isinstance(i, Only) and i.time.days:
            days.update(_SAID_DAY_FULL.values())
        for name in ("time", "to"):
            if (spec := getattr(i, name, None)) is not None:
                days.update(spec.days or [])
                clocks(spec.before, spec.after, *(spec.at or []))
                if spec.part:
                    parts.add(spec.part)
                    hours.update(range(9, 13) if spec.part == "morning" else range(13, 18))
        for name in ("which", "mine", "theirs", "other"):
            if (ref := getattr(i, name, None)) is not None:
                days.update([ref.day] if ref.day else [])
                clocks(ref.time)
        if isinstance(i, ClubBooking):
            days.update([i.day] if i.day else [])
            clocks(i.start, i.end)
        if (p := getattr(i, "period", None)) is not None:
            weeks.update(p.weeks or [])
            weeks.update(p.except_weeks or [])  # a week said to be left out is used too
            if p.from_week is not None or p.to_week is not None:
                weeks.update(range(p.from_week or 1, (p.to_week or 52) + 1))
    return days, hours, weeks, parts


# where a part of the day ends or starts: "before 1 pm" is the morning, "after 12 pm" the afternoon
_PART_EDGE = {"morning": ("before", 13), "afternoon": ("after", 12)}
_EDGE_WORD = {"before": re.compile(r"\b(before|until|till|til|by|no later than)\s*$", re.IGNORECASE),
              "after": re.compile(r"\b(after|from|no earlier than)\s*$", re.IGNORECASE)}


def _edge_of_part(text: str, m: re.Match, hour: int, parts: set[str]) -> bool:
    """Whether a said time is the edge of a part of the day the reading uses: "only mornings, before 1 pm"
    read as the morning uses the 1 pm, it is where the morning ends."""
    return any(hour == edge and _EDGE_WORD[side].search(text[:m.start()])
               for part, (side, edge) in _PART_EDGE.items() if part in parts)


def unaccounted(text: str, out: IntentOutput, where: Iterable[tuple[str, int]] = ()) -> list[str]:
    """What a message says that its reading does not use: a day, a time or a week nothing was read from, and
    a requirement ("only", "must", "can't") read as a wish. Shown to the sender and the approver, so a
    reading that dropped a clause ("... but not on Friday") is seen before anything is approved. ``where``:
    (day, clock hour) of the classes the reading picked out, now; "my lecture on Thursday" names the class by
    where it is, so that day and hour were used to find it, not dropped."""
    if out.action != "compile" or not out.intents:
        return []
    days, hours, weeks, parts = _used(out)
    for d, h in where:
        days.add(d)
        hours.add(h)
    notes = []
    said_days = list(dict.fromkeys(_SAID_DAY_FULL[m[1].lower()[:3]] for m in _SAID_DAY.finditer(text)))
    if missed := [d for d in said_days if d not in days]:
        notes.append(f"You mentioned {_and([FULL_DAY[d] for d in missed])}, which the request as read does not use.")
    said_hours = []
    for m in _SAID_TIME.finditer(text):
        h = int(m[1]) % 12 + (12 if m[3].lower() == "pm" else 0)
        if not _edge_of_part(text, m, h, parts):
            said_hours.append((h, m[0]))
    if missed := [s for h, s in said_hours if h not in hours]:
        notes.append(f"You mentioned {_and(missed)}, which the request as read does not use.")
    said_weeks = set()
    for m in _SAID_WEEKS.finditer(text):
        a, b = int(m[1]), int(m[2]) if m[2] else int(m[1])
        said_weeks.update({a, b})
    if missed := sorted(said_weeks - weeks):
        notes.append(f"You mentioned week {_and([str(w) for w in missed])}, which the request as read does not use.")
    soft = [i for i in out.intents if isinstance(i, (Avoid, Only, Move)) and not i.hard]
    if (m := _SAID_FIRM.search(text)) and soft:
        notes.append(f"You wrote \"{m[0]}\", but it was read as a wish the timetable may not meet, not a requirement.")
    return notes


_SAID_DAY_FULL = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}


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
    room_kind: str | None = None  # a kind of room asked for: the desk picks one that fits


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
    notes: list[str] = field(default_factory=list)  # what ``checks`` repaired or changed in the reading
    fallback: list = field(default_factory=list)  # the intents to fall back on when the request cannot be met

    @property
    def needs_clarification(self) -> bool:
        return self.question is not None


def _who(w: Who | None, sender: str) -> tuple[str, str]:
    return (w.kind, w.id) if w is not None else ("faculty", sender)


def _drafts(instance: Instance, intent, sender: str, timetable, current_week,
            text: str = "") -> list[DraftConstraint]:
    """Constraint drafts for the intents that are constraints (raises ``Unresolved``); ``text`` is the
    message, which says whether a must about a move is about where it goes or only about leaving."""
    from .lexicon import placement_firm_only_on_leaving, unable

    j = intent.justification
    match intent:
        case Unavailable() | Avoid() | Only():
            kind, sid = _who(intent.who, sender)
            ctype = {"unavailable": "unavailable", "avoid": "avoid", "only": "prefer"}[intent.kind]
            hard = True if isinstance(intent, Unavailable) else intent.hard
            if ctype != "prefer" and (kind, sid) == ("faculty", sender) and text:
                # how firm is the reader's; whether a requirement is an unavailability is the message's wording
                ctype = "unavailable" if hard and unable(text) else "avoid"
            return [DraftConstraint(type=ctype, hard=hard, scope_kind=kind, scope_id=sid,
                                    days=days(instance, intent.time), slots=slots(instance, intent.time),
                                    weeks=weeks(instance, intent.period, current_week), justification=j)]
        case Move():
            if not intent.to.has_times and intent.to.days is None:
                raise Unresolved("When would you like the class to be?", "slots")
            sid = meeting(instance, intent.which, sender, timetable, days(instance, intent.to))
            ws = weeks(instance, intent.period, current_week)
            to = DraftConstraint(type="prefer", hard=intent.hard, scope_kind="session", scope_id=sid,
                                 days=days(instance, intent.to), slots=slots(instance, intent.to), weeks=ws,
                                 justification=j)
            if (not intent.hard or not placement_firm_only_on_leaving(text)
                    or (now := (timetable or {}).get(sid)) is None):
                return [to]
            # a must about leaving only ("it can't stay where it is", a clash with where it is): that rules out
            # where the class is now, while where it should go is still a wish the solver keeps where it can
            return [DraftConstraint(type="avoid", hard=True, scope_kind="session", scope_id=sid, days=[now.day],
                                    slots=[now.slot], weeks=ws, justification=j),
                    to.model_copy(update={"hard": False})]
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


def _tier(request: Request, hard: bool) -> Tier:
    if not hard:
        return Tier.PREFERENCE
    return Tier.COMMITMENT if request.role in (Role.COORDINATOR, Role.EXAM_CELL) else Tier.OPERATIONAL


def _owner(request: Request) -> str | None:
    return request.sender_id if request.role in (Role.FACULTY, Role.HOD, Role.GUEST_FACULTY) else None


def _limit_constraint(instance: Instance, intent: Limit, request: Request, index: int) -> Constraint:
    return Constraint(id=f"C-{request.id[2:]}-{index}", type=ConstraintType.LOAD, hard=intent.hard,
                      tier=_tier(request, intent.hard), owner=_owner(request), scope=Scope(faculty=request.sender_id),
                      measure=intent.measure, limit=intent.n, justification=Justification(intent.justification),
                      source=Source(request=request.id), valid=Validity())


def _relate_constraint(instance: Instance, intent: Relate, request: Request,
                       timetable: Mapping[str, Placement] | None):
    """The classes a relation places and the ones it places them against, resolved now (a reference that
    fits nothing is a question); the constraint is made once its index is known."""
    sender = request.sender_id
    mine = classes(instance, intent.which, sender, timetable)
    others = None
    if intent.other is not None:
        others = [s for s in classes(instance, intent.other, sender, timetable) if s not in mine]
        if not others:
            raise Unresolved("Which classes should it be placed against?", "session")
    elif len(mine) < 2:
        raise Unresolved("Which classes should be placed against each other?", "session")
    owner = instance.session_by_id[mine[0]].faculty

    def make(index: int) -> Constraint:
        return Constraint(id=f"C-{request.id[2:]}-{index}", type=ConstraintType.RELATE, hard=intent.hard,
                          tier=_tier(request, intent.hard), owner=_owner(request), scope=Scope(faculty=owner),
                          sessions=mine, others=others, relation=intent.relation,
                          amount=intent.days if intent.relation == "day_gap" else None, any_other=intent.any,
                          negate=intent.negate, justification=Justification(intent.justification),
                          source=Source(request=request.id), valid=Validity())
    return make


def _member(instance: Instance, item, request: Request, timetable, current_week) -> Constraint:
    """One choice of an any_of or a ranked: a time, a limit or a relation, as a constraint of its own."""
    if isinstance(item, Limit):
        return _limit_constraint(instance, item, request, 0)
    if isinstance(item, Relate):
        return _relate_constraint(instance, item, request, timetable)(0)
    if not isinstance(item, (Unavailable, Avoid, Only, Move, RoomClosed, NeedsRoom)):
        raise Unresolved("A choice can be between times, limits and how classes sit together. Which do you mean?",
                         "session")
    drafts = _drafts(instance, item, request.sender_id, timetable, current_week, request.raw_text)
    if len(drafts) != 1:
        raise Unresolved("Each choice has to be one time or one class. Which do you mean?", "session")
    got = postprocess(instance, request, ParseOutput(request_type=REQUEST_TYPE[item.kind], action="compile",
                                                     constraints=drafts))
    if got.refusal or not got.constraints:
        raise Unresolved(f"One of the choices cannot be used ({got.refusal or '; '.join(got.errors)}). "
                         "Which do you mean?", "session")
    return got.constraints[0]


def _choice(instance: Instance, intent, items: Mapping[str, object], request: Request, timetable, current_week):
    """An any_of or a ranked choice: its members compiled now, the constraint made once its index is known."""
    members = []
    for ref in intent.of:
        if ref not in items:
            raise Unresolved("I could not tell which choices you mean. Could you say them again?", "session")
        members.append(_member(instance, items[ref], request, timetable, current_week))
    hard = getattr(intent, "hard", False)
    ctype = ConstraintType.ANY_OF if isinstance(intent, AnyOf) else ConstraintType.RANKED

    def make(index: int) -> Constraint:
        cid = f"C-{request.id[2:]}-{index}"
        return Constraint(id=cid, type=ctype, hard=hard, tier=_tier(request, hard), owner=_owner(request),
                          scope=Scope(faculty=request.sender_id), source=Source(request=request.id),
                          members=[m.model_copy(update={"id": f"{cid}.{k}"}) for k, m in enumerate(members, 1)])
    return make


def _bind(intents: list) -> list:
    """Each class named as another item's (``same``) replaced by that item's class reference."""
    refs = {i.id: getattr(i, "which", None) for i in intents if getattr(i, "id", None)}
    out = []
    for intent in intents:
        update = {}
        for name in ("which", "other"):
            ref = getattr(intent, name, None)
            if isinstance(ref, ClassRef) and ref.same is not None:
                bound = refs.get(ref.same)
                if not isinstance(bound, ClassRef):
                    raise Unresolved("I could not tell which class you mean. Which one is it?", "session")
                update[name] = bound
        out.append(intent.model_copy(update=update) if update else intent)
    return out


def _load_limit(instance: Instance, intent: LoadLimit, request: Request, index: int) -> Constraint:
    owner = request.sender_id if request.role in (Role.FACULTY, Role.HOD, Role.GUEST_FACULTY) else None
    return Constraint(id=f"C-{request.id[2:]}-{index}", type=ConstraintType.MAX_CONSECUTIVE, hard=True,
                      tier=Tier.COMMITMENT if request.role in (Role.COORDINATOR, Role.EXAM_CELL) else Tier.OPERATIONAL,
                      owner=owner, scope=Scope(faculty=request.sender_id), limit=intent.max_consecutive,
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
                   intent.attendees, list(intent.cohorts), intent.room, intent.room_kind)


def clash_ref(intent: ClashReport, instance: Instance) -> ClashRef:
    """Whatever of the report resolves: a report goes on even when its times are off."""
    try:
        d, q = days(instance, intent.time), slots(instance, intent.time)
    except Unresolved:
        d, q = None, None
    known = {s.course for s in instance.sessions}
    return ClashRef(d, q, [c for c in intent.courses if c in known])


def equipment_not_rooms(instance: Instance, out: IntentOutput) -> IntentOutput:
    """"My practical needs the routers, and it has to be on Tuesday" read as "routers is closed on Tuesday" plus
    a move: a closure of a room that does not exist, whose name is equipment some room has, is the equipment the
    message's class needs (with no class named, it is left as read, to be asked about)."""
    if out.action != "compile":
        return out
    rooms = set(instance.room_by_id) | {r.name.lower() for r in instance.rooms}
    vocab = {e.lower(): e for r in instance.rooms for e in r.equipment}

    def equipment(name: str) -> str | None:
        n = name.lower().strip().removeprefix("the ")
        return next((vocab[c] for c in (n, n.rstrip("s"), f"{n}s") if c in vocab), None)

    which = next((w for i in out.intents if isinstance(w := getattr(i, "which", None), ClassRef)), None)
    if which is None:
        return out
    kept: list = []
    changed = False
    for i in out.intents:
        if isinstance(i, RoomClosed) and i.room not in rooms and i.room.lower() not in rooms and (e := equipment(i.room)):
            changed = True
            if not any(isinstance(x, NeedsRoom) and e in x.equipment for x in [*kept, *out.intents]):
                kept.append(NeedsRoom(which=which, equipment=[e], justification=i.justification))
            continue
        kept.append(i)
    return out.model_copy(update={"intents": kept}) if changed else out


def compile_intents(instance: Instance, request: Request, out: IntentOutput, *,
                    timetable: Mapping[str, Placement] | None = None, current_week: int | None = None) -> Compiled:
    """Resolve every intent; the first reference that cannot be resolved turns the whole request into a question."""
    sender = request.sender_id
    from .names import to_ids

    out = ground_references(instance, request, equipment_not_rooms(instance, to_ids(instance, out)))
    drafts: list[DraftConstraint] = []
    compiled = Compiled(output=out, result=ParseResult(request=request, output=None))
    # constraints made after the drafts, once their index is known: hours in a row, limits, relations, choices
    direct: list = []
    if out.action == "investigate":
        compiled.clashes = [clash_ref(i, instance) for i in out.intents if isinstance(i, ClashReport)]
    try:
        intents = _bind(out.intents) if out.action == "compile" else []
        # a fallback's first item is a requirement: a wish is never impossible, so nothing would fall back
        first = {i.of[0] for i in intents if isinstance(i, Fallback)}
        intents = [i.model_copy(update={"hard": True}) if getattr(i, "id", None) in first and hasattr(i, "hard")
                   else i for i in intents]
        items = {i.id: i for i in intents if getattr(i, "id", None)}
        # the members of a choice are compiled with it, and a fallback's second item only if the first fails
        chosen = {x for i in intents if isinstance(i, (AnyOf, Ranked)) for x in i.of}
        chosen |= {i.of[1] for i in intents if isinstance(i, Fallback)}
        for intent in intents:
            if getattr(intent, "id", None) in chosen:
                continue
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
                    wks = named_weeks(instance, intent.period, current_week, "should the class be cancelled")
                    for sid in _cancelled_classes(instance, intent.which, sender, timetable, request.raw_text):
                        compiled.cancels.append(CancelRef(sid, wks))
                case LoadLimit():
                    direct.append(lambda index, i=intent: _load_limit(instance, i, request, index))
                case Limit():
                    direct.append(lambda index, i=intent: _limit_constraint(instance, i, request, index))
                case Relate():
                    direct.append(_relate_constraint(instance, intent, request, timetable))
                case AnyOf() | Ranked():
                    direct.append(_choice(instance, intent, items, request, timetable, current_week))
                case Fallback():
                    if intent.of[0] not in items or intent.of[1] not in items:
                        raise Unresolved("I could not tell what to do if the first change cannot be made. "
                                         "Could you say it again?", "session")
                    compiled.fallback = [items[intent.of[1]]]
                case _:
                    drafts.extend(_drafts(instance, intent, sender, timetable, current_week, request.raw_text))
    except Unresolved as e:
        compiled.question, compiled.missing = e.question, [e.missing]
        drafts, direct, compiled.fallback = [], [], []
        compiled.swaps, compiled.extras, compiled.cancels, compiled.clubs = [], [], [], []

    if out.action == "compile" and out.intents and all(isinstance(i, ClubBooking) for i in out.intents):
        # a club booking goes to the club desk, not the solver: anyone may ask, students included
        compiled.result = ParseResult(request=request, output=None, club=True)
        return compiled
    if out.action == "withdraw":
        compiled.result = ParseResult(request=request, output=None, withdraw=True)
        if request.role == Role.STUDENT:
            compiled.result.refusal = STUDENTS_REPORT
        return compiled
    # swaps and extra classes have their own flows, which ask their own questions: the old format routes them there
    own_flow = any(i.kind in ("swap", "extra_class") for i in out.intents)
    legacy = lower(out, drafts) if compiled.question is None or own_flow else ParseOutput(
        request_type=out.request_type, action="clarify", missing=compiled.missing,
        clarifying_question=compiled.question)
    compiled.result = postprocess(instance, request, legacy)
    if legacy.action == "investigate" and legacy.constraints and compiled.result.refusal is None:
        # constraints beside an extra class ("away on Wednesday; add a make-up class on Friday"): the old format
        # keeps only the investigation, so they are checked as a compile of their own and kept with it. The
        # orchestrator splits such a message into its parts; whoever reads the whole reading sees all of it.
        alone = postprocess(instance, request, legacy.model_copy(update={"action": "compile"}))
        compiled.result.constraints, compiled.result.refusal = alone.constraints, alone.refusal
        compiled.result.errors.extend(alone.errors)
    if compiled.result.refusal is None:
        for i, make in enumerate(direct, len(drafts) + 1):
            c = make(i)
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
    if out.intents and out.intents[0].kind in ("unavailable", "avoid") and drafts:  # as ``_drafts`` decided
        rtype = "unavailability" if drafts[0].type == "unavailable" else "preference"
    if rtype == "club_booking":  # a club booking has no place in the old format: the first other intent's type
        rtype = next((REQUEST_TYPE[i.kind] for i in out.intents if i.kind != "club_booking"), "out_of_scope")
    return ParseOutput(request_type=rtype, action=action, constraints=drafts, missing=out.missing,
                       clarifying_question=out.question)
