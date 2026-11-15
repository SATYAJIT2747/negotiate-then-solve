"""Checks on a parser's reading, before it is resolved: the same for the model and the rule parser.

A reading is what a parser says a message means. Some readings are near misses that one rule
puts right, some contradict the message or the timetable, and the parser cannot be trusted to
notice. These checks look at the reading beside the message and the timetable, and only ever
make it safer: they repair a near miss, make a requirement the message hedges a wish, or turn a
reading that contradicts the timetable into a question. Every change is noted, so a report can
show what was changed and why.

* ``repair_raw``: a JSON answer that misses the schema on a detail ("at": ["all"], "9:00",
  "3 pm") is put right before it is validated, instead of losing the whole answer;
* ``check_reading``: an ``IntentOutput`` beside the message and the timetable:
  - a before/after range that is one time ("after 15:00 and before 15:00") is that start time;
  - a requirement (``hard``) in a message that hedges and says nothing must be so is a wish; and a wish
    in a message of one request that the labelling rule (``lexicon.firmness_rule``) makes a requirement
    ("I need that time kept free", "a hard rule") is a requirement;
  - a period the message does not say is every week: weeks or a "this week" in a message that names no
    week, stretch of the term or date, and "the rest of the semester" for "the whole semester";
  - a move with no destination whose class reference names a time the class is not at reads that
    time as the destination ("I need my lecture on Thursday at 2 pm");
  - a wish to move a class to where it already is is asked about;
  - a reading that refers to no class, of a message that names one of the sender's classes, is asked
    about (is it that class, or every class?): the reading left out what the message is about; unless the
    message says the times are for that course's classes ("any day apart from Friday works for my Data
    Structures classes"), which the reading is then about, teaching only on all days but some read as
    avoiding those days;
  - a move of a course the message does not name, in a message that names one of the sender's, is made up:
    dropped, with what refers to it;
  - a lone move as firm as the labelling rule makes it ("it has to move"), and a clash at the time of the
    class the message moves dropped ("where it is now clashes with a meeting" is why it moves);
  - a club booking has the start and end the message gives as one range ("7-9pm", "from 2 till 4 pm");
  - a reading of room needs only, of a message that names a day the class is not referred to by, gets the
    time back as a move of that class ("needs the routers, and it has to be on Tuesday afternoon": the
    time was left out; the message says it plainly, so asking again would only repeat it);
  - items the message cannot have asked for are dropped (``_untangle``): an item said twice, a choice
    (any_of, ranked) in a message that offers no alternative ("or", "otherwise", "failing that"...) is its
    first item, a second move of the same class (``same``) that no choice holds, and labels (``id``)
    nothing refers to. A reader that has seen labels only in programs writes one on a plain move and
    then makes up the second option the label seemed to call for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from core.instance import Instance
from core.schemas import Placement, Request

from .corpus import FULL_DAY, hour
from .intents import (
    Avoid,
    Cancel,
    ClashReport,
    ClassRef,
    ClubBooking,
    IntentOutput,
    Limit,
    LoadLimit,
    Move,
    NeedsRoom,
    Only,
    Period,
    Relate,
    TimeSpec,
    Unresolved,
    Who,
    classes,
    meeting,
    slots,
)
from .lexicon import (
    acronym,
    firmness_rule,
    hedged,
    mentions,
    names_weeks,
    necessary,
    normalise,
    whole_term,
    words,
)
from .names import course_id

# the kinds whose firmness the message alone decides; a move has its own rule (``_check_move``)
_FIRMABLE = (Avoid, Only, Limit, Relate)

_CLOCK = re.compile(r"^\s*(\d{1,2})(?:[:.](\d\d))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$", re.IGNORECASE)
_DAYS = {d.lower(): d for d in FULL_DAY} | {v.lower(): k for k, v in FULL_DAY.items()}
_WHOLE_DAY = {"all", "any", "all day", "whole day", "full day", "entire day", "day", "anytime", "", "none"}


def clock(value) -> str | None:
    """``value`` as HH:MM ("3 pm" -> "15:00", "9:00" -> "09:00", 14 -> "14:00"), or None when it is not a time.
    A bare hour from 1 to 7 is afternoon: classes run from the morning to the early evening."""
    m = _CLOCK.match(str(value))
    if not m:
        return None
    h, mins, ampm = int(m[1]), int(m[2] or 0), (m[3] or "").lower().replace(".", "")
    if ampm == "pm" and h < 12:
        h += 12
    elif ampm == "am" and h == 12:
        h = 0
    elif not ampm and 1 <= h <= 7:
        h += 12
    return f"{h:02d}:{mins:02d}" if h < 24 and mins < 60 else None


def _repair_time(t: dict, notes: list[str], where: str) -> None:
    if isinstance(t.get("days"), list):
        days = [_DAYS.get(str(d).lower()[:3], _DAYS.get(str(d).lower())) for d in t["days"]]
        if days != t["days"]:
            notes.append(f"{where}: days {t['days']} read as {[d for d in days if d]}")
        t["days"] = [d for d in days if d] or None
    for k in ("before", "after"):
        if k in t and t[k] is not None:
            c = clock(t[k])
            if c != t[k]:
                notes.append(f"{where}: {k} {t[k]!r} read as {c}")
            t[k] = c
    if isinstance(t.get("at"), list):
        whole = [a for a in t["at"] if str(a).strip().lower() in _WHOLE_DAY]
        at = [c for a in t["at"] if (c := clock(a))]
        if at != t["at"]:
            notes.append(f"{where}: at {t['at']} read as {at or 'the whole day'}" + (" (whole day)" if whole else ""))
        t["at"] = at or None
    # one way of giving times: the most specific one the answer gives
    if t.get("before") is not None and t.get("before") == t.get("after"):
        notes.append(f"{where}: after and before {t['before']} read as at {t['before']}")
        t["at"], t["before"], t["after"] = [t["before"]], None, None
    if (t.get("before") or t.get("after")) and (t.get("at") or t.get("part")):
        notes.append(f"{where}: a range with other times; the range kept")
        t["at"], t["part"] = None, None
    elif t.get("at") and t.get("part"):
        notes.append(f"{where}: start times with a part of the day; the start times kept")
        t["part"] = None


def _repair_period(p: dict, notes: list[str], where: str) -> None:
    ways = [k for k in ("weeks", "from_week", "relative") if p.get(k) is not None]
    if len(ways) > 1:
        notes.append(f"{where}: several ways of giving weeks; {ways[0]} kept")
        for k in ("from_week", "to_week", "relative") if ways[0] == "weeks" else ("relative",):
            p[k] = None


def repair_raw(data: dict) -> tuple[dict, list[str]]:
    """``data`` (a parsed JSON answer for ``IntentOutput``) with near misses put right, and what was changed."""
    notes: list[str] = []
    for i, intent in enumerate(data.get("intents") or []):
        if not isinstance(intent, dict):
            continue
        for key in ("time", "to"):
            if isinstance(intent.get(key), dict):
                _repair_time(intent[key], notes, f"intent {i} {key}")
        if isinstance(intent.get("period"), dict):
            _repair_period(intent["period"], notes, f"intent {i} period")
        for key in ("start", "end"):
            if intent.get(key) is not None and (c := clock(intent[key])) != intent[key]:
                notes.append(f"intent {i}: {key} {intent[key]!r} read as {c}")
                intent[key] = c
        for ref in ("which", "mine", "theirs", "other"):
            r = intent.get(ref)
            if isinstance(r, dict):
                if r.get("day") is not None:
                    r["day"] = _DAYS.get(str(r["day"]).lower()[:3], r["day"])
                if r.get("time") is not None:
                    r["time"] = clock(r["time"])
        if intent.get("justification") not in (None, "none", "stated", "verified"):
            intent["justification"] = "stated"
    return data, notes


def read_repaired(text: str) -> tuple[IntentOutput, list[str]]:
    """An answer that failed validation, repaired and validated (raises ``ValueError`` when it cannot be)."""
    import json

    data, notes = repair_raw(json.loads(text))
    return IntentOutput.model_validate(data), notes


@dataclass
class Checked:
    output: IntentOutput
    notes: list[str] = field(default_factory=list)
    question: str | None = None
    missing: list[str] = field(default_factory=list)


def _class_name(instance: Instance, sid: str) -> str:
    s = instance.session_by_id[sid]
    return instance.course_title(s.course) + ("" if s.kind.value == "lecture" else f" {s.kind.value}")


def check_reading(instance: Instance, request: Request, out: IntentOutput,
                  timetable: Mapping[str, Placement] | None = None, *, strict: bool = False) -> Checked:
    """``out`` checked against the message and the timetable (see the module docstring). ``strict`` for a
    reader that only matches patterns (the rule parser): a requirement needs the message to say it must be so."""
    data, notes = repair_raw(out.model_dump(mode="json"))
    try:
        out = IntentOutput.model_validate(data)
    except ValueError:
        notes = []
    text, sender = request.raw_text, request.sender_id
    soft_words = (not necessary(text)) if strict else (hedged(text) and not necessary(text))
    intents, question, missing = [], None, []
    items = _untangle(_named_only(instance, out.intents, text, sender, notes), text, notes)
    for i, intent in enumerate(items):
        if getattr(intent, "hard", False) and soft_words:
            notes.append(f"intent {i} ({intent.kind}): hard, but the message "
                         + ("says nothing must be so" if strict else "hedges and says nothing must be so")
                         + "; read as a wish")
            intent = intent.model_copy(update={"hard": False})
        # the other way, where the message is one request: whose words the must is about is then plain
        elif (len(items) == 1 and isinstance(intent, _FIRMABLE) and not intent.hard
              and firmness_rule(intent.kind, text)):
            notes.append(f"intent {i} ({intent.kind}): a wish, but the message says it must be so; read as a "
                         "requirement")
            intent = intent.model_copy(update={"hard": True})
        elif (len(items) == 1 and isinstance(intent, Move) and not intent.hard
              and firmness_rule("move", text)):
            notes.append(f"intent {i} (move): a wish, but the message says the class has to move; read as a "
                         "requirement")
            intent = intent.model_copy(update={"hard": True})
        intent = _ground_period(intent, text, notes, i)
        if len(items) == 1:
            intent = _restore_weeks(intent, text, notes, i)
        intent = _alternate_weeks(intent, text, notes, i)
        if isinstance(intent, ClubBooking):
            intent = _club_times(intent, text, notes, i)
        if isinstance(intent, Move):
            intent = _same_day(intent, text, notes, i)
            intent, q = _check_move(instance, intent, sender, timetable, notes, i)
            if q and question is None:
                question, missing = q, ["slots"]
        if isinstance(intent, Cancel) and timetable and len(items) == 1:  # with more, the day may be another's
            intent = _cancel_day(instance, intent, text, sender, timetable, notes, i)
        intents.append(intent)
    # a limit on the week is about all of it by its nature: naming a course says why, not which classes
    if (question is None and intents and not any(map(_about_classes, intents))
            and not all(isinstance(i, (Limit, LoadLimit)) for i in intents)):
        named = _own_courses_named(instance, text, sender)
        said = _scope_said(text, named[0]) if len(named) == 1 else None
        if len(named) == 1 and (_FOR_COURSE(named[0]).search(normalise(text)) or said == "course"):
            notes.append(f"the reading is about all your times, in a message that says it is for your {named[0]} "
                         "classes: read as about them")
            intents = [_for_course(x, named[0], instance.calendar.days, text, notes) for x in intents]
        elif len(named) == 1 and said == "all":
            notes.append(f"the reading is about all your times, in a message about {named[0]} that says all your "
                         "classes: left as read")
        elif len(named) == 1 and _teaches_only(instance, sender, named[0]):
            notes.append(f"the reading is about all your times, in a message about {named[0]}, all you teach: "
                         "left as read")
        elif len(named) == 1:
            notes.append(f"the reading is about all your times, in a message about {named[0]}")
            question, missing = (f"Should this apply to your {named[0]} classes only, or to all your classes?",
                                 ["session"])
    if (question is None and intents and all(isinstance(x, NeedsRoom) and x.which.day is None for x in intents)
            and (days := _days_named(text))):
        to = _time_named(text, days)
        notes.append(f"the reading is only about the room, in a message that also says when: read as a move to "
                     f"{' and '.join(days)}{' ' + to.part if to.part else ''}{' at ' + ', '.join(to.at) if to.at else ''}")
        intents += [Move(which=x.which, to=to, hard=necessary(text), justification=x.justification)
                    for x in list(intents)]
    return Checked(out.model_copy(update={"intents": intents}), notes, question, missing)


_ALTERNATIVE = re.compile(r"\b(or|either|otherwise|instead|failing that|first choice|second choice|else|"
                          r"if (?:that|it|this)(?:'s| is)? not|if not|not possible|can't be done|cannot be done|"
                          r"won't fit|doesn't work|does not work|back-?up|next best|alternatively|whichever|"
                          r"(?:don't|do not) mind which|any (?:day|one|of them))\b", re.IGNORECASE)
_CHOICES = ("any_of", "ranked")


def _untangle(intents: list, text: str, notes: list[str]) -> list:
    """``intents`` without what the message cannot have asked for (see the module docstring)."""
    by_id = {x.id: x for x in intents if getattr(x, "id", None)}

    def same_as(x):  # what ``x`` says, with a class named as another item's spelled out
        d = x.model_dump(mode="json", exclude={"id"})
        for n in ("which", "other", "mine", "theirs"):
            ref = (d.get(n) or {}).get("same") if isinstance(d.get(n), dict) else None
            if ref in by_id and isinstance(getattr(by_id[ref], "which", None), ClassRef):
                d[n] = by_id[ref].which.model_dump(mode="json")
        return d

    out, renamed = [], {}
    for x in intents:  # an item said twice is one item; a reference to the copy is to the first
        twin = next((y for y in out if y.kind not in _CHOICES + ("fallback",) and same_as(y) == same_as(x)), None)
        if twin is not None:
            notes.append(f"{x.kind} said twice: read once")
            if getattr(x, "id", None) and getattr(twin, "id", None):
                renamed[x.id] = twin.id
            continue
        out.append(x)
    if renamed:
        out = [x.model_copy(update={"of": list(dict.fromkeys(renamed.get(i, i) for i in x.of))})
               if hasattr(x, "of") else x for x in out]
        out = [x for x in out if not hasattr(x, "of") or len(x.of) >= 2]
    if not _ALTERNATIVE.search(text):  # a choice the message does not offer: its first item
        for c in [x for x in out if x.kind in _CHOICES]:
            notes.append(f"{c.kind}: the message offers no alternative; read as its first item")
            rest = set(c.of[1:])
            out = [x for x in out if x is not c and getattr(x, "id", None) not in rest]
            if getattr(c, "hard", False):  # a must said of the choice is said of what is left of it
                out = [x.model_copy(update={"hard": True}) if getattr(x, "id", None) == c.of[0] and hasattr(x, "hard")
                       else x for x in out]
    moved = {(x.which.day, x.which.time) for x in out if isinstance(x, Move) and x.which.day and x.which.time}
    for x in list(out):  # "where it is now clashes with a meeting": why the class moves, not a clash of classes
        if (isinstance(x, ClashReport) and not x.courses and moved and x.time.days and x.time.at
                and {(d, a) for d in x.time.days for a in x.time.at} <= moved):
            notes.append("a clash at the time of a class the message moves: the reason for the move; dropped")
            out.remove(x)
    held = {i for x in out if hasattr(x, "of") for i in x.of}
    moves = {x.id for x in out if isinstance(x, Move) and x.id}
    for x in list(out):  # a second move of a class already moved, that no choice holds, repeats the first
        if isinstance(x, Move) and x.which.same in moves and x.id not in held:
            notes.append("a second move of the same class, outside any choice: dropped")
            out.remove(x)
    ids = {x.id for x in out if getattr(x, "id", None)}
    for k, x in enumerate(out):  # a class named as an item's that is not there names no class
        for n in ("which", "other", "mine", "theirs"):
            ref = getattr(x, n, None)
            if isinstance(ref, ClassRef) and ref.same and ref.same not in ids:
                notes.append(f"{x.kind}: refers to an item '{ref.same}' that is not there; reference dropped")
                out[k] = x = x.model_copy(update={n: ref.model_copy(update={"same": None})})
    used = {i for x in out if hasattr(x, "of") for i in x.of} | {
        r.same for x in out for n in ("which", "other", "mine", "theirs")
        if isinstance(r := getattr(x, n, None), ClassRef) and r.same}
    return [x.model_copy(update={"id": None}) if getattr(x, "id", None) and x.id not in used else x for x in out]


def _named_only(instance: Instance, intents: list, text: str, sender: str, notes: list[str]) -> list:
    """``intents`` without moves of a course the message does not name, in a message that names one of the
    sender's ("Wednesday morning would suit my Signals and Systems lab": a second lab moved and set apart from
    it is made up), and without what refers to them."""
    if not _own_courses_named(instance, text, sender):
        return intents
    stray = {x.id for x in intents if isinstance(x, Move) and x.id and x.which.course
             and not _names(text, instance.course_title(course_id(instance, x.which.course)))}
    if not stray:
        return intents
    notes.append(f"{len(stray)} move(s) of a course the message does not name: dropped")
    refers = lambda x: any(isinstance(r := getattr(x, n, None), ClassRef) and r.same in stray  # noqa: E731
                           for n in ("which", "other", "mine", "theirs")) or bool(set(getattr(x, "of", [])) & stray)
    return [x for x in intents if getattr(x, "id", None) not in stray and not refers(x)]


def _about_classes(intent) -> bool:
    """Whether ``intent`` is about particular classes (or courses), not about the sender's times."""
    who = getattr(intent, "who", None)
    return (any(isinstance(getattr(intent, n, None), ClassRef) for n in ("which", "mine", "theirs", "other"))
            or (who is not None and who.kind in ("course", "session"))
            or intent.kind in ("clash_report", "club_booking", "room_closed"))


def _days_named(text: str) -> list[str]:
    """The weekdays ``text`` names ("tues" and "Thursdays" too), in the week's order."""
    said = {w.removesuffix("s") for w in words(normalise(text))}
    return [d for d in FULL_DAY.values() if d.lower() in said]


_START = re.compile(r"\b(\d{1,2})(?::(\d\d))?\s*(am|pm)\b")


def _time_named(text: str, days: list[str]) -> TimeSpec:
    """When ``text`` says the class should be: the days, and a part of the day or start times."""
    low = normalise(text)
    codes = [k for k, v in FULL_DAY.items() if v in days]
    if re.search(r"\b(afternoons?|after lunch)\b", low):
        return TimeSpec(days=codes, part="afternoon")
    if re.search(r"\b(mornings?|before lunch)\b", low):
        return TimeSpec(days=codes, part="morning")
    at = sorted({c for m in _START.finditer(low) if (c := clock(m[0]))})
    return TimeSpec(days=codes, at=at or None)


def _own_courses_named(instance: Instance, text: str, sender: str) -> list[str]:
    """The titles of the sender's courses the message names: in full (a typo allowed), or by an acronym in
    capitals ("my ML lecture"; "this is a must" names no IS)."""
    titles = {instance.course_title(s.course) for s in instance.sessions if s.faculty == sender}
    return sorted(t for t in titles if _names(text, t))



def _teaches_only(instance: Instance, sender: str, title: str) -> bool:
    """Every class the sender teaches is of ``title`` (a guest who comes for Cloud Computing alone): "your
    Cloud Computing classes" and "all your classes" are then the same classes, and nothing is asked."""
    return {instance.course_title(s.course) for s in instance.sessions if s.faculty == sender} == {title}


def _names(text: str, title: str) -> bool:
    return mentions(text, title) or (len(a := acronym(title).upper()) >= 2
                                     and re.search(rf"(?<![\w.]){a}(?![\w.])", text) is not None)


def _FOR_COURSE(title: str) -> re.Pattern:
    """ "... works for my Data Structures classes": the message says whose times it is about."""
    return re.compile(rf"\bfor (?:my|the) {re.escape(normalise(title))} (?:class|lecture|lab|practical|tutorial|session)",
                      re.IGNORECASE)


_ALL_CLASSES = re.compile(r"\b(?:all|every|each)\s+(?:of\s+)?(?:my\s+|your\s+|the\s+)?(?:classes|class|lectures|"
                          r"courses|sessions|of them)\b|\beverything\b|\ball\s+(?:my|your)\s+(?:teaching|timetable)\b",
                          re.IGNORECASE)


def _scope_said(text: str, title: str) -> str | None:
    """Whether the message (with the sender's answer, which comes last) says the times are for all their classes
    ("all classes", "every class") or for the one course ("Cloud Computing only", "just Cloud Computing"): the
    answer to "should this apply to your ... classes only, or to all your classes?". The last one said counts."""
    t = normalise(text)
    name = re.escape(normalise(title))
    course = re.compile(rf"\b(?:only|just)\s+(?:(?:my|the|for)\s+)?{name}\b|\b{name}\s+(?:(?:classes|class|lectures?|"
                        rf"courses?)\s+)?only\b", re.IGNORECASE)
    last = max(((m.start(), what) for what, rx in (("all", _ALL_CLASSES), ("course", course)) for m in rx.finditer(t)),
               default=None)
    return last[1] if last else None


def _for_course(intent, title: str, week: list[str], text: str, notes: list[str]):
    """``intent`` about the classes of ``title``; and teaching only on all days but some is avoiding those days
    ("any day apart from Friday works"), as firm as the labelling rule makes an avoidance."""
    if not isinstance(intent, (Only, Avoid)) or intent.who is not None:
        return intent
    intent = intent.model_copy(update={"who": Who(kind="course", id=title)})
    t = intent.time
    if (isinstance(intent, Only) and t.days and not t.part and not t.at and not t.before and not t.after
            and set(t.days) < set(week) and len(t.days) >= len(week) - 2):
        rest = [d for d in week if d not in t.days]
        notes.append(f"only on every day but {', '.join(rest)}: read as avoiding {', '.join(rest)}")
        intent = Avoid(who=intent.who, time=TimeSpec(days=rest), period=intent.period,
                       hard=firmness_rule("avoid", text), justification=intent.justification)
    return intent


_RANGE = re.compile(r"\b(\d{1,2})(?::(\d\d))?\s*(am|pm)?\s*(?:-|–|to|till|until)\s*(\d{1,2})(?::(\d\d))?\s*(am|pm)\b",
                    re.IGNORECASE)


def _club_times(intent: ClubBooking, text: str, notes: list[str], i: int) -> ClubBooking:
    """A booking with the start and end the message gives as one range ("7-9pm", "from 2 till 4 pm"): a reader
    that leaves the end out has the request asked again, one that misreads the start books the wrong hours."""
    ranges = list(_RANGE.finditer(text))
    if len(ranges) != 1:
        return intent
    m = ranges[0]
    end = clock(f"{m[4]}:{m[5] or '00'} {m[6]}")
    start = clock(f"{m[1]}:{m[2] or '00'} {m[3] or m[6]}")
    if start and end and start > end and not m[3]:  # "11-1 pm": the start is in the morning
        start = clock(f"{m[1]}:{m[2] or '00'} am")
    if not start or not end or start >= end or (intent.start, intent.end) == (start, end):
        return intent
    notes.append(f"intent {i} (club_booking): {intent.start or '?'}-{intent.end or '?'}, but the message says "
                 f"{start}-{end}")
    return intent.model_copy(update={"start": start, "end": end})


def _ground_period(intent, text: str, notes: list[str], i: int):
    """``intent`` with a period only where the message gives one: the model's habit of adding "this week" or
    "rest of the semester" to a request that says neither changes which weeks it applies to."""
    period = getattr(intent, "period", None)
    if not isinstance(period, Period) or period.is_recurring:
        return intent
    if not names_weeks(text):
        notes.append(f"intent {i} ({intent.kind}): weeks, but the message names none; read as every week")
    elif period.relative == "rest_of_semester" and whole_term(text):
        notes.append(f"intent {i} ({intent.kind}): the rest of the semester, but the message says the whole "
                     "semester; read as every week")
    else:
        return intent
    return intent.model_copy(update={"period": Period()})


# a week the message names by number: "in week 10", "weeks 10-12", "weeks 10 and 11", "from week 10"
_WEEK_NUMBERS = re.compile(r"\b(from\s+)?weeks?\s+(\d{1,2})(?:\s*(-|–|to|through|and|&)\s*(\d{1,2}))?\b", re.IGNORECASE)
_EVERY_WEEK = re.compile(r"\b(every|each)\s+week\b|\bfrom now on\b|\b(whole|entire|rest of the)\s+(semester|term)\b",
                         re.IGNORECASE)


def _restore_weeks(intent, text: str, notes: list[str], i: int):
    """``intent`` with the weeks the message names by number, when the reading dropped them: "My lecture has
    to be on Monday at 11 am in week 10" read as every week would move the class for the whole semester.
    Only for a message that is one request and names one span of weeks, and says nothing of every week."""
    period = getattr(intent, "period", None)
    if not isinstance(period, Period) or not period.is_recurring or _EVERY_WEEK.search(text):
        return intent
    spans = _WEEK_NUMBERS.findall(text)
    if len(spans) != 1:
        return intent
    start, a, joint, b = spans[0]
    a = int(a)
    if b and joint.lower() in ("and", "&"):
        new = Period(weeks=sorted({a, int(b)}))
    elif b:
        new = Period(from_week=min(a, int(b)), to_week=max(a, int(b)))
    elif start:
        new = Period(from_week=a)
    else:
        new = Period(weeks=[a])
    try:
        out = intent.model_copy(update={"period": new})
        type(intent).model_validate(out.model_dump())
    except ValueError:
        return intent
    notes.append(f"intent {i} ({intent.kind}): every week, but the message names {spans[0][1]}"
                 f"{(' ' + joint + ' ' + b) if b else ''}; read as those weeks")
    return out


# "alternate weeks from week 9", "every other week, starting in week 9", "every second week from week 9"
_ALTERNATE = re.compile(r"\b(?:alternate|every\s+other|every\s+second)\s+weeks?\b", re.IGNORECASE)


def _alternate_weeks(intent, text: str, notes: list[str], i: int):
    """``intent`` with the odd or even weeks the message means, when the reading kept the start week and dropped
    the "every other week": "Alternate weeks from week 9: Wednesdays are out for me" read as every week from 9
    takes Wednesday from eight weeks instead of four. The parity is the start week's (9: odd)."""
    period = getattr(intent, "period", None)
    if (not isinstance(period, Period) or period.parity is not None or period.from_week is None
            or period.weeks or not _ALTERNATE.search(text)):
        return intent
    parity = "odd" if period.from_week % 2 else "even"
    try:
        out = intent.model_copy(update={"period": period.model_copy(update={"parity": parity})})
        type(intent).model_validate(out.model_dump())
    except ValueError:
        return intent
    notes.append(f"intent {i} ({intent.kind}): every week from week {period.from_week}, but the message says "
                 f"every other week; read as the {parity} weeks")
    return out


def _check_move(instance: Instance, intent: Move, sender: str, timetable: Mapping[str, Placement] | None,
                notes: list[str], i: int) -> tuple[Move, str | None]:
    ref, to = intent.which, intent.to
    # the reading names courses by title, the timetable by ID: looked up by ID, kept as the reading wrote it
    by_id = (lambda r: r.model_copy(update={"course": course_id(instance, r.course)}) if r.course else r)  # noqa: E731
    bare = ref.model_copy(update={"day": None, "time": None})
    if (ref.day or ref.time) and timetable and not _resolves(instance, by_id(ref), sender, timetable)             and _resolves(instance, by_id(bare), sender, timetable):
        if not to.has_times and to.days is None:
            target = TimeSpec(days=[ref.day] if ref.day else None, at=[ref.time] if ref.time else None)
            notes.append(f"intent {i}: no class of yours is at {ref.day or ''} {ref.time or ''}; read as where the "
                         "class should go")
            intent = intent.model_copy(update={"which": bare, "to": target})
            to = target
        elif ref.day in (to.days or [ref.day]) and ref.time in (to.at or [ref.time]):
            notes.append(f"intent {i}: no class of yours is at {ref.day or ''} {ref.time or ''}, where the class "
                         "should go; read as where it should go only")
            intent = intent.model_copy(update={"which": bare})
    if not timetable or not (to.has_times or to.days):
        return intent, None
    try:
        sid = meeting(instance, by_id(intent.which), sender, timetable, to.days)
        wanted = slots(instance, to)
    except Unresolved:
        return intent, None
    p = timetable.get(sid)
    # a soft move to where the class is changes nothing, and is likely a slip; a hard one keeps it there
    if not intent.hard and p is not None and wanted == [p.slot] and to.days in (None, [p.day]):
        notes.append(f"intent {i}: a move to where the class already is")
        return intent, (f"Your {_class_name(instance, sid)} is already on {FULL_DAY[p.day]} at {hour(p.slot)}. "
                        "When would you like it instead?")
    return intent, None


def _same_day(intent: Move, text: str, notes: list[str], i: int) -> Move:
    """A move to a part of the day or an hour, in a message that names no day but the one the class is on now ("move
    my Friday 9 am class to the afternoon"), stays on that day: the reader gave it other days the message never
    names."""
    ref, to = intent.which, intent.to
    if ref.day is None or not to.days or set(to.days) == {ref.day} or not (to.part or to.at or to.after or to.before):
        return intent
    named = {k for k, v in FULL_DAY.items() if v in _days_named(text)}
    if not named <= {ref.day}:
        return intent
    notes.append(f"intent {i} (move): to {', '.join(to.days)}, days the message does not name; it names only "
                 f"{ref.day}, where the class is: read as later or earlier the same day")
    return intent.model_copy(update={"to": to.model_copy(update={"days": [ref.day]})})


def _cancel_day(instance: Instance, intent: Cancel, text: str, sender: str, timetable: Mapping[str, Placement],
                notes: list[str], i: int) -> Cancel:
    """A cancellation whose class the reading left without the day the message names ("cancel my AI lecture next
    Wednesday"): the class on that day, when that picks one class and the reading alone picks none. Only for a
    message of one request: in "move it to Thursday; if no lab is free, cancel it" the day is the move's."""
    ref = intent.which
    days = [k for k, v in FULL_DAY.items() if v in _days_named(text)]
    if ref.day is not None or ref.session is not None or len(days) != 1:
        return intent
    by_id = (lambda r: r.model_copy(update={"course": course_id(instance, r.course)}) if r.course else r)  # noqa: E731
    on_day = ref.model_copy(update={"day": days[0]})
    try:
        if len(classes(instance, by_id(ref), sender, timetable)) < 2:
            return intent  # the reading already picks one class (or none: that is asked about as before)
        if len(classes(instance, by_id(on_day), sender, timetable)) != 1:
            return intent
    except Unresolved:
        return intent
    notes.append(f"intent {i} (cancel): the message names {days[0]}, which the reading left out: read as the class "
                 "on that day")
    return intent.model_copy(update={"which": on_day})


def _resolves(instance: Instance, ref: ClassRef, sender: str, timetable: Mapping[str, Placement]) -> bool:
    try:
        classes(instance, ref, sender, timetable)
    except Unresolved:
        return False
    return True


def checked_compile(instance: Instance, request: Request, out: IntentOutput, *,
                    timetable: Mapping[str, Placement] | None = None, current_week: int | None = None,
                    notes: list[str] | None = None, strict: bool = False):
    """``compile_intents`` on ``out`` after ``check_reading``: what a parser in use does with a reading
    (``notes``: what was already repaired in the answer). A question from the checks stands for the whole
    request, as a question from resolution does."""
    from .intents import compile_intents
    from .parsing import ParseOutput, postprocess

    checked = check_reading(instance, request, out, timetable, strict=strict)
    compiled = compile_intents(instance, request, checked.output, timetable=timetable, current_week=current_week)
    compiled.notes = [*(notes or []), *checked.notes]
    if checked.question is not None and compiled.question is None:
        compiled.question, compiled.missing = checked.question, checked.missing
        compiled.swaps, compiled.extras, compiled.cancels, compiled.clubs = [], [], [], []
        compiled.result = postprocess(instance, request, ParseOutput(
            request_type=checked.output.request_type, action="clarify", missing=checked.missing,
            clarifying_question=checked.question))
    return compiled
