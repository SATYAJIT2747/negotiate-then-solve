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
  - a requirement (``hard``) in a message that hedges and says nothing must be so is a wish;
  - a period the message does not say is every week: weeks or a "this week" in a message that names no
    week, stretch of the term or date, and "the rest of the semester" for "the whole semester";
  - a move with no destination whose class reference names a time the class is not at reads that
    time as the destination ("I need my lecture on Thursday at 2 pm");
  - a wish to move a class to where it already is is asked about;
  - a reading that refers to no class, of a message that names one of the sender's classes, is asked
    about (is it that class, or every class?): the reading left out what the message is about.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from core.instance import Instance
from core.schemas import Placement, Request

from .corpus import FULL_DAY, hour
from .intents import (
    ClassRef,
    IntentOutput,
    Move,
    Period,
    TimeSpec,
    Unresolved,
    classes,
    meeting,
    slots,
)
from .lexicon import acronym, hedged, mentions, names_weeks, necessary, whole_term

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
        for ref in ("which", "mine", "theirs"):
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
    for i, intent in enumerate(out.intents):
        if getattr(intent, "hard", False) and soft_words:
            notes.append(f"intent {i} ({intent.kind}): hard, but the message "
                         + ("says nothing must be so" if strict else "hedges and says nothing must be so")
                         + "; read as a wish")
            intent = intent.model_copy(update={"hard": False})
        intent = _ground_period(intent, text, notes, i)
        if isinstance(intent, Move):
            intent, q = _check_move(instance, intent, sender, timetable, notes, i)
            if q and question is None:
                question, missing = q, ["slots"]
        intents.append(intent)
    if question is None and intents and not any(map(_about_classes, intents)):
        named = _own_courses_named(instance, text, sender)
        if len(named) == 1:
            notes.append(f"the reading is about all your times, in a message about {named[0]}")
            question, missing = (f"Should this apply to your {named[0]} classes only, or to all your classes?",
                                 ["session"])
    return Checked(out.model_copy(update={"intents": intents}), notes, question, missing)


def _about_classes(intent) -> bool:
    """Whether ``intent`` is about particular classes (or courses), not about the sender's times."""
    who = getattr(intent, "who", None)
    return (any(isinstance(getattr(intent, n, None), ClassRef) for n in ("which", "mine", "theirs"))
            or (who is not None and who.kind in ("course", "session"))
            or intent.kind in ("clash_report", "club_booking", "room_closed"))


def _own_courses_named(instance: Instance, text: str, sender: str) -> list[str]:
    """The titles of the sender's courses the message names: in full (a typo allowed), or by an acronym in
    capitals ("my ML lecture"; "this is a must" names no IS)."""
    titles = {instance.course_title(s.course) for s in instance.sessions if s.faculty == sender}
    return sorted(t for t in titles if mentions(text, t)
                  or (len(a := acronym(t).upper()) >= 2 and re.search(rf"(?<![\w.]){a}(?![\w.])", text)))


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


def _check_move(instance: Instance, intent: Move, sender: str, timetable: Mapping[str, Placement] | None,
                notes: list[str], i: int) -> tuple[Move, str | None]:
    ref, to = intent.which, intent.to
    if not to.has_times and to.days is None and (ref.day or ref.time) and timetable:
        try:
            classes(instance, ref, sender, timetable)
        except Unresolved:
            bare = ref.model_copy(update={"day": None, "time": None})
            try:
                classes(instance, bare, sender, timetable)
            except Unresolved:
                return intent, None
            target = TimeSpec(days=[ref.day] if ref.day else None, at=[ref.time] if ref.time else None)
            notes.append(f"intent {i}: no class of yours is at {ref.day or ''} {ref.time or ''}; read as where the "
                         "class should go")
            intent = intent.model_copy(update={"which": bare, "to": target})
            to = target
    if not timetable or not (to.has_times or to.days):
        return intent, None
    try:
        sid = meeting(instance, intent.which, sender, timetable, to.days)
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
