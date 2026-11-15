"""Grounded, contrastive, private explanations (proposal Section 8.7).

A conflict is turned into numbered facts: the constraints in the MUS, the
rules they come from, which rooms have the equipment, the solver-verified
options and the ledger history. Explanations are built only from these:

* ``template``: deterministic sentences, one per fact (no LLM);
* ``grounded``: the LLM writes claims, each citing fact IDs; a verifier drops
  claims that cite nothing, cite unknown facts, or mention a day, number or
  name that is not in the facts they cite;
* ``free`` (ablation A3): the LLM writes freely from the raw requests; the
  same checks then measure how many sentences can be traced.

Facts name the constraint, never the private reason, so leakage can only
come from the model; ``leaks`` measures it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from core.instance import Instance
from core.schemas import Constraint, ConstraintType, Placement, Tier, When
from language.corpus import FIRST_SLOT_HOUR, FULL_DAY, hour, hour_range
from language.llm import LLMError
from language.paraphrase import FidelityCheck

from .policy import _CLOCK, _clock

TIER_NAME = {
    Tier.PHYSICAL: "physical", Tier.POLICY: "institute policy", Tier.COMMITMENT: "institutional commitment",
    Tier.VERIFIED_UNAVAILABILITY: "verified unavailability", Tier.OPERATIONAL: "operational requirement",
    Tier.PREFERENCE: "preference",
}
PRIVATE_TERMS = ("medical", "hospital", "doctor", "family", "surgery", "illness", "sick", "pregnan",
                 "funeral", "clinical", "therapy", "health", "personal matter", "wedding")


@dataclass(frozen=True)
class Fact:
    """``text`` is what the model is given and its claims are checked against: it names sessions by ID and
    tiers by number, and stays stable so prompts and cached answers do too. ``plain`` says the same to the
    person reading the message: no IDs, no tier codes, "you" for the recipient."""

    id: str
    text: str
    plain: str | None = None

    @property
    def reader(self) -> str:
        """``""`` when another fact's sentence already says this one (one sentence per class)."""
        return self.text if self.plain is None else self.plain


@dataclass(frozen=True)
class OptionView:
    """One solver-verified option as the recipient sees it."""

    key: str
    placements: dict[str, Placement]
    drop: tuple[str, ...]
    moved: int | None = None  # other classes that move


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _slots(slots: list[int]) -> str:
    slots = sorted(slots)
    if slots == list(range(slots[0], slots[-1] + 1)) and len(slots) > 1:
        return f"from {hour(slots[0])} to {hour(slots[-1] + 1)}"
    return "at " + ", ".join(hour(s) for s in slots)


def _runs(weeks: list[int]) -> str:
    """Sorted weeks as runs: "4-7 and 9-16", "14 and 16" (a gap is never read as a range)."""
    runs, start = [], weeks[0]
    for prev, w in zip(weeks, weeks[1:] + [None]):
        if w != prev + 1 if w is not None else True:
            runs.append(str(start) if start == prev else f"{start}-{prev}")
            start = w
    return runs[0] if len(runs) == 1 else ", ".join(runs[:-1]) + " and " + runs[-1]


def _when(instance: Instance, c: Constraint) -> str:
    parts = []
    if c.when.days:
        parts.append("on " + ", ".join(FULL_DAY[d] for d in c.when.days))
    if c.when.slots is not None:
        lunch = instance.calendar.lunch_slot
        afternoon = list(range(lunch + 1, instance.calendar.slots_per_day)) if lunch is not None else None
        parts.append("in the afternoon" if sorted(c.when.slots) == afternoon else _slots(c.when.slots))
    if c.when.weeks:
        w = sorted(c.when.weeks)
        parts.append(f"in week {w[0]}" if len(w) == 1 else f"in weeks {_runs(w)}")
    return " ".join(parts)


def session_name(instance: Instance, sid: str) -> str:
    s = next((x for x in instance.sessions if x.id == sid), None)
    if s is None:
        return sid
    return f"the {instance.course_title(s.course)} {s.kind.value} ({sid})"


def placement_text(instance: Instance, p: Placement, session: str | None = None) -> str:
    """Where a class is; given the ``session``, a class longer than an hour says when it ends too."""
    room = instance.room_by_id[p.room].name if p.room in instance.room_by_id else p.room
    s = instance.session_by_id.get(session) if session else None
    if s is not None and s.duration > 1:
        return f"{FULL_DAY[p.day]} from {hour_range(p.slot, s.duration, ' to ')} in {room}"
    return f"{FULL_DAY[p.day]} at {hour(p.slot)} in {room}"


def _body(instance: Instance, c: Constraint, you: str | None = None) -> str:
    """Who, what and when. With ``you``, the reader's own constraint is said to them, without IDs."""
    own = you is not None and c.owner == you
    who = "You" if own else instance.faculty_by_id[c.owner].name if c.owner in instance.faculty_by_id else (
        "The lab in-charge" if c.owner and c.owner.startswith("S-LAB") else "The institute")
    s = c.scope
    if s.session:
        x = instance.session_by_id.get(s.session)
        what = (session_name(instance, s.session) if you is None or x is None else
                f"{'your' if own else 'the'} {instance.course_title(x.course)} {x.kind.value}")
    else:
        what = (f"{instance.room_by_id[s.room].name}" if s.room and s.room in instance.room_by_id else
                f"{instance.group_by_id[s.group].name}'s classes" if s.group else
                f"the {instance.course_title(s.course)} classes" if s.course else
                ("your classes" if own else "their classes") if s.faculty and s.faculty == c.owner else "all classes")
    when = _when(instance, c)
    if c.type == ConstraintType.UNAVAILABLE:
        body = (f"{what[0].upper() + what[1:]} is unavailable {when}" if s.room else
                f"{who} {'are' if own else 'is'} unavailable {when}")
    elif c.type == ConstraintType.PREFER:
        body = f"{who} {('need' if own else 'needs') if c.hard else 'would like'} {what} {when}"
    elif c.type == ConstraintType.AVOID:
        body = f"{who} {'cannot have' if c.hard else 'would rather not have'} {what} {when}"
    elif c.type == ConstraintType.REQUIRE_ROOM:
        need = (", ".join(c.room.equipment) if c.room and c.room.equipment else
                " or ".join(instance.room_by_id[r].name for r in (c.room.rooms or [])) if c.room else "")
        verb = "need" if own else "needs"
        body = f"{who} {verb} {what} in a room with {need}" if c.room and c.room.equipment else \
            f"{who} {verb} {what} in {need}"
    elif c.type == ConstraintType.LOAD:
        verb = ("need" if own else "needs") if c.hard else "would like"
        body = (f"{who} {verb} at least {c.limit} free hour{'s' * (c.limit != 1)} between two classes"
                if c.measure == "gap" else f"{who} {verb} no more than {c.limit} " + {
                    "classes_per_day": "classes a day", "hours_per_day": "hours of teaching a day",
                    "days_per_week": "teaching days a week"}[c.measure])
    elif c.type == ConstraintType.RELATE:
        body = _relation(instance, c, who, own)
    elif c.type in (ConstraintType.ANY_OF, ConstraintType.RANKED):
        parts = [_option(_body(instance, m, you), who) for m in c.members or []]
        body = (f"{who} would like, in order of preference: " + "; else ".join(parts)
                if c.type == ConstraintType.RANKED else f"{who} {('need' if own else 'needs') if c.hard else 'would like'}"
                f" one of: " + "; or ".join(parts))
    else:
        body = f"No one may teach more than {c.limit} hours in a row"
    return body.strip()


def _option(body: str, who: str) -> str:
    """A choice's member without the speaker and verb the choice already says ("needs X on Tuesday": X on
    Tuesday)."""
    body = body.removeprefix(f"{who} ")
    for verb in ("needs ", "need ", "would like "):
        if body.startswith(verb):
            return body.removeprefix(verb)
    return body[0].lower() + body[1:]


def _classes(instance: Instance, sids: Iterable[str]) -> str:
    """Sessions in words: "the Algorithms lectures" when they are one course's of one kind, else each."""
    ss = [instance.session_by_id[s] for s in sids if s in instance.session_by_id]
    if len(ss) > 1 and len({(s.course, s.kind) for s in ss}) == 1:
        return f"the {instance.course_title(ss[0].course)} {ss[0].kind.value}s"
    return " and ".join(session_name(instance, s.id) for s in ss)


_TOGETHER = {"same_day": "on the same day", "different_day": "on different days", "same_hour": "at the same hour",
             "same_room": "in the same room"}
_AGAINST = {"same_day": "on the same day as", "different_day": "on a different day from", "right_after": "right after",
            "after": "after, on the same day,", "same_hour": "at the same hour as", "same_room": "in the same room as"}


def _relation(instance: Instance, c: Constraint, who: str, own: bool) -> str:
    verb = ("need" if own else "needs") if c.hard else "would like"
    mine = _classes(instance, c.sessions or [])
    if c.relation == "day_gap":
        apart = f"at least {c.amount or 1} free day{'s' * ((c.amount or 1) != 1)} apart"
        if c.others is None:
            return f"{who} {verb} {mine} {'not ' * c.negate}{apart}"
        return f"{who} {verb} {mine} {'not ' * c.negate}{apart} from {_classes(instance, c.others)}"
    if c.others is None:
        return f"{who} {verb} {mine} {'not ' * c.negate}{_TOGETHER.get(c.relation, c.relation)}"
    against = ("one of " if c.any_other else "") + _classes(instance, c.others)
    return f"{who} {verb} {mine} {'not ' * c.negate}{_AGAINST[c.relation]} {against}"


def describe(instance: Instance, c: Constraint) -> str:
    """One constraint in plain words: who, what, when, tier. Never the reason."""
    return f"{_body(instance, c)} (Tier {int(c.tier)}, {TIER_NAME[c.tier]})."


# what a fixed constraint is, to someone who cannot change it; Tiers 3-5 need no note
FIXED_NOTE = {Tier.PHYSICAL: "this cannot be changed", Tier.POLICY: "an institute rule",
              Tier.COMMITMENT: "an agreed commitment"}


def plain_describe(instance: Instance, c: Constraint, you: str | None) -> str:
    """One constraint to the person reading the message. A rule says which rule it is in the same sentence
    ("No class can be held at 1 pm: that is the institute's rule on common lunch break.")."""
    body = _body(instance, c, you or "")
    if body.startswith("The institute is unavailable "):
        body = "No class can be held " + body.removeprefix("The institute is unavailable ")
    if c.source.rule:
        return f"{body}: that is the institute's rule on {c.source.rule}."
    note = FIXED_NOTE.get(c.tier)
    return f"{body}{f' ({note})' if note else ''}."


def _only_free(instance: Instance, cs: list[Constraint], you: str | None,
               placed: Mapping[str, Placement] | None) -> str | None:
    """A person's unavailabilities said as the one time they are free: "Dr. Mehta can only teach on Friday in
    the afternoon, and their Computer Graphics practical is in Lab 1 then." None unless every constraint is
    an every-week unavailability of one person and what is left is part of one day."""
    if not cs or any(c.type != ConstraintType.UNAVAILABLE or c.when.weeks or not c.scope.faculty
                     or c.scope.faculty != cs[0].scope.faculty or c.scope.session or c.scope.room for c in cs):
        return None
    cal = instance.calendar
    free = {(d, q) for d in cal.days for q in range(cal.slots_per_day) if q != cal.lunch_slot}
    for c in cs:
        free -= {(d, q) for d in (c.when.days or cal.days)
                 for q in (c.when.slots if c.when.slots is not None else range(cal.slots_per_day))}
    days = {d for d, _ in free}
    if len(days) != 1:
        return None
    day = days.pop()
    window = Constraint(id="FREE", type=ConstraintType.UNAVAILABLE, hard=True, tier=cs[0].tier,
                        scope=cs[0].scope, when=When(days=[day], slots=sorted(q for _, q in free)))
    fac = cs[0].scope.faculty
    own = you is not None and fac == you
    who = "You" if own else instance.faculty_by_id[fac].name if fac in instance.faculty_by_id else "They"
    text = f"{who} can only teach {_when(instance, window)}"
    there = [(s, placed[s.id]) for s in instance.sessions
             if s.faculty == fac and placed and s.id in placed and (placed[s.id].day, placed[s.id].slot) in free]
    if len(there) == 1:
        s, p = there[0]
        room = instance.room_by_id[p.room].name if p.room in instance.room_by_id else p.room
        text += (f", and {'your' if own else 'their'} {instance.course_title(s.course)} {s.kind.value} "
                 f"is in {room} then")
    return text + "."


EQUIPMENT_WORDS = {"gpu": "GPUs"}


def _equipment(names: Iterable[str]) -> str:
    return _and([EQUIPMENT_WORDS.get(e, e) for e in names])


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def plain_group(instance: Instance, cs: list[Constraint], you: str | None,
                placed: Mapping[str, Placement] | None = None) -> str:
    """One sentence for one person's constraints on one class: "You need your ML practical on Tuesday in the
    afternoon, in a room with GPUs." A person's unavailabilities are said as when they are free
    (``_only_free``). Anything else gets a sentence per constraint."""
    if (free := _only_free(instance, cs, you, placed)) is not None:
        return free
    times = [c for c in cs if c.type == ConstraintType.PREFER]
    rooms = [c for c in cs if c.type == ConstraintType.REQUIRE_ROOM and c.room and c.room.equipment]
    if len(cs) < 2 or len(times) > 1 or len(times) + len(rooms) != len(cs) or not cs[0].scope.session:
        return " ".join(plain_describe(instance, c, you) for c in cs)
    base = times[0] if times else rooms[0]
    body = _body(instance, base, you or "") if times else _body(instance, base, you or "").split(" in a room")[0]
    need = _equipment(sorted({e for c in rooms for e in c.room.equipment}))
    note = next((FIXED_NOTE[c.tier] for c in cs if c.tier in FIXED_NOTE), None)
    return f"{body}, in a room with {need}{f' ({note})' if note else ''}."


def _as_asked(c: Constraint, sid: str, p: Placement) -> bool:
    return (c.type == ConstraintType.PREFER and c.scope.session == sid and
            (not c.when.days or p.day in c.when.days) and (c.when.slots is None or p.slot in c.when.slots))


def plain_option(instance: Instance, o: OptionView, asked: Iterable[Constraint], you: str | None,
                 dropped: Iterable[Constraint] = ()) -> str:
    """"Friday at 2 pm in Lab 3, as you asked (2 other classes move)"; an option that gives up a room
    requirement says so: "Wednesday at 3 pm in Lab 3, without routers"."""
    asked, dropped = list(asked), list(dropped)
    if (compact := _compact_option(instance, o, asked, you, dropped)) is not None:
        return compact + _moved_note(o)
    parts = []
    for sid, p in sorted(o.placements.items()):
        s = instance.session_by_id.get(sid)
        where = placement_text(instance, p)
        if len(o.placements) > 1 and s is not None:
            where = f"{'your' if s.faculty == you else 'the'} {instance.course_title(s.course)} {s.kind.value} on {where}"
        room = instance.room_by_id.get(p.room)
        lost = sorted({e for c in dropped if c.type == ConstraintType.REQUIRE_ROOM and c.scope.session == sid
                       and c.room for e in c.room.equipment if room is None or e not in room.equipment})
        if lost:
            where += f", without {_equipment(lost)}"
        if any(_as_asked(c, sid, p) for c in asked):
            where += ", as you asked" if not lost else ", at the time you asked"
        parts.append(where)
    return ("; ".join(parts) or "the timetable stays as it is") + _moved_note(o)


def _moved_note(o: OptionView) -> str:
    if o.moved is None:
        return ""
    return (" (nothing else moves)" if o.moved == 0 else
            f" ({o.moved} other class{'es' if o.moved != 1 else ''} move{'s' if o.moved == 1 else ''})")


def _compact_option(instance: Instance, o: OptionView, asked: list[Constraint], you: str | None,
                    dropped: list[Constraint]) -> str | None:
    """Several classes of one course and kind, said as a short list by day: "your Computer Networks lectures
    on Monday at 9 am, 10 am and 12 pm, and Tuesday at 9 am, all in Room 101". A room that differs is named
    at its time. None for a single class, a mix of classes, or one that gives up equipment."""
    ss = [instance.session_by_id.get(sid) for sid in o.placements]
    if len(ss) < 2 or None in ss or len({(s.course, s.kind) for s in ss}) != 1 or any(
            c.type == ConstraintType.REQUIRE_ROOM for c in dropped):
        return None
    s0 = ss[0]
    rooms = {p.room for p in o.placements.values()}
    name = lambda r: instance.room_by_id[r].name if r in instance.room_by_id else r  # noqa: E731
    days: dict[str, dict[str, list[str]]] = {}  # day -> room -> times
    for sid, p in sorted(o.placements.items(), key=lambda x: (instance.calendar.days.index(x[1].day)
                                                               if x[1].day in instance.calendar.days else 9, x[1].slot)):
        days.setdefault(p.day, {}).setdefault(p.room, []).append(hour(p.slot))
    when = "; ".join(f"{FULL_DAY[d]} at " + " and ".join(
        _and(ts) + (f" in {name(r)}" if len(rooms) > 1 else "") for r, ts in by_room.items())
        for d, by_room in days.items())
    whose = "your" if s0.faculty == you else "the"
    text = f"{whose} {instance.course_title(s0.course)} {s0.kind.value}s on {when}"
    if len(rooms) == 1:
        text += f", all in {name(rooms.pop())}"
    kept = sum(any(_as_asked(c, sid, p) for c in asked) for sid, p in o.placements.items())
    if kept:
        text += f" ({kept} of them at the time{'s' * (kept != 1)} you asked)" if kept < len(ss) else ", as you asked"
    return text


def conflict_facts(
    instance: Instance,
    constraints: Mapping[str, Constraint],
    mus: Iterable[str],
    options: Iterable[OptionView] = (),
    ledger_counts: Mapping[str, float] | None = None,
    recipient: str | None = None,
    placed: Mapping[str, Placement] | None = None,
) -> list[Fact]:
    """``placed``: the timetable as it stands, to say where another person's class is when they can only
    teach at one time. Ledger facts give everyone's count, the reader's too, once anyone has given way."""
    facts: list[Fact] = []
    involved = [constraints[i] for i in mus]
    groups: dict[tuple, list[Constraint]] = {}
    for c in involved:
        groups.setdefault((c.owner, tuple(c.scope.model_dump().values())), []).append(c)
    for c in involved:
        group = groups[(c.owner, tuple(c.scope.model_dump().values()))]
        plain = plain_group(instance, group, recipient, placed) if c is group[0] else ""
        facts.append(Fact(c.id, describe(instance, c), plain))
        if c.source.rule:  # the reader's sentence for the constraint names the rule already
            facts.append(Fact(f"RULE-{c.id}", f"This comes from the institute rule: {c.source.rule} [{c.id}].",
                              "" if plain else f"That is the institute's rule on {c.source.rule}."))
    equipment = sorted({e for c in involved if c.room for e in c.room.equipment})
    holders = {eq: tuple(r.name for r in instance.rooms if eq in r.equipment) for eq in equipment}
    said: set[tuple] = set()
    for eq in equipment:
        rooms = holders[eq]
        plain = ""
        if rooms not in said:  # "Only Lab 2 has GPUs and routers."
            said.add(rooms)
            same = [e for e in equipment if holders[e] == rooms]
            plain = (f"No room has {_equipment(same)}." if not rooms else
                     f"{rooms[0]} is the only room with " + " and the only room with ".join(
                         EQUIPMENT_WORDS.get(e, e) for e in same) + "." if len(rooms) == 1 else
                     f"Only {_and(list(rooms))} have {_equipment(same)}.")
        facts.append(Fact(f"RES-{eq}", f"Rooms with {eq}: {', '.join(rooms) if rooms else 'none'}.", plain))
    for o in options:
        where = "; ".join(f"{session_name(instance, s)} on {placement_text(instance, p)}"
                          for s, p in sorted(o.placements.items()))
        facts.append(Fact(f"OPT-{o.key}", f"Option {o.key}: {where}. The solver confirms this works.",
                          plain_option(instance, o, (c for c in involved if c.owner == recipient), recipient,
                                       (constraints[i] for i in o.drop if i in constraints))))
    owners = {c.owner for c in involved if c.owner in instance.faculty_by_id}
    counted = any((ledger_counts or {}).get(o, 0) for o in owners)
    for owner in sorted(owners):
        n = (ledger_counts or {}).get(owner, 0)
        if not n and counted:  # "Dr. Pillai has given way twice" says little without the reader's own count
            name = instance.faculty_by_id[owner].name
            facts.append(Fact(f"LED-{owner}", f"{name} has conceded 0 times before.",
                              "You haven't given way before." if owner == recipient else
                              f"{name} hasn't given way before."))
        elif n:
            times = f"{n:g} time{'s' if n != 1 else ''}"
            often = "once" if n == 1 else "twice" if n == 2 else times
            name = instance.faculty_by_id[owner].name
            facts.append(Fact(f"LED-{owner}", f"{name} has conceded {times} before.",
                              f"You have given way {often} before." if owner == recipient else
                              f"{name} has given way {often} before."))
    return facts


# ---------------------------------------------------------------------------
# Explanations and their checks
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    text: str
    facts: list[str]


class ExplanationOut(BaseModel):
    claims: list[Claim]


class FreeExplanationOut(BaseModel):
    text: str


@dataclass
class Explanation:
    mode: str
    claims: list[tuple[str, list[str], bool]] = field(default_factory=list)  # (text, facts, supported)
    rejected: int = 0
    leaks: list[str] = field(default_factory=list)
    body: str | None = None  # the full text, when the model wrote free prose or the template laid it out
    lead: str | None = None  # the template's explanation without its option list (messages list options apart)

    @property
    def text(self) -> str:
        return self.body if self.body is not None else " ".join(t for t, _, ok in self.claims if ok)

    @property
    def faithfulness(self) -> float:
        """Share of claims traceable to facts (H2a)."""
        anchored = [ok for _, _, ok in self.claims]
        return sum(anchored) / len(anchored) if anchored else 1.0


def _canon(text: str) -> str:
    return _CLOCK.sub(_clock, text)


class ClaimChecker:
    """A claim is supported when every day, number and name it mentions
    appears in the facts it cites."""

    def __init__(self, instance: Instance) -> None:
        self.check = FidelityCheck(instance)

    def anchors(self, text: str) -> tuple[set[str], set[str], set[str]]:
        t = _canon(text)
        return self.check.days(t), self.check.numbers(t), self.check.entities(t)

    def supported(self, text: str, cited: Iterable[Fact]) -> bool:
        cited = list(cited)
        if not cited:
            return False
        days, nums, ents = self.anchors(text)
        fd, fn, fe = set(), set(), set()
        for f in cited:
            d, n, e = self.anchors(f.text)
            fd |= d
            fn |= n
            fe |= e
        return days <= fd and nums <= fn and ents <= fe

    def traced(self, sentence: str, facts: Iterable[Fact]) -> bool | None:
        """For free text: ``None`` if the sentence mentions nothing checkable,
        else whether all its anchors appear somewhere in the facts."""
        days, nums, ents = self.anchors(sentence)
        if not (days or nums or ents):
            return None
        return self.supported(sentence, facts)


_AT = re.compile(r"\b(" + "|".join(FULL_DAY.values()) + r")s?\b[^.;!?]{0,30}?\bat\s+h(\d+)\b")


def times_at(text: str) -> set[tuple[str, int]]:
    """Every "<day> ... at <time>" in ``text``, as (day, hour)."""
    return {(m[1], int(m[2])) for m in _AT.finditer(_canon(text))}


def option_mismatches(text: str, options: Iterable[Mapping[str, Placement]], facts: Iterable[Fact],
                      raw_requests: Iterable[str] = ()) -> tuple[int, int]:
    """Does the prose agree with the options listed under it? Returns
    (invented, omitted): day-and-time mentions that are neither an offered
    placement nor a time in the conflict facts or requests, and offered
    options the prose never mentions."""
    offered = [{(FULL_DAY[p.day], FIRST_SLOT_HOUR + p.slot) for p in o.values()} for o in options]
    known = set().union(*offered, *(times_at(f.text) for f in facts if not f.id.startswith("OPT-")),
                        *(times_at(r) for r in raw_requests))
    said = times_at(text)
    return len(said - known), sum(not (o & said) for o in offered)


def leaks(text: str, private: Iterable[str] = ()) -> list[str]:
    low = text.lower()
    return [t for t in (*PRIVATE_TERMS, *(p.lower() for p in private)) if t and t in low]


GROUNDED_PROMPT = """You explain a timetabling conflict to one person and offer them options.
Write 3-6 short claims. Each claim cites the IDs of the facts it rests on;
use only those facts, and copy days, times, rooms and names exactly as the
facts give them. Be contrastive: say why the request as asked cannot work,
then what can. Mention the options by their letter. Never mention reasons
for anyone's unavailability. Address the recipient as "you"."""

FREE_PROMPT = """You explain a timetabling conflict to one person and offer them options.
Write a short, friendly message (3-6 sentences)."""

# For a model trained on the readable facts (``readable_facts``). The deployed model keeps GROUNDED_PROMPT
# until one trained on this prompt replaces it: a model must see the prompt it was trained on.
READABLE_PROMPT = """You write a short message to one person about a clash in their timetable and the
options they have. Write 3-6 claims, each one plain sentence of at most 25 words that cites the IDs of
the facts it rests on. Use only those facts, and copy days, times, rooms and names as they give them.
First say what cannot work as asked and why, in one or two claims. Then give each option in a claim of
its own that starts with "Option A:" (B, C, ...), saying where the class would go and what else moves.
Write as a colleague would: address the person as "you", and write no IDs, codes, brackets or tier
numbers. Never mention reasons for anyone's unavailability."""

MAX_CLAIM_WORDS = 30
_CODE = re.compile(r"\[[A-Z0-9-]+\]|\(Tier \d|\bTier \d\b|\b(?=[A-Z0-9-]*\d)[A-Z][A-Z0-9]*-[A-Z0-9-]+\b")


_INLINE_CODE = re.compile(r"\s*\[[A-Z][A-Z0-9-]*\]|\s*\((?=[A-Z0-9-]*\d)[A-Z][A-Z0-9]*-[A-Z0-9-]+\)"
                          r"|\s*\(Tier \d[^)]*\)")


def declutter(text: str) -> str:
    """A model's sentence without the IDs it copied from the facts: "the Algorithms practical (C3-P)", "the
    common lunch break [P-LUNCH] rule", "(Tier 3, verified unavailability)"."""
    return _INLINE_CODE.sub("", text)


_KEEPS = re.compile(r"\b(keeps?|stays?|remains?)\b", re.I)


def wrong_keep(text: str, own_days: set[str], checker: ClaimChecker) -> bool:
    """Does the claim say an option keeps a class on a day the recipient did not ask for ("Option A keeps
    your practical on Monday" when it was asked for Thursday)? Raters found this one word made them doubt
    the whole message. Unknown when the recipient asked for no day (``own_days`` empty)."""
    if not own_days or not _KEEPS.search(text):
        return False
    days, _, _ = checker.anchors(text)
    return bool(days) and not days <= own_days


def readable_facts(facts: Iterable[Fact]) -> list[Fact]:
    """The facts as the reader would put them (``Fact.reader``); facts another fact's sentence already
    says are left out. A readable model is given these, and its claims are checked against them."""
    return [Fact(f.id, f.reader) for f in facts if f.reader]


def unreadable(texts: Iterable[str]) -> str | None:
    """Why a message is hard to read (IDs, tier codes, copied fact markers, long or repeated sentences),
    or None."""
    seen: set[str] = set()
    for t in texts:
        if m := _CODE.search(t):
            return f"code {m[0]!r}"
        if len(t.split()) > MAX_CLAIM_WORDS:
            return "long sentence"
        key = re.sub(r"\W+", " ", t.lower()).strip()
        if key in seen:
            return "repeated sentence"
        seen.add(key)
    return None


# a full stop after a title or "e.g." does not end a sentence ("Hi Dr. Kumar, ...")
_SENTENCE_END = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
                           r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class Explainer:
    def __init__(self, instance: Instance, *, mode: str = "template", client=None, readable: bool = False) -> None:
        """``readable``: grounded explanations from the readable facts with READABLE_PROMPT (for a model
        trained on them); otherwise from the facts' model text with GROUNDED_PROMPT."""
        if mode not in ("template", "grounded", "free"):
            raise ValueError(mode)
        if mode != "template" and client is None:
            raise ValueError(f"{mode} explanations need an LLM client")
        self.instance = instance
        self.mode = mode
        self.client = client
        self.readable = readable
        self.prompt = READABLE_PROMPT if readable else GROUNDED_PROMPT
        self.checker = ClaimChecker(instance)

    def template(self, facts: list[Fact], recipient: str | None = None) -> Explanation:
        """What clashes, then the options, in the reader's words (``Fact.reader``), one claim per fact."""
        out = Explanation(mode="template")
        conflict = [f for f in facts if not f.id.startswith(("OPT-", "LED-"))]
        ledger = [f for f in facts if f.id.startswith("LED-")]
        options = [f for f in facts if f.id.startswith("OPT-")]
        why: list[str] = []
        if conflict:
            asked = [f for f in conflict if f.reader.startswith("You ")]
            lead = ("Your request can't be met exactly as asked." if asked else
                    "A change to the timetable needs one of your classes to move.")
            out.claims.append((lead, [], True))
            why.append(lead)
            # the reader's own request first, then what it clashes with, then rules and rooms
            for f in sorted(conflict, key=lambda f: (f.id.startswith(("RULE-", "RES-")), f not in asked)):
                if f.reader:
                    out.claims.append((f.reader, [f.id], True))
                    why.append(f.reader)
            if len([f for f in conflict if not f.id.startswith(("RULE-", "RES-"))]) == 1:
                rest = "It does not fit with the rest of the timetable as it stands."
                out.claims.append((rest, [], True))
                why.append(rest)
        for f in ledger:
            out.claims.append((f.reader, [f.id], True))
            why.append(f.reader)
        paragraphs = [" ".join(why)] if why else []
        out.lead = paragraphs[0] if paragraphs else ""
        if options:
            head = ("This option has been checked and works:" if len(options) == 1 else
                    "Each of these options has been checked and works:")
            out.claims.append((head, [], True))
            lines = [head]
            for f in options:
                line = f"{f.id.removeprefix('OPT-')}: {f.reader}."
                out.claims.append((line, [f.id], True))
                lines.append(line)
            paragraphs.append("\n".join(lines))
        out.body = "\n\n".join(paragraphs)
        return out

    def explain(self, facts: list[Fact], recipient: str | None = None, *,
                raw_requests: Iterable[str] = (), private: Iterable[str] = ()) -> Explanation:
        if self.mode == "template":
            out = self.template(facts, recipient)
        else:
            try:
                out = (self._grounded(facts, recipient) if self.mode == "grounded"
                       else self._free(facts, recipient, raw_requests))
            except LLMError:  # no usable output: send the template, and say so in the mode
                out = self.template(facts, recipient)
                out.mode = f"{self.mode}-failed"
        out.leaks = leaks(out.text, private)
        return out

    def _recipient(self, recipient: str | None) -> str:
        fac = self.instance.faculty_by_id.get(recipient or "")
        return fac.name if fac else (recipient or "the stakeholder")

    def shown(self, facts: list[Fact]) -> list[Fact]:
        """The facts the model is given and checked against."""
        return readable_facts(facts) if self.readable else list(facts)

    def grounded_prompt(self, facts: list[Fact], recipient: str | None) -> str:
        listing = "\n".join(f"[{f.id}] {f.text}" for f in self.shown(facts))
        return f"Recipient: {self._recipient(recipient)}\n\n# Facts\n{listing}"

    def _grounded(self, facts: list[Fact], recipient: str | None) -> Explanation:
        by_id = {f.id: f for f in self.shown(facts)}
        got = self.client.generate(self.prompt, self.grounded_prompt(facts, recipient), ExplanationOut)
        out = Explanation(mode="grounded")
        name = self._recipient(recipient)
        own_days: set[str] = set()  # the days the recipient's own constraints name
        for f in facts:
            if not f.id.startswith(("OPT-", "LED-", "RES-", "RULE-")) and f.text.startswith(f"{name} "):
                own_days |= self.checker.anchors(f.text)[0]
        for c in got.claims:
            c.facts = [i.strip().strip("[]") for i in c.facts]  # small models copy the listing's brackets
            cited = [by_id[i] for i in c.facts if i in by_id]
            ok = (len(cited) == len(c.facts) and self.checker.supported(c.text, cited)
                  and not wrong_keep(c.text, own_days, self.checker))
            out.claims.append((declutter(c.text), c.facts, ok))
            out.rejected += not ok
        if not any(ok for _, _, ok in out.claims) or not any(
                ok and any(i.startswith("OPT-") for i in ids) for _, ids, ok in out.claims) and any(
                f.id.startswith("OPT-") for f in facts):
            fallback = self.template(facts, recipient)  # nothing usable, or the options got lost
            fallback.rejected = out.rejected
            fallback.mode = "grounded-fallback"
            return fallback
        if self.readable:  # the explanation, then the options a line each
            kept = [(t, ids) for t, ids, ok in out.claims if ok]
            why = " ".join(t for t, ids in kept if not any(i.startswith("OPT-") for i in ids))
            options = "\n".join(t for t, ids in kept if any(i.startswith("OPT-") for i in ids))
            out.lead = why
            out.body = "\n\n".join(x for x in (why, options) if x)
        return out

    def _free(self, facts: list[Fact], recipient: str | None, raw_requests: Iterable[str]) -> Explanation:
        options = "\n".join(f.text for f in facts if f.id.startswith("OPT-"))
        requests = "\n".join(f"- {r}" for r in raw_requests)
        prompt = (f"Recipient: {self._recipient(recipient)}\n\nRequests that clash:\n{requests}\n\n"
                  f"The timetable cannot satisfy all of them. Options:\n{options}")
        got = self.client.generate(FREE_PROMPT, prompt, FreeExplanationOut)
        out = Explanation(mode="free", body=got.text)
        for s in sentences(got.text):
            t = self.checker.traced(s, facts)
            if t is not None:
                out.claims.append((s, [], t))
        return out
