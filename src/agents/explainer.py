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
from core.schemas import Constraint, ConstraintType, Placement, Tier
from language.corpus import FIRST_SLOT_HOUR, FULL_DAY, hour
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
        parts.append(f"in week {w[0]}" if len(w) == 1 else f"in weeks {w[0]}-{w[-1]}")
    return " ".join(parts)


def session_name(instance: Instance, sid: str) -> str:
    s = next((x for x in instance.sessions if x.id == sid), None)
    if s is None:
        return sid
    return f"the {instance.course_title(s.course)} {s.kind.value} ({sid})"


def placement_text(instance: Instance, p: Placement) -> str:
    room = instance.room_by_id[p.room].name if p.room in instance.room_by_id else p.room
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
    else:
        body = f"No one may teach more than {c.limit} hours in a row"
    return body.strip()


def describe(instance: Instance, c: Constraint) -> str:
    """One constraint in plain words: who, what, when, tier. Never the reason."""
    return f"{_body(instance, c)} (Tier {int(c.tier)}, {TIER_NAME[c.tier]})."


# what a fixed constraint is, to someone who cannot change it; Tiers 3-5 need no note
FIXED_NOTE = {Tier.PHYSICAL: "this cannot be changed", Tier.POLICY: "an institute rule",
              Tier.COMMITMENT: "an agreed commitment"}


def plain_describe(instance: Instance, c: Constraint, you: str | None) -> str:
    note = FIXED_NOTE.get(c.tier)
    return f"{_body(instance, c, you or '')}{f' ({note})' if note else ''}."


EQUIPMENT_WORDS = {"gpu": "GPUs"}


def _equipment(names: Iterable[str]) -> str:
    return _and([EQUIPMENT_WORDS.get(e, e) for e in names])


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def plain_group(instance: Instance, cs: list[Constraint], you: str | None) -> str:
    """One sentence for one person's constraints on one class: "You need your ML practical on Tuesday in the
    afternoon, in a room with GPUs." Anything else gets a sentence per constraint."""
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


def plain_option(instance: Instance, o: OptionView, asked: Iterable[Constraint], you: str | None) -> str:
    """"Friday at 2 pm in Lab 3, as you asked; 2 other classes move"."""
    asked = list(asked)
    parts = []
    for sid, p in sorted(o.placements.items()):
        s = instance.session_by_id.get(sid)
        where = placement_text(instance, p)
        if len(o.placements) > 1 and s is not None:
            where = f"{'your' if s.faculty == you else 'the'} {instance.course_title(s.course)} {s.kind.value} on {where}"
        if any(_as_asked(c, sid, p) for c in asked):
            where += ", as you asked"
        parts.append(where)
    text = "; ".join(parts) or "the timetable stays as it is"
    if o.moved is not None:
        text += (" (nothing else moves)" if o.moved == 0 else
                 f" ({o.moved} other class{'es' if o.moved != 1 else ''} move{'s' if o.moved == 1 else ''})")
    return text


def conflict_facts(
    instance: Instance,
    constraints: Mapping[str, Constraint],
    mus: Iterable[str],
    options: Iterable[OptionView] = (),
    ledger_counts: Mapping[str, float] | None = None,
    recipient: str | None = None,
) -> list[Fact]:
    facts: list[Fact] = []
    involved = [constraints[i] for i in mus]
    groups: dict[tuple, list[Constraint]] = {}
    for c in involved:
        groups.setdefault((c.owner, tuple(c.scope.model_dump().values())), []).append(c)
    for c in involved:
        group = groups[(c.owner, tuple(c.scope.model_dump().values()))]
        plain = plain_group(instance, group, recipient) if c is group[0] else ""
        facts.append(Fact(c.id, describe(instance, c), plain))
        if c.source.rule:
            facts.append(Fact(f"RULE-{c.id}", f"This comes from the institute rule: {c.source.rule} [{c.id}].",
                              f"That is the institute's rule on {c.source.rule}."))
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
                     f"Only {_and(list(rooms))} {'has' if len(rooms) == 1 else 'have'} {_equipment(same)}.")
        facts.append(Fact(f"RES-{eq}", f"Rooms with {eq}: {', '.join(rooms) if rooms else 'none'}.", plain))
    for o in options:
        where = "; ".join(f"{session_name(instance, s)} on {placement_text(instance, p)}"
                          for s, p in sorted(o.placements.items()))
        facts.append(Fact(f"OPT-{o.key}", f"Option {o.key}: {where}. The solver confirms this works.",
                          plain_option(instance, o, (c for c in involved if c.owner == recipient), recipient)))
    owners = {c.owner for c in involved if c.owner in instance.faculty_by_id}
    for owner in sorted(owners):
        n = (ledger_counts or {}).get(owner, 0)
        if n:
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


# a full stop after a title or "e.g." does not end a sentence ("Hi Dr. Kumar, ...")
_SENTENCE_END = re.compile(r"(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bMrs\.)(?<!\bProf\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
                           r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


class Explainer:
    def __init__(self, instance: Instance, *, mode: str = "template", client=None) -> None:
        if mode not in ("template", "grounded", "free"):
            raise ValueError(mode)
        if mode != "template" and client is None:
            raise ValueError(f"{mode} explanations need an LLM client")
        self.instance = instance
        self.mode = mode
        self.client = client
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

    def grounded_prompt(self, facts: list[Fact], recipient: str | None) -> str:
        listing = "\n".join(f"[{f.id}] {f.text}" for f in facts)
        return f"Recipient: {self._recipient(recipient)}\n\n# Facts\n{listing}"

    def _grounded(self, facts: list[Fact], recipient: str | None) -> Explanation:
        by_id = {f.id: f for f in facts}
        got = self.client.generate(GROUNDED_PROMPT, self.grounded_prompt(facts, recipient), ExplanationOut)
        out = Explanation(mode="grounded")
        for c in got.claims:
            c.facts = [i.strip().strip("[]") for i in c.facts]  # small models copy the listing's brackets
            cited = [by_id[i] for i in c.facts if i in by_id]
            ok = len(cited) == len(c.facts) and self.checker.supported(c.text, cited)
            out.claims.append((c.text, c.facts, ok))
            out.rejected += not ok
        if not any(ok for _, _, ok in out.claims) or not any(
                ok and any(i.startswith("OPT-") for i in ids) for _, ids, ok in out.claims) and any(
                f.id.startswith("OPT-") for f in facts):
            fallback = self.template(facts, recipient)  # nothing usable, or the options got lost
            fallback.rejected = out.rejected
            fallback.mode = "grounded-fallback"
            return fallback
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
