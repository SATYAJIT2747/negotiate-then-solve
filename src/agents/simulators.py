"""Stakeholder simulators with hidden flexibility (proposal Sections 8.6, 12).

A simulated faculty member states a constraint as absolute, but privately
knows which other times would really work (``windows``), whether a room
without the requested equipment would do (``room_flexible``), whether they
volunteer an alternative when asked (``reveal``) and whether they answer at
all (``responsive``). The oracle is computed with all of this known.

Policy families (proposal Section 13.1) shape the hidden profile, and each
family's stakeholders respond stochastically, so no system can succeed by
exploiting a deterministic reply rule:

* ``strict``: one narrow alternative at most, never volunteered;
* ``flexible``: any time on the days that work, and volunteers one;
* ``cost_sensitive``: only small changes (the same time of day);
* ``history_sensitive``: refuses acceptable offers more often after having
  conceded recently, independent of the negotiator's ledger;
* ``unusual`` (hard tier only): only a time of day other than the one asked
  for works, on one day, and it is never volunteered: the "unusual slot" an
  option list that starts from the time asked for is least likely to reach.

With ``refuse_p`` an acceptable offer is refused anyway, and with
``silence_p`` a message goes unanswered; both draw from the simulator's own
seeded ``rng``. A profile without a family behaves deterministically.

The reply text is either a template (no API calls) or written by the
simulator LLM (Flash-Lite) from the scripted decision, in which case the
negotiator only sees text and must parse it (``structured=False``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from core.instance import Instance
from core.schemas import Placement
from language.corpus import FULL_DAY, hour

from .negotiation import Message, Reply


class Window(BaseModel):
    days: list[str] | None = None
    slots: list[int] | None = None

    def covers(self, day: str, slots: range) -> bool:
        return (self.days is None or day in self.days) and (self.slots is None or set(slots) <= set(self.slots))

    def text(self) -> str:
        days = " or ".join(FULL_DAY[d] for d in self.days) if self.days else "any day"
        if not self.slots:
            return days
        return f"{days} from {hour(min(self.slots))} to {hour(max(self.slots) + 1)}"


Family = Literal["strict", "flexible", "cost_sensitive", "history_sensitive", "unusual"]
FAMILIES: tuple[Family, ...] = ("strict", "flexible", "cost_sensitive", "history_sensitive")
HARD_FAMILIES: tuple[Family, ...] = (*FAMILIES, "unusual")
REFUSE_P = 0.1  # chance of refusing an acceptable offer anyway
SILENCE_P = 0.05  # chance of not answering a message
RELUCTANT_P = 0.5  # refusal chance for a history-sensitive stakeholder who conceded recently


class Profile(BaseModel):
    """Hidden truth about one stakeholder in a scenario."""

    owner: str
    windows: list[Window] = Field(default_factory=list)  # times that really work, beyond the stated ones
    needs: list[str] = Field(default_factory=list)  # equipment they really need
    room_flexible: bool = False  # would accept a room without that equipment
    reveal: bool = True  # volunteers a window when no offer fits
    # turns down the first message it gets whatever it offers (a demo's two-round negotiation); without
    # ``reveal`` it suggests nothing and then takes the next fitting option, whatever the time
    holds_out: bool = False
    responsive: bool = True
    family: Family | None = None
    refuse_p: float = 0.0
    silence_p: float = 0.0


def apply_family(profile: Profile, family: Family, *, stated_slots: list[int] | None, all_slots: list[int],
                 conceded_recently: bool) -> Profile:
    """Reshape a scenario's hidden profile to a policy family. Families only
    narrow or widen flexibility that exists, so a deadlock stays a deadlock."""
    p = profile.model_copy(deep=True)
    p.family, p.refuse_p, p.silence_p = family, REFUSE_P, SILENCE_P
    if family == "strict":
        p.windows = [Window(days=w.days[:1] if w.days else None, slots=w.slots) for w in p.windows[:1]]
        p.reveal = False
    elif family == "flexible":
        p.windows = [Window(days=w.days, slots=all_slots) for w in p.windows]
        p.reveal = True
    elif family == "cost_sensitive" and stated_slots:
        near = [Window(days=w.days, slots=[s for s in (w.slots or all_slots) if s in stated_slots])
                for w in p.windows]
        p.windows = [w for w in near if w.slots]
    elif family == "history_sensitive" and conceded_recently:
        p.refuse_p = RELUCTANT_P
    elif family == "unusual":
        other = [s for s in all_slots if s not in (stated_slots or [])]
        p.windows = [Window(days=w.days[:1] if w.days else None, slots=other) for w in p.windows[:1]]
        p.reveal = False
    return p


class _Voiced(BaseModel):
    text: str


VOICE_PROMPT = """You play a university faculty member answering a message from the timetable
office. Write a short, natural reply (1-2 sentences) that says exactly what you
are told to say and nothing else. Do not invent other conditions."""


@dataclass
class Simulator:
    instance: Instance
    profile: Profile
    client: object | None = None  # voices replies with the LLM when set
    structured: bool = True  # hand the negotiator the decision, not just text
    rng: random.Random | None = None  # draws refusals and silences; None means no noise
    revealed: bool = False
    held_out: bool = False  # has turned down a first message already (``Profile.holds_out``)
    log: list[Reply] = field(default_factory=list)

    def _chance(self, p: float) -> bool:
        return p > 0 and self.rng is not None and self.rng.random() < p

    def acceptable(self, sid: str, p: Placement) -> bool:
        session = next(s for s in self.instance.sessions if s.id == sid)
        span = range(p.slot, p.slot + session.duration)
        if not any(w.covers(p.day, span) for w in self.profile.windows):
            return False
        room = self.instance.room_by_id[p.room]
        needs = set(self.profile.needs) | set(session.equipment)
        return self.profile.room_flexible or needs <= set(room.equipment)

    def respond(self, message: Message) -> Reply:
        if not self.profile.responsive or self._chance(self.profile.silence_p):
            reply = Reply(decision="no_reply")
            self.log.append(reply)
            return reply
        fits = [o for o in message.offers if o.placements and all(
            self.acceptable(s, p) for s, p in o.placements.items())]
        if fits and self._chance(self.profile.refuse_p):
            fits = []
        if self.profile.holds_out and not self.held_out:
            self.held_out, fits = True, []
            if not (self.profile.reveal and self.profile.windows and not self.revealed):
                self.profile.windows = [Window()]  # said no, suggested nothing: takes what comes next
        if fits:
            reply = Reply(decision="accept", choice=fits[0].key)
            say = f"You accept option {fits[0].key}."
        elif self.profile.reveal and self.profile.windows and not self.revealed:
            w = self.profile.windows[0]
            self.revealed = True
            reply = Reply(decision="counter", counter_days=w.days, counter_slots=w.slots)
            say = f"None of the options work, but {w.text()} would work for you."
        else:
            reply = Reply(decision="reject")
            say = "None of the options work for you, and you have nothing else to suggest."
        reply.text = self._voice(reply, say, message)
        if not self.structured:
            reply = Reply(text=reply.text)
        self.log.append(reply)
        return reply

    def _voice(self, reply: Reply, say: str, message: Message) -> str:
        if self.client is None:
            if reply.decision == "accept":
                return f"Option {reply.choice} works for me, thanks."
            if reply.decision == "counter":
                return f"Sorry, those don't suit me, but {self.profile.windows[0].text()} would work."
            return "Sorry, none of those work for me."
        return voice(self.client, self.instance, self.profile.owner, message.text, say)


def voice(client, instance: Instance, owner: str, message_text: str, say: str) -> str:
    """The simulator LLM writes what ``owner`` would reply to a message."""
    me = instance.faculty_by_id.get(owner)
    prompt = (f"You are {me.name if me else owner}.\nThe message you received:\n"
              f"<message>\n{message_text}\n</message>\n\nWhat to say: {say}")
    return client.generate(VOICE_PROMPT, prompt, _Voiced, temperature=0.7).text
