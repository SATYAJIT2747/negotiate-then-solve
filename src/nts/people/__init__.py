"""Whoever answers the agent's talk tools: ``People.reply(ask, world) -> Reply``.

    Cast(requester=Requester("I mean my Data Structures lecture on Tuesday."),
         teachers={"F-104": Teacher(accept=True)})      # simulators, for evaluation and training traces
    Terminal()                                           # a person at the terminal answers (``nts ask --talk``)

Without people the harness pauses the episode at the first ask (status "waiting") and
``Harness.resume(outcome, reply)`` goes on with the answer, which is how the web app waits for a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from nts.state import World
from nts.tools import Ask, Reply
from nts.tools.talk import REQUESTER


class People(Protocol):
    def reply(self, ask: Ask, world: World) -> Reply: ...


@dataclass
class Requester:
    """The person who asked, who knows what they meant: one answer to every question (None: never answers)."""

    answer: str | None = None

    def reply(self, ask: Ask) -> Reply:
        return Reply(answer="no_reply") if self.answer is None else Reply(text=self.answer)


@dataclass
class Teacher:
    """Another teacher: says yes or no to every offer, with ``why`` as the reason given."""

    accept: bool = False
    why: str = ""
    responsive: bool = True

    def reply(self, ask: Ask) -> Reply:
        if not self.responsive:
            return Reply(answer="no_reply")
        if ask.kind == "notice":
            return Reply(text="Noted.")
        if self.accept:
            weeks = f"in week {', '.join(map(str, ask.weeks))}" if ask.weeks else "from now on"
            return Reply(answer="yes", text=f"Yes, that's fine {weeks}." + (f" {self.why}" if self.why else ""))
        return Reply(answer="no", text="No, sorry, I can't." + (f" {self.why}" if self.why else ""))


@dataclass
class Cast:
    """The simulated people of one scenario: the requester, and teachers by id (anyone else says no)."""

    requester: Requester = field(default_factory=Requester)
    teachers: dict[str, Teacher] = field(default_factory=dict)
    asked: list[tuple[Ask, Reply]] = field(default_factory=list)

    def reply(self, ask: Ask, world: World) -> Reply:
        who = self.requester if ask.to == REQUESTER else self.teachers.get(ask.to, Teacher())
        out = who.reply(ask)
        self.asked.append((ask, out))
        return out


class Terminal:
    """A person at the terminal plays whoever is asked."""

    def reply(self, ask: Ask, world: World) -> Reply:
        if ask.kind == "notice":
            print(f"  [message to {ask.to_name}] {ask.text}")
            return Reply(text="")
        print(f"  [{ask.kind} to {ask.to_name}] {ask.text}")
        if ask.kind == "offer":
            print(f"    rule: {ask.rule_text}; waive " + (f"in week {ask.weeks}" if ask.weeks else "for good"))
            yes = input("    agree? [y/n] ").strip().lower().startswith("y")
            return Reply(answer="yes" if yes else "no", text=input("    say: ").strip() or ("Yes." if yes else "No."))
        text = input("    answer (empty: no reply): ").strip()
        return Reply(text=text) if text else Reply(answer="no_reply")


__all__ = ["Cast", "People", "Requester", "Teacher", "Terminal"]
