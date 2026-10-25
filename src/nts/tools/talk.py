"""Talk tools: the agent asks someone something, and the episode waits for the answer.

A talk tool does not answer itself: it returns an ``Ask``, and the harness gets the ``Reply`` from
whoever plays the people (``nts.people``: simulators in evaluation, a person at a terminal or in the
web app), or pauses the episode until it arrives. ``settle`` then turns the reply into what the model
sees, and, for an accepted offer, writes the waiver into the draft:

    ask_requester  a question to the person who asked (which class? which week?)
    make_offer     ask the owner of a rule that blocks the draft to waive it for the weeks it blocks
    message_party  tell someone something; no answer is awaited

Only an owner's yes changes their rule, so a teacher's draft can touch another teacher's rule only
through ``make_offer``; the office's rules are not offered (the agent escalates those).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nts.state import ConstraintAdded, ConstraintRemoved

from . import words
from .draft import _draft
from .names import Directory
from .registry import Context, Refusal, Scope, tool

REQUESTER = "requester"


class Ask(BaseModel):
    """What a talk tool asks, of whom: the harness hands it to the people and waits for a Reply."""

    kind: Literal["question", "offer", "notice"]
    to: str  # a faculty id, or REQUESTER
    to_name: str
    text: str
    rule: str | None = None  # an offer: the id of the rule asked to be waived
    rule_text: str | None = None
    weeks: list[int] = Field(default_factory=list)  # an offer: the weeks it would be waived in; empty: dropped

    @property
    def waits(self) -> bool:
        return self.kind != "notice"


class Reply(BaseModel):
    answer: Literal["yes", "no", "no_reply", "said"] = "said"  # an offer is answered yes or no; a question, "said"
    text: str = ""


def theirs(ctx: Context) -> dict[int, tuple[str, list[int]]]:
    """Rules of other people that a check found blocking: number -> (rule id, the weeks it blocks; [] the term)."""
    return ctx.scratch.setdefault("theirs", {})


def number_theirs(ctx: Context, cid: str, weeks: list[int]) -> int:
    """The number a blocking rule is shown with (the same number each time it is found)."""
    rules = theirs(ctx)
    for n, (c, _) in rules.items():
        if c == cid:
            rules[n] = (cid, weeks)
            return n
    n = max(rules, default=0) + 1
    rules[n] = (cid, weeks)
    return n


class AskRequester(BaseModel):
    question: str = Field(description="One short question")


@tool(Scope.TALK)
def ask_requester(ctx: Context, args: AskRequester) -> Ask:
    """Ask the person who made the request a question, and wait for the answer: when the request is unclear,
    or to ask whether another time would do."""
    inst = ctx.state.instance
    name = inst.faculty_by_id[ctx.actor].name if ctx.actor in inst.faculty_by_id else "the timetable office"
    return Ask(kind="question", to=REQUESTER, to_name=name, text=args.question)


class MakeOffer(BaseModel):
    to: str = Field(description="The person whose rule it is")
    rule: int = Field(description="The rule's number, from check_draft")
    message: str = Field(description="What you ask and why, in a sentence or two")


@tool(Scope.TALK)
def make_offer(ctx: Context, args: MakeOffer) -> Ask:
    """Ask the owner of a rule that blocks the draft (check_draft shows it with a number) to waive it for
    the weeks it blocks, and wait for the answer. If they agree, the waiver goes into the draft."""
    _draft(ctx)
    rules = theirs(ctx)
    if args.rule not in rules:
        raise Refusal("unknown_rule", f"no blocking rule {args.rule}; check_draft numbers the rules others set",
                      rules=sorted(rules))
    cid, weeks = rules[args.rule]
    world = ctx.state
    c = world.constraints.get(cid)
    if c is None:
        raise Refusal("already_waived", f"rule {args.rule} is already waived in the draft")
    inst = world.instance
    owner = inst.faculty_by_id[c.owner]
    who = Directory(inst).teacher(args.to, ctx.actor)
    if who.id != owner.id:
        raise Refusal("not_theirs", f"rule {args.rule} is {owner.name}'s, not {who.name}'s", owner=owner.name)
    if (cid, "no") in ctx.scratch.get("answers", set()):
        raise Refusal("already_refused", f"{owner.name} already said no to waiving it")
    return Ask(kind="offer", to=owner.id, to_name=owner.name, text=args.message, rule=cid,
               rule_text=words.constraint(inst, c), weeks=weeks)


class MessageParty(BaseModel):
    to: str = Field(description="A teacher's name")
    message: str


@tool(Scope.TALK)
def message_party(ctx: Context, args: MessageParty) -> Ask:
    """Tell a teacher something they need to know. No answer is awaited."""
    who = Directory(ctx.state.instance).teacher(args.to, ctx.actor)
    if who.id == ctx.actor:
        raise Refusal("bad_arguments", "that is the person asking; answer them instead")
    return Ask(kind="notice", to=who.id, to_name=who.name, text=args.message)


def settle(ctx: Context, ask: Ask, reply: Reply) -> dict:
    """What the model is shown of a reply; an accepted offer also writes the waiver into the draft."""
    if ask.kind == "notice":
        return {"sent_to": ask.to_name}
    if reply.answer == "no_reply":
        return {"from": ask.to_name, "answer": "no reply"}
    out = {"from": ask.to_name, "says": reply.text}
    if ask.kind != "offer":
        return out
    assert ask.rule is not None
    ctx.scratch.setdefault("answers", set()).add((ask.rule, "yes" if reply.answer == "yes" else "no"))
    out["answer"] = "yes" if reply.answer == "yes" else "no"
    if reply.answer == "yes":
        draft = _draft(ctx)
        c = draft.world.constraints[ask.rule]
        draft.add(ConstraintRemoved(id=c.id))
        if ask.weeks:
            valid = c.valid.model_copy(update={"except_weeks": sorted({*c.valid.except_weeks, *ask.weeks})})
            waived = c.model_copy(update={"id": f"{c.id}~{draft.episode or 'draft'}", "valid": valid,
                                          "source": c.source.model_copy(update={"request": f"waived: {ask.to_name} agreed"})})
            draft.add(ConstraintAdded(constraint=waived))
            out["draft"] = f"now: {words.constraint(draft.world.instance, waived)}"
        else:
            out["draft"] = f"dropped: {ask.rule_text}"
        ctx.scratch.setdefault("waivers", []).append(out["draft"])
        out["next"] = "call check_draft again"
    return out


def waived(cid: str) -> bool:
    """Whether a rule id is a waived copy (``settle``): looser than the rule it replaced, so it breaks no timetable."""
    return "~" in cid


TALK_TOOLS = [ask_requester, make_offer, message_party]
__all__ = ["REQUESTER", "TALK_TOOLS", "Ask", "Reply", "make_offer", "settle", "waived"]
