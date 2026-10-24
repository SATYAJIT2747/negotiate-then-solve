"""End tools: how an episode that changes something finishes. The harness stops after the first that succeeds.

``submit`` puts the timetables ``check_draft`` found into the draft and hands it on for approval; it
refuses a draft changed since its last check, so what is approved is always what was checked. The
draft is not committed here: whoever approves calls ``Store.commit``, which checks it once more.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nts.state import TimetableSet

from .evaluate import Checked
from .registry import Context, Refusal, Scope, tool


class Submit(BaseModel):
    message: str = Field(description="What you tell the person: what changes, from when, who else it affects")


@tool(Scope.END)
def submit(ctx: Context, args: Submit) -> dict:
    """Send the draft for approval, with your message to the person. Only after check_draft said ok."""
    draft = ctx.draft
    checked: Checked | None = ctx.scratch.get("check")
    if draft is None or not draft.changes:
        raise Refusal("empty_draft", "there is nothing to submit; answer the person instead")
    if checked is None or checked.version != draft.world.version:
        raise Refusal("not_checked", "the draft changed since it was last checked (or was never); call check_draft")
    if not checked.ok:
        raise Refusal("check_failed", "the last check found the draft cannot be done; change it, decline or escalate")
    for s in checked.sets:
        draft.add(TimetableSet(week=s.week, assignment=s.assignment, cancelled=s.cancelled))
    return {"status": "submitted", "message": args.message}


class Decline(BaseModel):
    message: str = Field(description="What you tell the person: why it cannot be done, and what could be instead")


@tool(Scope.END)
def decline(ctx: Context, args: Decline) -> dict:
    """End without changing anything, telling the person why."""
    return {"status": "declined", "message": args.message}


class Escalate(BaseModel):
    message: str = Field(description="For the timetable office: what was asked, what clashes, what you tried")


@tool(Scope.END)
def escalate(ctx: Context, args: Escalate) -> dict:
    """Pass the request to a person in the timetable office: when it needs someone with more authority."""
    return {"status": "escalated", "message": args.message}


END_TOOLS = [submit, decline, escalate]
