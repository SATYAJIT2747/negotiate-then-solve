"""The ``rules`` tool: look up the department's regulations (``nts.policy.Handbook``) and cite them."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .registry import Context, Refusal, Scope, tool


class Rules(BaseModel):
    question: str = Field(description="What the rule is about: 'lunch hour', 'who may move a class', 'Saturday'")


@tool(Scope.POLICY)
def rules(ctx: Context, args: Rules) -> list[dict]:
    """Look up the department's regulations. Cite the rule ("rule": its number and id) when you rely on it."""
    if ctx.handbook is None:
        raise Refusal("no_handbook", "this department has no regulations loaded")
    found = ctx.handbook.search(args.question, k=3)
    if not found:
        raise Refusal("no_rule", "no rule matches; ask in other words, or say that no rule covers it")
    return [{"rule": r.cite(), "text": r.text} for r in found]


POLICY_TOOLS = [rules]
