"""An agent is configuration, not code: a prompt, the tool scopes it may use, and its budgets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from nts.tools import Context, Scope


@dataclass(frozen=True)
class Budgets:
    max_turns: int = 8  # model calls per episode; the last one is offered no tools
    max_tool_calls: int = 16
    max_calls_per_turn: int = 4  # calls past this in one turn are refused, not run
    max_tokens: int | None = 24_000  # prompt + completion tokens over the episode (a cost guard); None: no limit
    max_seconds: float = 120.0
    max_repeats: int = 3  # refused repeats of an identical call before the episode counts as stuck
    context_tokens: int = 7_000  # one call's prompt, tool schemas included (estimated); old tool results are trimmed
    keep_recent: int = 4  # the latest tool results, never trimmed


@dataclass(frozen=True)
class AgentConfig:
    name: str
    prompt: str
    scopes: frozenset[Scope] = frozenset({Scope.READ})
    budgets: Budgets = field(default_factory=Budgets)

    def system(self, ctx: Context) -> str:
        inst = ctx.world.instance
        who = inst.faculty_by_id[ctx.actor].name if ctx.actor in inst.faculty_by_id else "the timetable office"
        week = f"week {ctx.week} of {inst.calendar.weeks}" if ctx.week else "outside term"
        return f"{self.prompt}\n\nThe person asking: {who}. It is {week}."

    def version(self, tools: Sequence[dict]) -> str:
        """A hash of everything that shapes what the model sees, so traces can be grouped by agent version."""
        body = json.dumps([self.prompt, sorted(s.value for s in self.scopes), asdict(self.budgets), list(tools)],
                          sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:12]
