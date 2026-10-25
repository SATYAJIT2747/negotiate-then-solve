"""What one model call is sent: the transcript, with old tool results trimmed to fit the context budget.

The transcript itself is never changed; ``fit`` returns a copy. Trimming is deterministic, so a
replayed episode is sent exactly what the recorded one was.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from nts.protocol import Message, ToolResult

LAST_TURN = ("This is your last turn: no more lookups or changes. Answer now with what you have, "
             "or end the request if you can.")
UNCHECKED = ("This is your last turn, and the draft changed after its last check, so it cannot be submitted. "
             "Decline or escalate, saying what you asked for, what you tried and what you found.")
WRAP_UP = ("One turn left after this one. Finish what you need now (if you changed the draft, call check_draft); "
           "next turn you answer or end the request.")
OMITTED = json.dumps({"omitted": "an older result, left out to save space; call the tool again if you need it"})


def tool_message(result: ToolResult) -> Message:
    body = result.content if result.ok else {"error": result.error.model_dump() if result.error else None}
    return Message(role="tool", call_id=result.call_id, name=result.name,
                   content=json.dumps(body, ensure_ascii=False, separators=(",", ":")))


def estimate(messages: Sequence[Message]) -> int:
    """Rough tokens (4 characters each, plus a few per message): enough to keep within a budget."""
    chars = sum(len(m.content) + sum(len(c.name) + len(json.dumps(c.arguments)) for c in m.tool_calls)
                for m in messages)
    return chars // 4 + 4 * len(messages)


def schema_tokens(tools: Sequence[dict]) -> int:
    """Rough tokens of the tool schemas (JSON runs nearer 3 characters a token than 4)."""
    return len(json.dumps(list(tools))) // 3 if tools else 0


def fit(messages: Sequence[Message], budget: int, keep_recent: int) -> list[Message]:
    """``messages`` with the oldest tool results replaced by a note until the estimate fits ``budget``.
    The system prompt, the request, every assistant turn and the latest ``keep_recent`` results stay."""
    out = list(messages)
    tools = [i for i, m in enumerate(out) if m.role == "tool"]
    for i in tools[: max(0, len(tools) - keep_recent)]:
        if estimate(out) <= budget:
            break
        out[i] = out[i].model_copy(update={"content": OMITTED})
    return out
