"""What one model call is sent: the transcript, with old tool results trimmed to fit the context budget.

The transcript itself is never changed; ``fit`` returns a copy. Trimming is deterministic, so a
replayed episode is sent exactly what the recorded one was.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from nts.protocol import Message, ToolResult

LAST_TURN = "This is your last turn: no more tools. Answer now with what you have, or say what you could not find."
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
