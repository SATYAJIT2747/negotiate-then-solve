"""What passes between the model, the harness and the tools.

These are also the records telemetry keeps and the training exporters read, so they hold everything
needed to rebuild a turn: the exact messages, the raw completion, the parsed calls and their results.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

UNPARSED = "__unparsed__"  # a call's only argument when the model's arguments were not valid JSON (the raw text)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    """Why a call failed, written for the model to fix it: ``hint`` holds e.g. ``did_you_mean``."""

    code: str
    message: str
    hint: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    content: Any = None
    error: ToolError | None = None
    ms: float = 0.0


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant messages
    call_id: str | None = None  # tool messages: the call they answer
    name: str | None = None  # tool messages: the tool
    meta: dict[str, Any] | None = None  # what one provider needs back on later turns (Gemini's thought signatures)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AssistantTurn(BaseModel):
    """One model reply: text, tool calls or both, with what it cost."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw: str | None = None  # the completion as the server sent it, before parsing
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    ms: float = 0.0
    meta: dict[str, Any] | None = None

    def as_message(self) -> Message:
        return Message(role="assistant", content=self.text, tool_calls=self.tool_calls, meta=self.meta)


def request_hash(messages: Sequence[Message], tools: Sequence[dict]) -> str:
    """One hash for everything a model call is asked with: the same hash means the same request."""
    body = json.dumps([[m.model_dump(mode="json") for m in messages], list(tools)], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def tools_hash(tools: Sequence[dict]) -> str:
    return hashlib.sha256(json.dumps(list(tools), sort_keys=True).encode()).hexdigest()[:16]
