"""The one interface every model has: the conversation and the tools in, one assistant turn out."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nts.protocol import AssistantTurn, Message


class Model(Protocol):
    name: str  # what telemetry records as the model: server id plus adapter, or "scripted"

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> AssistantTurn: ...
