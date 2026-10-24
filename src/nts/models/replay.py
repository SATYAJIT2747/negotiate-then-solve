"""A model that answers from recorded model calls: rerun an episode exactly, with no server.

    model = ReplayModel.from_spans(load_trace("runs/telemetry/spans", "ep-3f2a..."))
    harness.run(agent, request, store, actor, week, episode_id="ep-3f2a...")   # same turns, same calls

Each recorded call keeps the hash of what the model was asked (``request_hash``). In strict mode a
call must match the recorded one, so a replay that drifts (a tool now answers differently, a prompt
changed) stops with ``ReplayDiverged`` at the first turn that differs, instead of going on from answers to
questions nobody asked. ``strict=False`` serves the recorded turns in order whatever is asked.
"""

from __future__ import annotations

from collections.abc import Sequence

from nts.protocol import AssistantTurn, Message, request_hash
from nts.telemetry import Span


class ReplayDiverged(RuntimeError):
    pass


class ReplayExhausted(RuntimeError):
    pass


class ReplayModel:
    def __init__(self, turns: Sequence[tuple[str, AssistantTurn]], *, name: str = "replay", strict: bool = True) -> None:
        self.turns = list(turns)
        self.name = name
        self.strict = strict
        self.served = 0

    @classmethod
    def from_spans(cls, spans: Sequence[Span], *, strict: bool = True) -> ReplayModel:
        calls = sorted((s for s in spans if s.kind == "model_call" and "response" in s.attrs), key=lambda s: s.seq)
        if not calls:
            raise ValueError("no recorded model calls")
        turns = [(s.attrs["request_hash"], AssistantTurn.model_validate(s.attrs["response"])) for s in calls]
        return cls(turns, name=f"replay:{calls[0].name}", strict=strict)

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> AssistantTurn:
        if self.served == len(self.turns):
            raise ReplayExhausted(f"only {len(self.turns)} turns were recorded")
        expected, turn = self.turns[self.served]
        if self.strict and (got := request_hash(messages, tools)) != expected:
            raise ReplayDiverged(f"turn {self.served + 1} was asked differently (request {got}, recorded {expected})")
        self.served += 1
        return turn
