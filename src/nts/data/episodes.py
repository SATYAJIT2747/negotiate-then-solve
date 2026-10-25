"""Episodes rebuilt from telemetry, with the labels that decide what they are worth for training.

    for ep in episodes("runs/telemetry/spans"):
        ep.label()          # {"good": True, "why": "scenario passed", "status": "submitted", ...}

An episode's worth comes from the strongest judgement on its trace:
  1. an evaluation (a scenario's checks on the committed state)   good = passed
  2. an approval (a person, or the eval runner, approving)         good = approved and committed
  3. nothing: an answered episode with no refused call             good = None (unknown; not exported by default)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from nts.protocol import AssistantTurn, Message
from nts.telemetry import Span, read_spans


@dataclass
class Episode:
    episode: Span
    model_calls: list[Span] = field(default_factory=list)
    tool_calls: list[Span] = field(default_factory=list)
    approvals: list[Span] = field(default_factory=list)
    evaluations: list[Span] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.episode.trace_id

    @property
    def model(self) -> str:
        return self.episode.attrs.get("model", "")

    @property
    def tools(self) -> list[dict]:
        return self.episode.attrs.get("tools", [])

    def label(self) -> dict:
        a = self.episode.attrs
        out = {"status": a.get("status"), "turns": a.get("turns"), "tool_calls": a.get("tool_calls"),
               "refused": a.get("refused"), "model": self.model, "agent_version": a.get("agent_version")}
        if self.evaluations:
            e = self.evaluations[-1].attrs
            return {**out, "good": bool(e.get("passed")), "why": "scenario " + ("passed" if e.get("passed") else
                    "failed: " + "; ".join(e.get("failures", []))), "family": e.get("family")}
        if self.approvals:
            p = self.approvals[-1].attrs
            ok = bool(p.get("approved")) and bool(p.get("committed"))
            return {**out, "good": ok, "why": "approved" if ok else f"not approved: {p.get('note') or p.get('reason', '')}"}
        return {**out, "good": None, "why": "not judged"}

    def turns(self) -> list[tuple[list[Message], AssistantTurn]]:
        """Each model call: the messages it was sent and the reply."""
        return [([Message.model_validate(m) for m in s.attrs["messages"]], AssistantTurn.model_validate(s.attrs["response"]))
                for s in self.model_calls if "response" in s.attrs]


def episodes(folder: Path | str) -> Iterator[Episode]:
    """Every finished episode in the JSONL files (ones without an episode span never finished)."""
    by: dict[str, list[Span]] = defaultdict(list)
    for s in read_spans(folder):
        by[s.trace_id].append(s)
    for spans in by.values():
        spans.sort(key=lambda s: (s.start, s.seq))
        top = next((s for s in spans if s.kind == "episode"), None)
        if top is None:
            continue
        for r in (s for s in spans if s.kind == "resume"):  # a paused episode ends where its last resume did
            top = top.model_copy(update={"attrs": {**top.attrs, **r.attrs}})
        ep = Episode(top)
        for s in spans:
            {"model_call": ep.model_calls, "tool_call": ep.tool_calls, "approval": ep.approvals,
             "evaluation": ep.evaluations}.get(s.kind, []).append(s)
        ep.model_calls.sort(key=lambda s: s.seq)
        ep.tool_calls.sort(key=lambda s: s.seq)
        yield ep
