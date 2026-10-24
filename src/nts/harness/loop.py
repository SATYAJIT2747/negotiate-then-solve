"""The agent loop: ask the model, run the tools it calls, feed back the results, until it answers.

    harness = Harness(Registry(READ_TOOLS), model, default_tracer())
    outcome = harness.run(READER, "when am I free on Tuesday?", store, actor="F-103", week=5)
    outcome.status, outcome.text     # "answered", "You are free on Tuesday at 9 am, ..."

Every episode ends with one status:
  answered          the model replied without calling a tool
  budget_exhausted  turns, tool calls, tokens or time ran out
  stuck             the model kept repeating calls it had already made
  model_error       the model call failed (server down, a replay that drifted, ...)
  submitted         an end tool succeeded: the draft waits for approval (``Store.commit``)
  declined          ... the model turned the request down, with a reason
  escalated         ... the model passed it to a person

The episode works in a draft of the store (``outcome.draft``): read tools see it, and nothing the
episode does reaches the store. Everything is recorded as spans: the episode, each turn, each model
call with the exact messages and reply, and each tool call with its result.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from nts.models import Model
from nts.protocol import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolError,
    ToolResult,
    Usage,
    request_hash,
    tools_hash,
)
from nts.state import Draft, Store
from nts.telemetry import Tracer, new_id
from nts.tools import Context, Registry, Scope

from .agent import AgentConfig
from .context import LAST_TURN, estimate, fit, schema_tokens, tool_message


@dataclass
class Outcome:
    episode_id: str
    status: str
    text: str
    turns: int
    tool_calls: int
    refused: int  # calls that came back as errors (bad names, bad arguments, repeats, ...)
    usage: Usage
    seconds: float
    transcript: list[Message]
    draft: Draft
    detail: str = ""  # why it stopped, when it did not answer
    calls: list[ToolResult] = field(default_factory=list)


class Harness:
    def __init__(self, registry: Registry, model: Model, tracer: Tracer | None = None) -> None:
        self.registry = registry
        self.model = model
        self.tracer = tracer or Tracer()

    def run(self, agent: AgentConfig, request: str, store: Store, actor: str, week: int | None = None,
            *, episode_id: str | None = None) -> Outcome:
        eid = episode_id or new_id("ep-")
        draft = store.draft(actor, eid)
        ctx = Context(store.world, actor, week, draft)
        tools = self.registry.specs(agent.scopes)
        transcript = [Message(role="system", content=agent.system(ctx)), Message(role="user", content=request)]
        run = _Run(eid, agent, ctx, transcript)

        with self.tracer.span("episode", agent.name, trace_id=eid, agent=agent.name, agent_version=agent.version(tools),
                              model=self.model.name, actor=actor, week=week, request=request,
                              state_version=store.version, tools=tools, budgets=asdict(agent.budgets)) as ep:
            self._loop(run, tools)
            outcome = run.outcome()
            ep.attrs.update(status=outcome.status, detail=outcome.detail, text=outcome.text, turns=outcome.turns,
                            tool_calls=outcome.tool_calls, refused=outcome.refused,
                            usage=outcome.usage.model_dump(), seconds=outcome.seconds,
                            draft_changes=len(draft.changes))
        return outcome

    def _loop(self, run: _Run, tools: list[dict]) -> None:
        b = run.agent.budgets
        for n in range(1, b.max_turns + 1):
            last = n == b.max_turns
            allowed = {Scope.END} & run.agent.scopes if last else run.agent.scopes  # the last turn can still end it
            offered = [t for t in tools if self.registry.tools[t["function"]["name"]].scope in allowed]
            sent = list(run.transcript) + ([Message(role="user", content=LAST_TURN)] if last else [])  # Qwen allows one system message, first
            sent = fit(sent, b.context_tokens - schema_tokens(offered), b.keep_recent)
            with self.tracer.span("turn", f"turn {n}", index=n, last=last,
                                  offered=[t["function"]["name"] for t in offered]) as span:
                run.turns = n
                reply = self._ask(run, sent, offered)
                if reply is None:
                    return
                run.transcript.append(reply.as_message())
                if not reply.tool_calls:
                    span.attrs["answered"] = True
                    run.stop("answered", text=reply.text)
                    return
                for i, call in enumerate(reply.tool_calls):
                    if run.status:  # an end tool already ended it: answer the rest so the transcript stays whole
                        result = _refuse(call, "episode_ended", "the request was already ended")
                    else:
                        result = self._call(run, call, allowed, extra=i >= b.max_calls_per_turn)
                        if result.ok and self.registry.tools[call.name].scope == Scope.END:
                            run.stop(result.content["status"], text=result.content["message"])
                    run.transcript.append(tool_message(result))
                if run.status:
                    return
                if last:
                    run.stop("budget_exhausted", "the model still called tools on its last turn")
                    return
                if reason := run.over_budget():
                    run.stop("budget_exhausted", reason)
                    return
                if run.repeats >= b.max_repeats:
                    run.stop("stuck", f"{run.repeats} repeated calls")
                    return
        run.stop("budget_exhausted", f"{b.max_turns} turns")

    def _ask(self, run: _Run, sent: list[Message], offered: list[dict]) -> AssistantTurn | None:
        try:
            with self.tracer.span("model_call", self.model.name, model=self.model.name,
                                  messages=[m.model_dump(mode="json") for m in sent],
                                  request_hash=request_hash(sent, offered), tools_hash=tools_hash(offered),
                                  estimated_tokens=estimate(sent)) as span:
                reply = self.model.chat(sent, offered)
                span.attrs["response"] = reply.model_dump(mode="json")
        except Exception as e:  # noqa: BLE001 -- any model failure ends the episode, recorded on its span
            run.stop("model_error", f"{type(e).__name__}: {e}")
            return None
        run.usage.prompt_tokens += reply.usage.prompt_tokens
        run.usage.completion_tokens += reply.usage.completion_tokens
        return reply

    def _call(self, run: _Run, call: ToolCall, allowed: set[Scope], *, extra: bool) -> ToolResult:
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str), run.ctx.state.version)
        with self.tracer.span("tool_call", call.name, call=call.model_dump(mode="json"),
                              state_version=run.ctx.state.version) as span:
            if extra:
                result = _refuse(call, "too_many_calls",
                                 f"at most {run.agent.budgets.max_calls_per_turn} calls in one turn; make this one next turn")
            elif key in run.seen:
                run.repeats += 1
                result = _refuse(call, "repeated_call", "you already made this exact call; its result is above")
            else:
                run.seen[key] += 1
                result = self.registry.run(call, run.ctx, allowed)
            span.attrs["result"] = result.model_dump(mode="json")
            span.attrs["ok"] = result.ok
        run.calls.append(result)
        return result


def _refuse(call: ToolCall, code: str, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=call.name, ok=False, error=ToolError(code=code, message=message))


class _Run:
    """The mutable state of one episode while it runs."""

    def __init__(self, eid: str, agent: AgentConfig, ctx: Context, transcript: list[Message]) -> None:
        self.eid, self.agent, self.ctx, self.transcript = eid, agent, ctx, transcript
        self.start = time.perf_counter()
        self.usage = Usage()
        self.turns = 0
        self.calls: list[ToolResult] = []
        self.seen: Counter = Counter()
        self.repeats = 0
        self.status, self.detail, self.text = "", "", ""

    def stop(self, status: str, detail: str = "", *, text: str = "") -> None:
        self.status, self.detail, self.text = status, detail, text

    def over_budget(self) -> str:
        b = self.agent.budgets
        if len(self.calls) >= b.max_tool_calls:
            return f"{b.max_tool_calls} tool calls"
        if b.max_tokens is not None and self.usage.prompt_tokens + self.usage.completion_tokens >= b.max_tokens:
            return f"{b.max_tokens} tokens"
        if time.perf_counter() - self.start >= b.max_seconds:
            return f"{b.max_seconds:g} s"
        return ""

    def outcome(self) -> Outcome:
        assert self.ctx.draft is not None
        return Outcome(episode_id=self.eid, status=self.status, text=self.text, turns=self.turns,
                       tool_calls=len(self.calls), refused=sum(not r.ok for r in self.calls), usage=self.usage,
                       seconds=round(time.perf_counter() - self.start, 3), transcript=self.transcript,
                       draft=self.ctx.draft, detail=self.detail, calls=self.calls)
