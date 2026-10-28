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
  waiting           a talk tool asked someone, and no one was there to answer (``people=None``):
                    ``harness.resume(outcome, Reply(...))`` goes on once the answer comes

A talk tool's ask goes to ``people`` (simulators, a terminal), and the reply is what the model sees
as the tool's result. Calls the model made in the same turn after an ask are refused: they were made
before the answer.

The turn before the last is told one turn is left; the last is offered only the end tools, and if the
draft changed after its last check, not ``submit`` (it would be refused): the model declines or
escalates, saying what it found, instead of ending on a refusal.

The episode works in a draft of the store (``outcome.draft``): read tools see it, and nothing the
episode does reaches the store. Everything is recorded as spans: the episode, each turn, each model
call with the exact messages and reply, each tool call with its result, each reply to an ask, and a
"resume" span for the part of an episode after a pause.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

from nts.models import Model
from nts.people import People
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
from nts.screen import REFUSED, screen
from nts.state import Draft, Store, TimetableSet
from nts.telemetry import Tracer, new_id
from nts.tools import Ask, Context, Registry, Reply, Scope, settle

from .agent import AgentConfig
from .context import (
    LAST_TURN,
    UNCHECKED,
    WRAP_UP,
    estimate,
    fit,
    schema_tokens,
    tool_message,
)


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
    asks: list[tuple[Ask, Reply]] = field(default_factory=list)  # what talk tools asked, and the replies
    pending: Ask | None = None  # status "waiting": the ask that waits for an answer
    screened: bool = False  # an instruction to the system was left out of the request (``nts.screen``)
    run: _Run | None = field(default=None, repr=False)  # what ``Harness.resume`` goes on from


class Harness:
    def __init__(self, registry: Registry, model: Model, tracer: Tracer | None = None, *, handbook=None) -> None:
        self.registry = registry
        self.model = model
        self.tracer = tracer or Tracer()
        self.handbook = handbook  # the department's regulations, for the rules tool (nts.policy.Handbook)

    def run(self, agent: AgentConfig, request: str, store: Store, actor: str, week: int | None = None,
            *, episode_id: str | None = None, people: People | None = None) -> Outcome:
        eid = episode_id or new_id("ep-")
        draft = store.draft(actor, eid)
        screened = screen(request)  # instructions to the system are never read
        read = screened.text or ""
        ctx = Context(store.world, actor, week, draft, handbook=self.handbook, request=read, scopes=agent.scopes)
        tools = self.registry.specs(agent.scopes)
        transcript = [Message(role="system", content=agent.system(ctx)), Message(role="user", content=read)]
        run = _Run(eid, agent, ctx, transcript, tools, people)
        run.screened = screened.flagged

        with self.tracer.span("episode", agent.name, trace_id=eid, agent=agent.name, agent_version=agent.version(tools),
                              model=self.model.name, actor=actor, week=week, request=request,
                              state_version=store.version, tools=tools, budgets=asdict(agent.budgets),
                              screened=screened.flagged) as ep:
            if screened.text is None:
                run.stop("declined", "only instructions to the system", text=REFUSED)
            else:
                self._loop(run)
            outcome = run.outcome()
            ep.attrs.update(_ended(outcome))
        return outcome

    def resume(self, outcome: Outcome, reply: Reply, *, people: People | None = None) -> Outcome:
        """Go on with an episode that was waiting, ``reply`` being the answer to ``outcome.pending``."""
        run = outcome.run
        if run is None or run.waiting is None:
            raise ValueError(f"episode {outcome.episode_id} is not waiting ({outcome.status})")
        w, run.waiting, run.people = run.waiting, None, people
        run.restart_clock()
        with self.tracer.span("resume", run.agent.name, trace_id=run.eid) as span:
            result = self._settle(run, ToolResult(call_id=w.call.id, name=w.call.name, ok=True, content=w.ask),
                                  self._reply(run, w.ask, reply))
            run.transcript.append(tool_message(result))
            self._calls(run, w.rest, set(), w.last, after_ask=True)
            if not run.status and not run.waiting:
                self._loop(run)
            outcome = run.outcome()
            span.attrs.update(_ended(outcome))
        return outcome

    def _loop(self, run: _Run) -> None:
        b = run.agent.budgets
        while run.turns < b.max_turns:
            n = run.turns + 1
            last = n == b.max_turns
            allowed = {Scope.END} & run.agent.scopes if last else run.agent.scopes  # the last turn can still end it
            unchecked = last and run.unchecked()
            offered = [t for t in run.tools if self.registry.tools[t["function"]["name"]].scope in allowed
                       and not (unchecked and t["function"]["name"] == "submit")]  # it would be refused
            note = (UNCHECKED if unchecked else LAST_TURN) if last else WRAP_UP if n == b.max_turns - 1 > 1 else None
            sent = list(run.transcript) + ([Message(role="user", content=note)] if note else [])  # Qwen allows one system message, first
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
                self._calls(run, reply.tool_calls, allowed, last)
            if run.status or run.waiting:
                return
        run.stop("budget_exhausted", f"{b.max_turns} turns")

    def _calls(self, run: _Run, calls: list[ToolCall], allowed: set[Scope], last: bool, *,
               after_ask: bool = False) -> None:
        """Run one turn's calls in order, then the checks at the end of a turn. Stops early at an ask no
        one is there to answer: the calls after it wait in ``run.waiting``."""
        b = run.agent.budgets
        for i, call in enumerate(calls):
            if run.status:  # an end tool already ended it: answer the rest so the transcript stays whole
                result = _refuse(call, "episode_ended", "the request was already ended")
            elif after_ask:
                result = _refuse(call, "made_before_answer",
                                 "this call was made before the answer above came; make it again if you still need it")
                run.calls.append(result)
            else:
                result = self._call(run, call, allowed, extra=i >= b.max_calls_per_turn)
                if result.ok and isinstance(result.content, Ask):
                    ask = result.content
                    if ask.waits and run.people is None:
                        run.waiting = _Waiting(call, ask, calls[i + 1:], last)
                        return
                    result = self._settle(run, result, self._reply(run, ask))
                    after_ask = ask.waits
                elif result.ok and self.registry.tools[call.name].scope == Scope.END:
                    run.stop(result.content["status"], text=result.content["message"])
            run.transcript.append(tool_message(result))
        if run.status:
            return
        if last:
            run.stop("budget_exhausted", "the model still called tools on its last turn")
        elif reason := run.over_budget():
            run.stop("budget_exhausted", reason)
        elif run.repeats >= b.max_repeats:
            run.stop("stuck", f"{run.repeats} repeated calls")

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
                if (first := run.refused.get(key)) is None:
                    result = _refuse(call, "repeated_call", "you already made this exact call; its result is above")
                else:  # the same refused call again: say so, and how to get out of it
                    out = " or ask_requester" if Scope.TALK in allowed else " or decline"
                    result = _refuse(call, "repeated_call", f"this exact call was refused before ({first}); the same "
                                     f"arguments get the same answer: change them as the refusal says,{out}")
            else:
                run.seen[key] += 1
                result = self.registry.run(call, run.ctx, allowed)
                if not result.ok and result.error is not None:
                    run.refused[key] = result.error.message
            span.attrs["result"] = result.model_dump(mode="json")
            span.attrs["ok"] = result.ok
        run.calls.append(result)
        return result

    def _reply(self, run: _Run, ask: Ask, given: Reply | None = None) -> Reply:
        """The answer to ``ask``: ``given`` (a resumed episode), else from the people; recorded as a span."""
        with self.tracer.span("reply", ask.to_name, ask=ask.model_dump(mode="json")) as span:
            if given is not None:
                reply = given
            elif run.people is None:  # a notice, with no one to hand it to now: kept on the outcome
                reply = Reply()
            else:
                try:
                    reply = run.people.reply(ask, run.ctx.state)
                except Exception as e:  # noqa: BLE001 -- a broken simulator is a person who did not answer
                    span.attrs["error"] = f"{type(e).__name__}: {e}"
                    reply = Reply(answer="no_reply")
            span.attrs["reply"] = reply.model_dump(mode="json")
        return reply

    def _settle(self, run: _Run, result: ToolResult, reply: Reply) -> ToolResult:
        ask = result.content
        assert isinstance(ask, Ask)
        run.asks.append((ask, reply))
        try:
            content = settle(run.ctx, ask, reply)
        except ValueError as e:  # the waiver could not be written into the draft
            return _refuse(ToolCall(id=result.call_id, name=result.name), "bad_change", str(e))
        return result.model_copy(update={"content": content})


def _refuse(call: ToolCall, code: str, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=call.name, ok=False, error=ToolError(code=code, message=message))


def _ended(o: Outcome) -> dict:
    """What the episode (or resume) span records of how it ended."""
    return {"status": o.status, "detail": o.detail, "text": o.text, "turns": o.turns, "tool_calls": o.tool_calls,
            "refused": o.refused, "usage": o.usage.model_dump(), "seconds": o.seconds,
            "draft_changes": len(o.draft.changes), "asks": len(o.asks), "screened": o.screened,
            "pending": o.pending.model_dump(mode="json") if o.pending else None}


@dataclass
class _Waiting:
    call: ToolCall  # the talk call
    ask: Ask
    rest: list[ToolCall]  # the turn's calls after it, refused once the answer comes
    last: bool  # whether it was the last turn


class _Run:
    """The mutable state of one episode while it runs (and while it waits)."""

    def __init__(self, eid: str, agent: AgentConfig, ctx: Context, transcript: list[Message], tools: list[dict],
                 people: People | None) -> None:
        self.eid, self.agent, self.ctx, self.transcript = eid, agent, ctx, transcript
        self.tools, self.people = tools, people
        self.start = time.perf_counter()
        self.usage = Usage()
        self.turns = 0
        self.calls: list[ToolResult] = []
        self.asks: list[tuple[Ask, Reply]] = []
        self.waiting: _Waiting | None = None
        self.seen: Counter = Counter()
        self.refused: dict = {}  # call key -> why it was refused
        self.repeats = 0
        self.status, self.detail, self.text = "", "", ""
        self.elapsed = 0.0
        self.screened = False

    def stop(self, status: str, detail: str = "", *, text: str = "") -> None:
        self.status, self.detail, self.text = status, detail, text

    def restart_clock(self) -> None:
        """The time spent waiting for an answer does not count against the time budget."""
        self.start = time.perf_counter() - self.elapsed

    def unchecked(self) -> bool:
        """The draft has rules that its last check (if any) did not see: submit would refuse it."""
        draft = self.ctx.draft
        if draft is None or not any(not isinstance(c, TimetableSet) for c in draft.changes):
            return False
        checked = self.ctx.scratch.get("check")
        return checked is None or checked.version != draft.world.version

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
        self.elapsed = time.perf_counter() - self.start
        status, detail = self.status, self.detail
        if self.waiting:
            status, detail = "waiting", f"waiting for {self.waiting.ask.to_name}"
        return Outcome(episode_id=self.eid, status=status, text=self.text, turns=self.turns,
                       tool_calls=len(self.calls), refused=sum(not r.ok for r in self.calls), usage=self.usage,
                       seconds=round(self.elapsed, 3), transcript=self.transcript, draft=self.ctx.draft,
                       detail=detail, calls=self.calls, asks=list(self.asks),
                       pending=self.waiting.ask if self.waiting else None, screened=self.screened, run=self)
