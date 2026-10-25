"""Episodes behind the API: each runs in a thread, its spans become events a client can follow.

    svc = Service(store, model, tracer)
    ep = svc.start("Please move my class to Friday", actor="F-103", week=5)   # returns at once
    svc.wait(ep.id)                     # (tests) until it ends or waits for someone
    svc.reply(ep.id, Reply(text="My Monday lecture."))                       # an answer: it goes on
    svc.approve(ep.id)                  # commit a submitted draft through the kernel

Every span the harness writes for an episode is turned into an event (``events``): what the model
said and called, each tool's result, each ask and its reply, and how the episode ended or what it
waits for. A client follows them with ``after`` (the number of events it has seen).

With ``simulate`` the agent's asks are answered at once by simple stand-ins (the requester does not
answer; other teachers say no), so a demo never waits; otherwise an ask waits for a person: the
requester, or the teacher it is addressed to, answers it through ``reply``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nts.harness import AgentConfig, Harness, Outcome, approve, reject
from nts.models import Model
from nts.people import Cast
from nts.state import Store
from nts.telemetry import Span, Tracer, new_id
from nts.tools import ALL_TOOLS, Ask, Registry, Reply
from nts.tools import words as wording
from nts.tools.draft import OFFICE
from nts.tools.talk import REQUESTER


class NotFound(KeyError):
    pass


class Conflict(RuntimeError):
    pass


@dataclass
class Episode:
    id: str
    request: str
    actor: str
    actor_name: str
    week: int | None
    started: datetime
    status: str = "running"  # running, waiting, a final status, or after approval: approved / rejected / not_committed
    text: str = ""
    detail: str = ""
    events: list[dict] = field(default_factory=list)
    outcome: Outcome | None = None
    decision: dict | None = None  # the approval or rejection
    busy: bool = True  # a thread is running the harness for it
    cond: threading.Condition = field(default_factory=threading.Condition)

    def push(self, event: dict) -> None:
        with self.cond:
            event["n"] = len(self.events)
            self.events.append(event)
            self.cond.notify_all()

    @property
    def pending(self) -> Ask | None:
        return self.outcome.pending if self.outcome and self.outcome.status == "waiting" else None

    def summary(self) -> dict:
        p = self.pending
        return {"id": self.id, "request": self.request, "actor": self.actor_name, "week": self.week,
                "started": self.started.isoformat(timespec="seconds"), "status": self.status, "text": self.text,
                "detail": self.detail,
                "pending": None if p is None else {"kind": p.kind, "to": self.actor_name if p.to == REQUESTER else p.to_name,
                                                   "to_requester": p.to == REQUESTER, "text": p.text,
                                                   "rule": p.rule_text, "weeks": p.weeks}}


class _Events:
    """A sink that turns an episode's spans into events, as the harness writes them."""

    def __init__(self, service: Service) -> None:
        self.service = service

    def write(self, span: Span) -> None:
        ep = self.service.episodes.get(span.trace_id)
        if ep is not None and (event := to_event(span)) is not None:
            ep.push(event)


def to_event(span: Span) -> dict | None:
    a = span.attrs
    if span.kind == "model_call" and "response" in a:
        r = a["response"]
        return {"type": "model", "text": r.get("text", ""),
                "calls": [{"name": c["name"], "arguments": c.get("arguments", {})} for c in r.get("tool_calls", [])],
                "ms": span.ms}
    if span.kind == "model_call":
        return {"type": "model_error", "error": a.get("error", {})}
    if span.kind == "tool_call":
        r = a.get("result", {})
        ask = isinstance(r.get("content"), dict) and "kind" in r["content"] and "to_name" in r["content"]
        if ask:
            return None  # the reply event shows the ask with its answer
        return {"type": "tool", "name": span.name, "arguments": a.get("call", {}).get("arguments", {}),
                "ok": r.get("ok", False), "result": r.get("content") if r.get("ok") else r.get("error"), "ms": span.ms}
    if span.kind == "reply":
        ask, reply = a.get("ask", {}), a.get("reply", {})
        return {"type": "reply", "kind": ask.get("kind"), "to": ask.get("to_name"), "text": ask.get("text"),
                "rule": ask.get("rule_text"), "weeks": ask.get("weeks", []),
                "answer": reply.get("answer"), "says": reply.get("text", "")}
    if span.kind in ("episode", "resume"):
        status = a.get("status", "")
        return {"type": "waiting" if status == "waiting" else "end", "status": status, "text": a.get("text", ""),
                "detail": a.get("detail", ""), "pending": a.get("pending"), "turns": a.get("turns"),
                "tool_calls": a.get("tool_calls"), "refused": a.get("refused")}
    return None


class Service:
    def __init__(self, store: Store, model: Model, tracer: Tracer | None = None, *, agent: AgentConfig | None = None,
                 simulate: bool = False, handbook=None) -> None:
        from nts.agents import COORDINATOR

        self.store = store
        self.agent = agent or COORDINATOR
        self.simulate = simulate
        self.episodes: dict[str, Episode] = {}
        self.tracer = Tracer([*(tracer.sinks if tracer else []), _Events(self)])
        self.handbook = handbook
        self.harness = Harness(Registry(ALL_TOOLS), model, self.tracer, handbook=handbook)
        self.lock = threading.Lock()  # one commit at a time; drafts are private and need none

    # -- episodes -----------------------------------------------------------------

    def start(self, request: str, actor: str, week: int | None = None, *, simulate: bool | None = None) -> Episode:
        inst = self.store.world.instance
        name = inst.faculty_by_id[actor].name if actor in inst.faculty_by_id else "the timetable office"
        ep = Episode(new_id("ep-"), request, actor, name, week, datetime.now())
        self.episodes[ep.id] = ep
        people = Cast() if (self.simulate if simulate is None else simulate) else None
        self._spawn(ep, lambda: self.harness.run(self.agent, request, self.store, actor, week, episode_id=ep.id,
                                                 people=people))
        return ep

    def reply(self, eid: str, reply: Reply) -> Episode:
        ep = self.get(eid)
        with ep.cond:
            if ep.busy or ep.pending is None:
                raise Conflict(f"episode {eid} is not waiting for an answer ({ep.status})")
            ep.busy = True
            outcome = ep.outcome
        assert outcome is not None
        self._spawn(ep, lambda: self.harness.resume(outcome, reply))
        return ep

    def _spawn(self, ep: Episode, work) -> None:
        def go() -> None:
            try:
                out = work()
            except Exception as e:  # noqa: BLE001 -- shown on the episode, the server goes on
                with ep.cond:
                    ep.status, ep.detail, ep.busy = "model_error", f"{type(e).__name__}: {e}", False
                ep.push({"type": "end", "status": "model_error", "text": "", "detail": ep.detail})
                return
            with ep.cond:
                ep.outcome, ep.status, ep.text, ep.detail, ep.busy = out, out.status, out.text, out.detail, False
                ep.cond.notify_all()

        ep.busy = True
        threading.Thread(target=go, name=f"episode-{ep.id}", daemon=True).start()

    def wait(self, eid: str, timeout: float = 60.0) -> Episode:
        """Until the episode's thread is done (it ended or waits for someone)."""
        ep = self.get(eid)
        with ep.cond:
            if not ep.cond.wait_for(lambda: not ep.busy, timeout):
                raise TimeoutError(eid)
        return ep

    def events(self, eid: str, after: int = 0, timeout: float = 15.0) -> tuple[list[dict], bool]:
        """Events past ``after``, waiting up to ``timeout`` for one; and whether more may come without a reply."""
        ep = self.get(eid)
        with ep.cond:
            ep.cond.wait_for(lambda: len(ep.events) > after or not ep.busy, timeout)
            return ep.events[after:], ep.busy

    def get(self, eid: str) -> Episode:
        try:
            return self.episodes[eid]
        except KeyError:
            raise NotFound(eid) from None

    def inbox(self, person: str) -> list[dict]:
        """Episodes waiting for ``person``: their own requests' questions, and offers addressed to them."""
        out = []
        for ep in self.episodes.values():
            p = ep.pending
            if p is not None and not ep.busy and ((p.to == REQUESTER and ep.actor == person) or p.to == person):
                out.append(ep.summary())
        return out

    # -- the draft and its approval ------------------------------------------------

    def draft(self, eid: str) -> list[str]:
        ep = self.get(eid)
        if ep.outcome is None:
            return []
        d = ep.outcome.draft
        inst = d.world.instance
        base = d.base.constraints
        added = [wording.constraint(inst, c) for cid, c in d.world.constraints.items() if cid not in base]
        removed = [f"no longer: {wording.constraint(inst, c)}" for cid, c in base.items() if cid not in d.world.constraints]
        return added + removed

    def approve(self, eid: str, by: str = OFFICE) -> Episode:
        ep = self._decidable(eid)
        assert ep.outcome is not None
        with self.lock:
            world = approve(self.store, ep.outcome, self.tracer, by=by)
        ep.status = "approved" if world else "not_committed"
        ep.decision = {"by": by, "approved": world is not None, "version": world.version if world else None}
        ep.push({"type": "decision", **ep.decision})
        return ep

    def reject(self, eid: str, note: str = "", by: str = OFFICE) -> Episode:
        ep = self._decidable(eid)
        assert ep.outcome is not None
        reject(ep.outcome, self.tracer, by=by, note=note)
        ep.status = "rejected"
        ep.decision = {"by": by, "approved": False, "note": note}
        ep.push({"type": "decision", **ep.decision})
        return ep

    def _decidable(self, eid: str) -> Episode:
        ep = self.get(eid)
        if ep.busy or ep.outcome is None or ep.status != "submitted":
            raise Conflict(f"episode {eid} has no submitted draft to decide on ({ep.status})")
        if ep.outcome.draft.base.version != self.store.version:
            raise Conflict("the timetable changed since this draft was made; run the request again")
        return ep

    def detail(self, eid: str) -> dict[str, Any]:
        ep = self.get(eid)
        return {**ep.summary(), "draft": self.draft(eid), "decision": ep.decision, "busy": ep.busy,
                "stale": bool(ep.outcome and ep.status == "submitted"
                              and ep.outcome.draft.base.version != self.store.version)}
