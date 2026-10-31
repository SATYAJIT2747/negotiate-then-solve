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

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nts.bookings import Desk
from nts.harness import AgentConfig, Harness, Outcome, approve, reject
from nts.models import Model
from nts.notices import notices
from nts.people import Cast
from nts.semester import rebuild
from nts.state import ClassCancelled, ConstraintAdded, ConstraintRemoved, InvalidChange, StaleDraft, Store, TimetableSet
from nts.state.world import World
from nts.telemetry import Span, Tracer, new_id, read_spans
from nts.tools import ALL_TOOLS, Ask, Registry, Reply
from nts.tools import words as wording
from nts.tools.draft import OFFICE
from nts.tools.talk import REQUESTER

from .choices import choices


MEMORY = 4  # earlier messages of a conversation the agent sees
MEMORY_CHARS = 600  # of each earlier answer


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
    thread: str = ""  # the conversation it belongs to (the id of its first episode)
    busy: bool = True  # a thread is running the harness for it
    options: list[str] = field(default_factory=list)  # answers to pick for the requester's pending question
    notes: list[dict] = field(default_factory=list)  # the office's messages to the requester about it
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
        return {"id": self.id, "thread": self.thread, "request": self.request, "actor": self.actor_name, "week": self.week,
                "started": self.started.isoformat(timespec="seconds"), "status": self.status, "text": self.text,
                "detail": self.detail,
                "pending": None if p is None else {"kind": p.kind, "to": self.actor_name if p.to == REQUESTER else p.to_name,
                                                   "to_requester": p.to == REQUESTER, "text": p.text,
                                                   "rule": p.rule_text, "weeks": p.weeks, "options": self.options}}


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
    if span.kind == "model_call":  # a call asked again with a shorter prompt is no failure to show
        return None if a.get("retried") else {"type": "model_error", "error": a.get("error", {})}
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
    if span.kind == "note":
        return {"type": "note", "by": a.get("by", ""), "text": a.get("text", ""),
                "at": span.start.isoformat(timespec="seconds")}
    if span.kind in ("episode", "resume"):
        status = a.get("status", "")
        return {"type": "waiting" if status == "waiting" else "end", "status": status, "text": a.get("text", ""),
                "detail": a.get("detail", ""), "pending": a.get("pending"), "turns": a.get("turns"),
                "tool_calls": a.get("tool_calls"), "refused": a.get("refused")}
    return None


class Service:
    def __init__(self, store: Store, model: Model, tracer: Tracer | None = None, *, agent: AgentConfig | None = None,
                 simulate: bool = False, handbook=None, picker=None) -> None:
        from nts.agents import COORDINATOR

        self.store = store
        self.agent = agent or COORDINATOR
        self.simulate = simulate
        self.episodes: dict[str, Episode] = {}
        self.tracer = Tracer([*(tracer.sinks if tracer else []), _Events(self)])
        self.handbook = handbook
        self.harness = Harness(Registry(ALL_TOOLS), model, self.tracer, handbook=handbook, picker=picker)
        self.lock = threading.Lock()  # one commit at a time; drafts are private and need none
        self.desk = Desk(store)
        self.rebuild: dict = {"status": "none"}  # the office's semester rebuild: none, running, ready, failed
        self.told: dict[str, list[dict]] = {}  # person -> notices of committed changes to their classes

    # -- the semester timetable ---------------------------------------------------------

    def start_rebuild(self, from_week: int | None) -> dict:
        with self.lock:
            if self.rebuild["status"] == "running":
                raise Conflict("a rebuild is already running")
            self.rebuild = {"status": "running", "from_week": from_week, "started": datetime.now().isoformat("T", "seconds")}

        def go() -> None:
            try:
                r = rebuild(self.store, from_week=from_week)
                self.rebuild = {**self.rebuild, "status": "ready", "stats": r.stats, "unmet": r.unmet, "draft": r.draft}
            except Exception as e:  # noqa: BLE001 -- shown to the office
                self.rebuild = {**self.rebuild, "status": "failed", "error": f"{type(e).__name__}: {e}"}

        threading.Thread(target=go, name="rebuild", daemon=True).start()
        return self.rebuild_view()

    def rebuild_view(self) -> dict:
        out = {k: v for k, v in self.rebuild.items() if k != "draft"}
        d = self.rebuild.get("draft")
        if d is not None:
            out["stale"] = d.base.version != self.store.version
        return out

    def publish_rebuild(self) -> dict:
        d = self.rebuild.get("draft")
        if self.rebuild["status"] != "ready" or d is None:
            raise Conflict("there is no rebuilt timetable to publish")
        if d.base.version != self.store.version:
            raise Conflict("the timetable changed since the rebuild; rebuild again")
        with self.lock:
            world = self.store.commit(d)
        self.rebuild = {**self.rebuild, "status": "published", "version": world.version, "draft": None}
        return self.rebuild_view()

    # -- episodes -----------------------------------------------------------------

    def restore(self, spans: Path) -> int:
        """The episodes an earlier run of the server had on this store, rebuilt from their spans (``spans``, the
        telemetry's JSONL folder), so a restart keeps every request and what happened in it. They are shown, not
        run: one that was waiting or not yet approved ends as ``interrupted`` (its draft lived in memory)."""
        versions = {e.hash for e in self.store.events}
        traces: dict[str, list[Span]] = defaultdict(list)
        for span in read_spans(spans):  # in the order they ended, the order their events were pushed
            traces[span.trace_id].append(span)
        inst, n = self.store.world.instance, 0
        for eid, trace in traces.items():
            head = next((t for t in trace if t.kind == "episode"), None)
            if eid in self.episodes or head is None or head.attrs.get("state_version") not in versions:
                continue
            a = head.attrs
            name = inst.faculty_by_id[a["actor"]].name if a.get("actor") in inst.faculty_by_id else "the timetable office"
            ep = Episode(eid, a.get("request", ""), a.get("actor", ""), name, a.get("week"), head.start, busy=False,
                         thread=a.get("thread") or eid)
            for span in trace:
                if (event := to_event(span)) is not None:
                    ep.push(event)
                if span.kind in ("episode", "resume"):
                    ep.status, ep.text, ep.detail = (span.attrs.get(k, "") for k in ("status", "text", "detail"))
                elif span.kind == "note":
                    ep.notes.append(to_event(span))
                    undone = bool(span.attrs.get("undo"))
                    if undone:
                        ep.status, ep.decision = "undone", {**(ep.decision or {}), "undone": True}
                    self._tell(ep.actor, "undone" if undone else "office", span.attrs.get("text", ""), ep, span.start)
                elif span.kind == "approval":
                    b = span.attrs
                    ep.status = "approved" if b.get("committed") else "rejected" if "note" in b else "not_committed"
                    ep.decision = {"by": b.get("by"), "approved": bool(b.get("committed")), "version": b.get("version"),
                                   "note": b.get("note", "")}
                    ep.push({"type": "decision", **ep.decision})
                    self._decided(ep, b.get("notices") or {}, span.start)
            if ep.status in ("running", "waiting", "submitted"):
                ep.detail = f"{ep.status} when the server restarted; send the request again"
                ep.status = "interrupted"
            self.episodes[eid] = ep
            n += 1
        return n

    def start(self, request: str, actor: str, week: int | None = None, *, simulate: bool | None = None,
              thread: str | None = None) -> Episode:
        """A new request, or with ``thread`` the next message of that conversation: the agent sees the earlier
        messages and what came of them (``MEMORY`` of them). A message to a conversation whose agent waits for the
        requester's answer is that answer. A draft of the conversation still waiting for the office is replaced:
        the later message says what is wanted now."""
        earlier = self.thread(thread) if thread else []
        if earlier:
            last = earlier[-1]
            if last.busy:
                raise Conflict("the agent is still working on the last message")
            if (p := last.pending) is not None:
                if p.to != REQUESTER:
                    raise Conflict(f"the agent is waiting for {p.to_name} to answer first")
                return self.reply(last.id, Reply(answer="said", text=request))
            if last.status == "submitted" and last.decision is None:
                last.status, last.detail = "replaced", "a later message in the conversation replaced this draft"
                last.push({"type": "end", "status": "replaced", "text": "", "detail": last.detail})
        inst = self.store.world.instance
        name = inst.faculty_by_id[actor].name if actor in inst.faculty_by_id else "the timetable office"
        eid = new_id("ep-")
        ep = Episode(eid, request, actor, name, week, datetime.now(), thread=earlier[0].thread if earlier else eid)
        self.episodes[ep.id] = ep
        people = Cast() if (self.simulate if simulate is None else simulate) else None
        history = [(e.request, self.remembered(e)) for e in earlier[-MEMORY:]]
        self._spawn(ep, lambda: self.harness.run(self.agent, request, self.store, actor, week, episode_id=ep.id,
                                                 people=people, history=history, thread=ep.thread))
        return ep

    def _options(self, ep: Episode) -> list[str]:
        """Answers to pick for a question to the requester: what the question offers ("9 am or 11 am?",
        "(lecture or practical)", yes / no), or for "which class?" the requester's own classes."""
        p = ep.pending
        if p is None or p.to != REQUESTER:
            return []
        if picked := choices(p.text):
            return picked
        inst = self.store.world.instance
        if re.search(r"\bwhich (?:of your )?(?:class|course|lecture|lab|practical|session)", p.text, re.IGNORECASE):
            mine = sorted({f"{inst.course_title(s.course)} {s.kind.value}" for s in inst.sessions if s.faculty == ep.actor})
            return mine if 1 < len(mine) <= 8 else []
        return []

    def thread(self, tid: str) -> list[Episode]:
        """The episodes of a conversation, oldest first."""
        eps = sorted((e for e in self.episodes.values() if e.thread == tid), key=lambda e: e.started)
        if not eps:
            raise NotFound(tid)
        return eps

    def threads(self) -> list[list[Episode]]:
        """Every conversation, the most recently active first."""
        by: dict[str, list[Episode]] = defaultdict(list)
        for e in self.episodes.values():
            by[e.thread].append(e)
        convs = [sorted(eps, key=lambda e: e.started) for eps in by.values()]
        return sorted(convs, key=lambda eps: eps[-1].started, reverse=True)

    def remembered(self, ep: Episode) -> str:
        """What the agent remembers of an earlier message: its answer, and the change it led to, if any."""
        said = ep.text or ep.detail or "(no answer)"
        drafted = "; ".join(self.draft(ep.id)) if ep.outcome is not None else ""
        what = f"Drafted: {drafted}. " if drafted else ""
        fate = {  # whether the change is in the timetable is what a follow-up needs, said plainly
            "approved": "The office approved it: it is in the timetable now.",
            "submitted": "Sent to the office, not decided yet: not in the timetable.",
            "replaced": "Withdrawn by a later message: not in the timetable.",
            "rejected": "The office rejected it: not in the timetable.",
            "not_committed": "It could not be committed: not in the timetable.",
            "interrupted": "Never finished (the server restarted): nothing was changed.",
            "escalated": "Passed to the office: nothing was changed yet.",
            "undone": "Approved, then the office undid it: not in the timetable now.",
        }.get(ep.status, "" if ep.status == "answered" else "Nothing was changed.")
        office = " ".join(f"The office wrote: {n['text']}" for n in ep.notes)
        tail = " ".join(x for x in (what + fate, office) if x)
        return (said[:MEMORY_CHARS] + (f" [{tail}]" if tail else ""))[:MEMORY_CHARS + 300]

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
                ep.options = self._options(ep)
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

    def notices(self, person: str) -> list[dict]:
        """What committed changes did to ``person``'s classes, newest first."""
        return list(reversed(self.told.get(person, [])))

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

    def note(self, eid: str, text: str, by: str = OFFICE, *, undo: bool = False) -> Episode:
        """A message from the office to the requester about this request: shown in their chat (and recorded; the
        tracer's event sink adds it to the episode's events). ``undo``: the message that says the office undid it."""
        ep = self.get(eid)
        with self.tracer.span("note", by, trace_id=eid, by=by, text=text, undo=undo) as span:
            pass
        ep.notes.append(to_event(span))
        self._tell(ep.actor, "undone" if undo else "office", text, ep)
        return ep

    def needs_office(self, ep: Episode) -> bool:
        """A draft to decide on, or a request passed to the office that it has not answered yet."""
        if ep.busy:
            return False
        if ep.status == "submitted":
            return ep.outcome is not None and ep.outcome.draft.base.version == self.store.version
        return ep.status == "escalated" and not ep.notes

    def approve(self, eid: str, by: str = OFFICE, note: str = "") -> Episode:
        ep = self._decidable(eid)
        assert ep.outcome is not None
        with self.lock:
            world = approve(self.store, ep.outcome, self.tracer, by=by)
        affected = notices(ep.outcome.draft.base, world, ep.actor) if world is not None else {}
        ep.status = "approved" if world else "not_committed"
        ep.decision = {"by": by, "approved": world is not None, "version": world.version if world else None}
        ep.push({"type": "decision", **ep.decision})
        self._decided(ep, affected)
        if note:
            self.note(eid, note, by)
        return ep

    def undo(self, eid: str, note: str = "", by: str = OFFICE) -> Episode:
        """Take an approved change back out of the timetable: the rules it added are removed, its cancellations
        lifted, and the week timetables it repaired set back to what they were before it. Refused when a later change
        touched the same weeks (setting them back would undo that one too); then ask the agent for a new change."""
        ep = self.get(eid)
        if ep.status != "approved":
            raise Conflict(f"only an approved change can be undone ({ep.status})")
        events = list(self.store.events)
        mine = [i for i, e in enumerate(events) if e.episode == eid]
        if not mine:
            raise Conflict("nothing of this request is in the timetable")
        before = World.replay((e.change, e.hash) for e in events[:mine[0]])
        weeks = {e.change.week for e in (events[i] for i in mine) if isinstance(e.change, TimetableSet | ClassCancelled)}
        # a later change that was itself undone, with its undo, adds up to nothing: newest first, undo is allowed
        undone = {e.episode.removeprefix("undo:") for e in events if (e.episode or "").startswith("undo:")}
        later = {e.change.week for e in events[mine[-1] + 1:] if isinstance(e.change, TimetableSet | ClassCancelled)
                 and (e.episode or "").removeprefix("undo:") not in undone}
        if clash := weeks & later:
            raise Conflict(f"a later change also changed {', '.join('the semester' if w is None else f'week {w}' for w in clash)}; "
                           "ask the agent for a new change instead")
        # by what it changed, not step by step: a draft may add a rule, take it back and add it again
        draft, now = self.store.draft(by, f"undo:{eid}"), self.store.world
        changed = [events[i].change for i in mine]
        added = {ch.constraint.id for ch in changed if isinstance(ch, ConstraintAdded)}
        removed = {ch.id for ch in changed if isinstance(ch, ConstraintRemoved)}
        for cid in sorted(added - set(before.constraints)):
            if cid in now.constraints:
                draft.add(ConstraintRemoved(id=cid))
        for cid in sorted(removed & set(before.constraints)):
            if cid not in now.constraints:
                draft.add(ConstraintAdded(constraint=before.constraints[cid]))
        # (its cancellations are lifted by setting its weeks' timetables back, below)
        for w in sorted(weeks, key=lambda w: -1 if w is None else w):
            if (old := before.timetable(w)) is not None:
                draft.add(TimetableSet(week=w, assignment=dict(old.assignment), cancelled=sorted(old.cancelled)))
        with self.lock:
            try:
                self.store.commit(draft)
            except (InvalidChange, StaleDraft) as e:
                raise Conflict(f"it cannot be undone: {'; '.join(getattr(e, 'problems', [str(e)]))}") from None
        ep.status = "undone"
        ep.decision = {**(ep.decision or {}), "undone": True}
        self.note(eid, "The office undid this change: it is no longer in the timetable." + (f" {note}" if note else ""),
                  by, undo=True)
        return ep

    def reject(self, eid: str, note: str = "", by: str = OFFICE) -> Episode:
        ep = self._decidable(eid)
        assert ep.outcome is not None
        reject(ep.outcome, self.tracer, by=by, note=note)
        ep.status = "rejected"
        ep.decision = {"by": by, "approved": False, "note": note}
        ep.push({"type": "decision", **ep.decision})
        self._decided(ep)
        return ep

    # -- notifications: what each person is told, newest last ------------------------------

    def _tell(self, person: str, kind: str, text: str, ep: Episode, at: datetime | None = None) -> None:
        """``kind``: approved, rejected, not_committed (your request), office (a message about it), affected (someone
        else's approved change moved or cancelled one of your classes)."""
        self.told.setdefault(person, []).append({"at": (at or datetime.now()).isoformat(timespec="seconds"),
                                                 "kind": kind, "text": text, "episode": ep.id, "thread": ep.thread,
                                                 "request": ep.request, "mine": person == ep.actor})

    def _decided(self, ep: Episode, affected: dict[str, list[str]] | None = None, at: datetime | None = None) -> None:
        d = ep.decision or {}
        lines = ("; ".join(self.draft(ep.id)) if ep.outcome is not None else "") or ep.text  # after a restart: its summary
        if ep.status == "approved":
            self._tell(ep.actor, "approved", "The office approved it" + (f": {lines}" if lines else "."), ep, at)
        elif ep.status == "rejected":
            self._tell(ep.actor, "rejected", "The office did not approve it" + (f": {d['note']}" if d.get("note") else "."),
                       ep, at)
        elif ep.status == "not_committed":
            self._tell(ep.actor, "not_committed", "It could not be committed: the timetable changed. Send it again.", ep, at)
        for person, said in (affected or {}).items():
            for t in said:
                self._tell(person, "affected", t, ep, at)

    def _decidable(self, eid: str) -> Episode:
        ep = self.get(eid)
        if ep.busy or ep.outcome is None or ep.status != "submitted":
            raise Conflict(f"episode {eid} has no submitted draft to decide on ({ep.status})")
        if ep.outcome.draft.base.version != self.store.version:
            raise Conflict("the timetable changed since this draft was made; run the request again")
        return ep

    def detail(self, eid: str) -> dict[str, Any]:
        ep = self.get(eid)
        return {**ep.summary(), "draft": self.draft(eid), "decision": ep.decision, "busy": ep.busy, "notes": ep.notes,
                "needs_office": self.needs_office(ep),
                "screened": bool(ep.outcome and ep.outcome.screened),
                "stale": bool(ep.outcome and ep.status == "submitted"
                              and ep.outcome.draft.base.version != self.store.version)}
