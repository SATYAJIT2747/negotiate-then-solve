"""The agent behind HTTP: sign in, start an episode, follow its turns as server-sent events, answer asks, approve.

    python -m nts serve --store runs/agentic/demo.jsonl --model llama --adapter <lora> --port 8100
    # then open http://127.0.0.1:8100/ (the page in nts/api/static), or call the API:

    POST /api/login                 {"dept": "demo", "name": "Dr. Iyer" | "office", "password": "..."} -> a session cookie
    POST /api/episodes              {"request": "...", "week": 5, "simulate": false}   (as the signed-in person)
    GET  /api/episodes/{id}/events  text/event-stream: model, tool, reply, waiting, end, decision
    GET  /api/episodes/{id}/poll?after=n   the same events as JSON (for clients without SSE)
    POST /api/episodes/{id}/reply   {"answer": "yes" | "no" | "said" | "no_reply" | "counter", "text": "...",
                                     "day": "Thu", "hour": "2 pm"}   (the one asked; day and hour for a counter)
    POST /api/episodes/{id}/approve | /reject {"note": "..."}                             (the office)
    GET  /api/inbox                 what waits for my answer
    GET  /api/notices               what approved changes did to my classes
    GET  /api/department, /api/timetable?week=5&teacher=Dr. Iyer, /api/fairness, /api/rules?q=lunch
    GET  /api/obs/overview, /api/obs/episodes, /api/obs/trace/{id}, /api/obs/evals          (the office)
    GET  /api/bookings?week=6; POST /api/bookings {...}; POST /api/bookings/{id}/approve | /refuse; DELETE ...
    POST /api/semester/rebuild {"from_week": 5}; GET /api/semester/rebuild; POST /api/semester/publish (the office)

Several departments can be served at once (one store each, ``departments``); a session belongs to one.
Who may do what is decided here, from the session (``auth``): a teacher sees their own requests and what
waits for them; the office sees all, starts requests for the office and decides on drafts.
"""

# no `from __future__ import annotations`: FastAPI must see the dependency types defined in create_app
import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from nts.bookings import Booking
from nts.fairness import standing
from nts.state import InvalidChange
from nts.tools import Refusal, Reply
from nts.tools.draft import OFFICE
from nts.tools.names import FULL_DAY, Directory, hour
from nts.tools.read import _at
from nts.tools.talk import REQUESTER

from .auth import CLUB, Session, Sessions, Users
from .obs import Observability
from .service import Conflict, Episode, NotFound, Service

STATIC = Path(__file__).parent / "static"
COOKIE = "nts_session"


class Login(BaseModel):
    dept: str
    name: str
    password: str = ""


FOR_OFFICE = frozenset({"submitted", "approved", "rejected", "not_committed", "escalated", "replaced"})


class Start(BaseModel):
    request: str
    week: int | None = None
    simulate: bool | None = None
    thread: str | None = None  # the conversation this message continues; none: a new one


class Answer(BaseModel):
    answer: Literal["yes", "no", "said", "no_reply", "counter"] = "said"
    text: str = ""
    day: str | None = None  # "counter": another time for the requester's class
    hour: str | None = None


class Note(BaseModel):
    note: str = ""


class BookingIn(BaseModel):
    club: str | None = None  # the office books for a club; a club books for itself
    purpose: str
    week: int
    day: str
    from_hour: str
    to_hour: str
    attendees: int = Field(ge=1)
    room: str | None = None


class RebuildIn(BaseModel):
    from_week: int | None = None


def create_app(departments: dict[str, Service] | Service, users: Users | None = None, *,
               telemetry: Path = Path("runs/telemetry"), evals: Path = Path("runs/eval"),
               secure_cookie: bool = False) -> FastAPI:
    depts = departments if isinstance(departments, dict) else {"demo": departments}
    sessions = Sessions(users or Users.open())
    obs = Observability(telemetry, evals)
    app = FastAPI(title="Negotiate, Then Solve: agent API")

    # -- who is calling ---------------------------------------------------------------

    def token_of(request: Request) -> str | None:
        """The page keeps its session per browser tab and sends it as a bearer token (an event stream, which sends no
        headers, as ``?token=``), so two tabs can be two people; other clients may use the cookie."""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:]
        return request.query_params.get("token") or request.cookies.get(COOKIE)

    def session(request: Request) -> Session:
        token = token_of(request)
        s = sessions.get(token)
        if s is None or s.dept not in depts:
            raise HTTPException(401, "sign in first")
        return s

    Me = Annotated[Session, Depends(session)]

    def office(s: Me) -> Session:
        if not s.office:
            raise HTTPException(403, "only the timetable office can do this")
        return s

    Office = Annotated[Session, Depends(office)]

    def service(s: Session) -> Service:
        return depts[s.dept]

    def episode(s: Session, eid: str) -> Episode:
        try:
            ep = service(s).get(eid)
        except NotFound:
            raise HTTPException(404, f"no episode {eid}") from None
        if not _sees(s, ep):
            raise HTTPException(404, f"no episode {eid}")  # not 403: do not say it exists
        return ep

    def _sees(s: Session, ep: Episode) -> bool:
        """Your own messages, the ones that ask you something, and for the office the ones that reach it: a change
        sent for approval (and what came of it) or a request passed on. A teacher's questions stay theirs."""
        return ep.actor == s.person or _asks(ep, s.person) or (s.office and ep.status in FOR_OFFICE)

    def _asks(ep: Episode, person: str) -> bool:
        """Whether ``person`` was asked something in ``ep`` (or is asked now)."""
        p = ep.pending
        if p is not None and p.to == person:
            return True
        return bool(ep.outcome and any(a.to == person for a, _ in ep.outcome.asks))

    @app.get("/api/departments")
    def departments_() -> dict:
        out = []
        for d, svc in depts.items():
            inst = svc.store.world.instance
            out.append({"id": d, "teachers": [f.name for f in inst.faculty], "faculty": len(inst.faculty)})
        return {"departments": out, "passwords": sessions.users.needs_password}

    @app.post("/api/login")
    def login(body: Login, response: Response) -> dict:
        if body.dept not in depts:
            raise HTTPException(404, f"no department {body.dept}")
        if body.name.lower() in ("office", OFFICE, "the timetable office"):
            person, name = OFFICE, "office"
        elif body.name.lower().startswith("club:") and body.name[5:].strip():
            name = body.name[5:].strip()
            person = f"{CLUB}{name}"
        else:
            try:
                f = Directory(depts[body.dept].store.world.instance).teacher(body.name)
            except Refusal:
                raise HTTPException(401, "wrong name or password") from None
            person, name = f.id, f.name
        s = sessions.start(body.dept, person, name, body.password)
        if s is None:
            raise HTTPException(401, "wrong name or password")
        response.set_cookie(COOKIE, s.token, httponly=True, samesite="strict", secure=secure_cookie,
                            max_age=12 * 3600)
        return {**s.view(), "token": s.token}

    @app.post("/api/logout")
    def logout(request: Request, response: Response) -> dict:
        sessions.end(token_of(request) or "")
        response.delete_cookie(COOKIE)
        return {}

    @app.get("/api/me")
    def me(s: Me) -> dict:
        return s.view()

    # -- the department ---------------------------------------------------------------

    @app.get("/api/department")
    def department(s: Me) -> dict:
        svc = service(s)
        w = svc.store.world
        inst = w.instance
        return {"id": s.dept, "version": w.version, "weeks": inst.calendar.weeks,
                "days": [FULL_DAY[d] for d in inst.calendar.days],
                "hours": [hour(q) for q in range(inst.calendar.slots_per_day + 1)],
                "lunch": hour(inst.calendar.lunch_slot) if inst.calendar.lunch_slot is not None else None,
                "teachers": [f.name for f in inst.faculty], "sections": [g.name for g in inst.groups],
                "rooms": [r.name for r in inst.rooms], "agent": svc.agent.name, "model": svc.harness.model.name,
                "simulate": svc.simulate, "rules": svc.handbook is not None}

    @app.get("/api/timetable")
    def timetable(s: Me, week: int | None = None, teacher: str | None = None, section: str | None = None) -> dict:
        w = service(s).store.world
        inst = w.instance
        tt = w.timetable(week)
        if tt is None:
            raise HTTPException(404, "no timetable yet")
        d = Directory(inst)
        try:
            fid = d.teacher(teacher).id if teacher else None
            gid = d.section(section).id if section else None
        except Refusal as r:
            raise HTTPException(404, {"message": r.error.message, **r.error.hint}) from None
        rows = [_at(inst, x, p) for sid, p in tt.assignment.items() if sid not in tt.cancelled
                for x in [inst.session_by_id[sid]] if (fid is None or x.faculty == fid) and (gid is None or gid in x.groups)]
        return {"week": week, "own": week is not None and week in w.timetables, "version": w.version, "classes": rows}

    @app.get("/api/fairness")
    def fairness(s: Me) -> dict:
        w = service(s).store.world
        rows, spread = standing(w)
        inst = w.instance
        log = [{"teacher": inst.faculty_by_id[e.stakeholder].name,
                "for": inst.faculty_by_id[e.conceded_to].name if e.conceded_to in inst.faculty_by_id else "the office",
                "week": e.week, "credit": e.credit} for e in w.concessions]
        return {"teachers": rows, "spread": spread, "concessions": log}

    @app.get("/api/rules")
    def rules(s: Me, q: str = "") -> dict:
        hb = service(s).handbook
        if hb is None:
            return {"rules": []}
        found = hb.search(q, k=5) if q.strip() else hb.rules
        return {"rules": [{"id": r.id, "number": r.number, "title": r.title, "text": r.text} for r in found]}

    # -- episodes -----------------------------------------------------------------------

    @app.get("/api/episodes")
    def episodes(s: Me) -> list[dict]:
        eps = [e for e in service(s).episodes.values() if _sees(s, e)]
        return [e.summary() for e in sorted(eps, key=lambda e: e.started, reverse=True)]

    @app.post("/api/episodes")
    def start(s: Me, body: Start) -> dict:
        if s.club:
            raise HTTPException(403, "a club books rooms (Bookings); requests to the office come from staff")
        if not body.request.strip():
            raise HTTPException(422, "say what you need")
        if body.thread is not None:
            try:
                first = service(s).thread(body.thread)[0]
            except NotFound:
                raise HTTPException(404, f"no conversation {body.thread}") from None
            if first.actor != s.person:  # the office reads every conversation but writes only its own
                raise HTTPException(404, f"no conversation {body.thread}")
        try:
            ep = service(s).start(body.request.strip(), s.person, body.week, simulate=body.simulate, thread=body.thread)
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return ep.summary()

    # -- conversations: the episodes of a thread, as a chat -------------------------------------


    @app.get("/api/threads")
    def threads(s: Me) -> list[dict]:
        out = []
        for eps in service(s).threads():
            if not any(_sees(s, e) for e in eps):
                continue
            last = eps[-1]
            out.append({"id": eps[0].thread, "title": eps[0].request, "actor": eps[0].actor_name,
                        "mine": eps[0].actor == s.person, "updated": last.started.isoformat(timespec="seconds"),
                        "status": last.status, "messages": len(eps)})
        return out

    @app.get("/api/threads/{tid}")
    def thread(s: Me, tid: str) -> dict:
        try:
            eps = [e for e in service(s).thread(tid) if _sees(s, e)]
        except NotFound:
            eps = []
        if not eps:
            raise HTTPException(404, f"no conversation {tid}")
        return {"id": tid, "mine": eps[0].actor == s.person,
                "episodes": [{**service(s).detail(e.id), "can": _can(s, e.id), "events": list(e.events)} for e in eps]}

    @app.get("/api/episodes/{eid}")
    def get(s: Me, eid: str) -> dict:
        episode(s, eid)
        return {**service(s).detail(eid), "can": _can(s, eid)}

    @app.get("/api/episodes/{eid}/poll")
    def poll(s: Me, eid: str, after: int = 0, wait: float = 0.0) -> dict:
        episode(s, eid)
        events, busy = service(s).events(eid, after, timeout=min(wait, 25.0))
        return {"events": events, "busy": busy, **service(s).detail(eid), "can": _can(s, eid)}

    @app.get("/api/episodes/{eid}/events")
    async def events(s: Me, eid: str, after: int = 0) -> StreamingResponse:
        episode(s, eid)
        svc = service(s)

        async def stream():
            n = after
            while True:
                got, busy = await asyncio.to_thread(svc.events, eid, n, 15.0)
                for e in got:
                    yield f"id: {e['n']}\nevent: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                n += len(got)
                if not busy and not got:
                    break  # ended, or waiting for an answer: the client reconnects after replying
                if not got:
                    yield ": still working\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def _can(s: Session, eid: str) -> dict:
        ep = service(s).get(eid)
        p = ep.pending
        answer = p is not None and not ep.busy and (p.to == s.person or (p.to == REQUESTER and ep.actor == s.person))
        return {"answer": answer, "decide": s.office and ep.status == "submitted" and not ep.busy}

    @app.post("/api/episodes/{eid}/reply")
    def reply(s: Me, eid: str, body: Answer) -> dict:
        episode(s, eid)
        if not _can(s, eid)["answer"]:
            raise HTTPException(403, "this is not waiting for your answer")
        try:
            service(s).reply(eid, Reply(answer=body.answer, text=body.text, day=body.day, hour=body.hour))
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service(s).detail(eid)

    @app.post("/api/episodes/{eid}/approve")
    def approve(s: Office, eid: str, body: Note | None = None) -> dict:
        episode(s, eid)
        try:
            service(s).approve(eid, note=(body.note.strip() if body else ""))
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service(s).detail(eid)

    @app.post("/api/episodes/{eid}/reject")
    def reject(s: Office, eid: str, body: Note) -> dict:
        episode(s, eid)
        try:
            service(s).reject(eid, body.note)
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service(s).detail(eid)

    @app.post("/api/episodes/{eid}/note")
    def note(s: Office, eid: str, body: Note) -> dict:
        """The office's own message to the requester about a request that reached it."""
        ep = episode(s, eid)
        if ep.status not in FOR_OFFICE or ep.actor == s.person:
            raise HTTPException(409, "this request did not come to the office")
        if not body.note.strip():
            raise HTTPException(422, "write a message")
        service(s).note(eid, body.note.strip())
        return service(s).detail(eid)

    @app.get("/api/office/requests")
    def office_requests(s: Office) -> list[dict]:
        """What reached the office from others, newest first: what still needs it, and what it has dealt with."""
        eps = [e for e in service(s).episodes.values() if e.actor != s.person and e.status in FOR_OFFICE]
        return [{**service(s).detail(e.id), "can": _can(s, e.id), "events": list(e.events)}
                for e in sorted(eps, key=lambda e: e.started, reverse=True)]

    @app.get("/api/inbox")
    def inbox(s: Me) -> list[dict]:
        return service(s).inbox(s.person)

    @app.get("/api/notices")
    def told(s: Me) -> list[dict]:
        return service(s).notices(s.person)

    # -- room bookings ----------------------------------------------------------------------------

    @app.get("/api/bookings")
    def bookings(s: Me, week: int | None = None) -> list[dict]:
        rows = service(s).desk.bookings(week)
        return rows if not s.club else [r for r in rows if r["club"] == s.club]

    @app.post("/api/bookings")
    def book(s: Me, body: BookingIn) -> dict:
        if not (s.club or s.office):
            raise HTTPException(403, "rooms are booked by clubs, or by the office for them")
        club = s.club or (body.club or "").strip()
        if not club:
            raise HTTPException(422, "say which club")
        svc = service(s)
        b = Booking(club=club, **body.model_dump(exclude={"club"}))
        try:
            with svc.lock:
                d = svc.desk.request(b, by=s.person)
        except InvalidChange as e:  # the gate found a class there after all
            raise HTTPException(409, "; ".join(e.problems)) from None
        return {"status": d.status, "checks": d.checks, "room": d.room, "id": d.id, "alternatives": d.alternatives}

    @app.post("/api/bookings/{bid}/approve")
    def book_approve(s: Office, bid: str) -> dict:
        svc = service(s)
        try:
            with svc.lock:
                svc.desk.approve(bid)
        except KeyError:
            raise HTTPException(404, f"no booking {bid} waits for approval") from None
        except InvalidChange as e:
            raise HTTPException(409, "; ".join(e.problems)) from None
        return {"status": "booked"}

    @app.post("/api/bookings/{bid}/refuse")
    def book_refuse(s: Office, bid: str) -> dict:
        try:
            service(s).desk.refuse(bid)
        except KeyError:
            raise HTTPException(404, f"no booking {bid} waits for approval") from None
        return {"status": "refused"}

    @app.delete("/api/bookings/{bid}")
    def book_cancel(s: Me, bid: str) -> dict:
        svc = service(s)
        mine = [r for r in svc.desk.bookings() if r["id"] == bid and (s.office or r["club"] == s.club)]
        if not mine:
            raise HTTPException(404, f"no booking {bid} of yours")
        with svc.lock:
            if mine[0]["status"] == "needs_approval":
                svc.desk.refuse(bid)
            else:
                svc.desk.cancel(bid, s.person)
        return {"status": "cancelled"}

    # -- the semester timetable (the office) -------------------------------------------------------

    @app.post("/api/semester/rebuild")
    def semester_rebuild(s: Office, body: RebuildIn) -> dict:
        try:
            return service(s).start_rebuild(body.from_week)
        except Conflict as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/semester/rebuild")
    def semester_status(s: Office) -> dict:
        return service(s).rebuild_view()

    @app.post("/api/semester/publish")
    def semester_publish(s: Office) -> dict:
        try:
            return service(s).publish_rebuild()
        except Conflict as e:
            raise HTTPException(409, str(e)) from None

    # -- observability (the office) -----------------------------------------------------------

    @app.get("/api/obs/overview")
    def obs_overview(s: Office, days: int = 7) -> dict:
        return obs.overview(days)

    @app.get("/api/obs/episodes")
    def obs_episodes(s: Office, limit: int = 50, status: str | None = None, model: str | None = None,
                     verdict: str | None = None) -> list[dict]:
        return obs.episodes(limit=min(limit, 500), status=status, model=model, verdict=verdict)

    @app.get("/api/obs/trace/{trace_id}")
    def obs_trace(s: Office, trace_id: str) -> list[dict]:
        return obs.trace(trace_id)

    @app.get("/api/obs/evals")
    def obs_evals(s: Office) -> list[dict]:
        return obs.evals()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


__all__ = ["Service", "Users", "create_app"]
