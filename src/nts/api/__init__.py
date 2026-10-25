"""The agent behind HTTP: sign in, start an episode, follow its turns as server-sent events, answer asks, approve.

    python -m nts serve --store runs/agentic/demo.jsonl --model llama --adapter <lora> --port 8100
    # then open http://127.0.0.1:8100/ (the page in nts/api/static), or call the API:

    POST /api/login                 {"dept": "demo", "name": "Dr. Iyer" | "office", "password": "..."} -> a session cookie
    POST /api/episodes              {"request": "...", "week": 5, "simulate": false}   (as the signed-in person)
    GET  /api/episodes/{id}/events  text/event-stream: model, tool, reply, waiting, end, decision
    GET  /api/episodes/{id}/poll?after=n   the same events as JSON (for clients without SSE)
    POST /api/episodes/{id}/reply   {"answer": "yes" | "no" | "said" | "no_reply", "text": "..."}   (the one asked)
    POST /api/episodes/{id}/approve | /reject {"note": "..."}                             (the office)
    GET  /api/inbox                 what waits for my answer
    GET  /api/department, /api/timetable?week=5&teacher=Dr. Iyer, /api/fairness, /api/rules?q=lunch
    GET  /api/obs/overview, /api/obs/episodes, /api/obs/trace/{id}, /api/obs/evals          (the office)

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
from pydantic import BaseModel

from nts.fairness import standing
from nts.tools import Refusal, Reply
from nts.tools.draft import OFFICE
from nts.tools.names import FULL_DAY, Directory, hour
from nts.tools.read import _at
from nts.tools.talk import REQUESTER

from .auth import Session, Sessions, Users
from .obs import Observability
from .service import Conflict, Episode, NotFound, Service

STATIC = Path(__file__).parent / "static"
COOKIE = "nts_session"


class Login(BaseModel):
    dept: str
    name: str
    password: str = ""


class Start(BaseModel):
    request: str
    week: int | None = None
    simulate: bool | None = None


class Answer(BaseModel):
    answer: Literal["yes", "no", "said", "no_reply"] = "said"
    text: str = ""


class Note(BaseModel):
    note: str = ""


def create_app(departments: dict[str, Service] | Service, users: Users | None = None, *,
               telemetry: Path = Path("runs/telemetry"), evals: Path = Path("runs/eval"),
               secure_cookie: bool = False) -> FastAPI:
    depts = departments if isinstance(departments, dict) else {"demo": departments}
    sessions = Sessions(users or Users.open())
    obs = Observability(telemetry, evals)
    app = FastAPI(title="Negotiate, Then Solve: agent API")

    # -- who is calling ---------------------------------------------------------------

    def session(request: Request) -> Session:
        token = request.cookies.get(COOKIE)
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:]
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
        if not (s.office or ep.actor == s.person or _asks(ep, s.person)):
            raise HTTPException(404, f"no episode {eid}")  # not 403: do not say it exists
        return ep

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
        sessions.end(request.cookies.get(COOKIE, ""))
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
        eps = [e for e in service(s).episodes.values() if s.office or e.actor == s.person or _asks(e, s.person)]
        return [e.summary() for e in sorted(eps, key=lambda e: e.started, reverse=True)]

    @app.post("/api/episodes")
    def start(s: Me, body: Start) -> dict:
        if not body.request.strip():
            raise HTTPException(422, "say what you need")
        return service(s).start(body.request.strip(), s.person, body.week, simulate=body.simulate).summary()

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
            service(s).reply(eid, Reply(answer=body.answer, text=body.text))
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service(s).detail(eid)

    @app.post("/api/episodes/{eid}/approve")
    def approve(s: Office, eid: str) -> dict:
        episode(s, eid)
        try:
            service(s).approve(eid)
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

    @app.get("/api/inbox")
    def inbox(s: Me) -> list[dict]:
        return service(s).inbox(s.person)

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
