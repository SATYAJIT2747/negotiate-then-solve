"""The agent behind HTTP: start an episode, follow its turns as server-sent events, answer its asks, approve.

    python -m nts serve --store runs/agentic/demo.jsonl --model llama --adapter <lora> --port 8100
    # then open http://127.0.0.1:8100/ (the page in nts/api/static), or call the API:

    POST /api/episodes              {"request": "...", "actor": "Dr. Iyer" | "office", "week": 5, "simulate": false}
    GET  /api/episodes/{id}/events  text/event-stream: model, tool, reply, waiting, end, decision
    GET  /api/episodes/{id}/poll?after=n   the same events as JSON (for clients without SSE)
    POST /api/episodes/{id}/reply   {"answer": "yes" | "no" | "said" | "no_reply", "text": "..."}
    POST /api/episodes/{id}/approve | /reject {"note": "..."}
    GET  /api/inbox?person=Dr. Iyer what waits for that person's answer
    GET  /api/department, /api/timetable?week=5&teacher=Dr. Iyer

Authentication is not done here: whoever calls says who they are (``actor``, ``person``). Behind a
real login, the server would take those from the session instead.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from nts.tools import Refusal, Reply
from nts.tools.draft import OFFICE
from nts.tools.names import FULL_DAY, Directory, hour
from nts.tools.read import _at

from .service import Conflict, NotFound, Service

STATIC = Path(__file__).parent / "static"


class Start(BaseModel):
    request: str
    actor: str = "office"
    week: int | None = None
    simulate: bool | None = None


class Answer(BaseModel):
    answer: Literal["yes", "no", "said", "no_reply"] = "said"
    text: str = ""


class Note(BaseModel):
    note: str = ""


def create_app(service: Service) -> FastAPI:
    app = FastAPI(title="Negotiate, Then Solve: agent API")
    store = service.store

    def who_is(name: str) -> str:
        if name.lower() in ("office", OFFICE, "the timetable office"):
            return OFFICE
        try:
            return Directory(store.world.instance).teacher(name).id
        except Refusal as r:
            raise HTTPException(404, {"message": r.error.message, **r.error.hint}) from None

    def episode(eid: str):
        try:
            return service.get(eid)
        except NotFound:
            raise HTTPException(404, f"no episode {eid}") from None

    @app.get("/api/department")
    def department() -> dict:
        w = store.world
        inst = w.instance
        return {"version": w.version, "weeks": inst.calendar.weeks, "days": [FULL_DAY[d] for d in inst.calendar.days],
                "hours": [hour(q) for q in range(inst.calendar.slots_per_day + 1)],
                "lunch": hour(inst.calendar.lunch_slot) if inst.calendar.lunch_slot is not None else None,
                "teachers": [f.name for f in inst.faculty], "sections": [g.name for g in inst.groups],
                "rooms": [r.name for r in inst.rooms], "agent": service.agent.name,
                "model": service.harness.model.name, "simulate": service.simulate}

    @app.get("/api/timetable")
    def timetable(week: int | None = None, teacher: str | None = None, section: str | None = None) -> dict:
        w = store.world
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
        rows = [_at(inst, s, p) for sid, p in tt.assignment.items() if sid not in tt.cancelled
                for s in [inst.session_by_id[sid]] if (fid is None or s.faculty == fid) and (gid is None or gid in s.groups)]
        own = week is not None and week in w.timetables
        return {"week": week, "own": own, "version": w.version, "classes": rows}

    @app.get("/api/episodes")
    def episodes() -> list[dict]:
        return [e.summary() for e in sorted(service.episodes.values(), key=lambda e: e.started, reverse=True)]

    @app.post("/api/episodes")
    def start(body: Start) -> dict:
        if not body.request.strip():
            raise HTTPException(422, "say what you need")
        ep = service.start(body.request.strip(), who_is(body.actor), body.week, simulate=body.simulate)
        return ep.summary()

    @app.get("/api/episodes/{eid}")
    def get(eid: str) -> dict:
        episode(eid)
        return service.detail(eid)

    @app.get("/api/episodes/{eid}/poll")
    def poll(eid: str, after: int = 0, wait: float = 0.0) -> dict:
        episode(eid)
        events, busy = service.events(eid, after, timeout=min(wait, 25.0))
        return {"events": events, "busy": busy, **service.detail(eid)}

    @app.get("/api/episodes/{eid}/events")
    async def events(eid: str, after: int = 0) -> StreamingResponse:
        episode(eid)

        async def stream():
            n = after
            while True:
                got, busy = await asyncio.to_thread(service.events, eid, n, 15.0)
                for e in got:
                    yield f"id: {e['n']}\nevent: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                n += len(got)
                if not busy and not got:
                    break  # ended, or waiting for an answer: the client reconnects after replying
                if not got:
                    yield ": still working\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/episodes/{eid}/reply")
    def reply(eid: str, body: Answer) -> dict:
        episode(eid)
        try:
            service.reply(eid, Reply(answer=body.answer, text=body.text))
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service.detail(eid)

    @app.post("/api/episodes/{eid}/approve")
    def approve(eid: str) -> dict:
        episode(eid)
        try:
            service.approve(eid)
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service.detail(eid)

    @app.post("/api/episodes/{eid}/reject")
    def reject(eid: str, body: Note) -> dict:
        episode(eid)
        try:
            service.reject(eid, body.note)
        except Conflict as e:
            raise HTTPException(409, str(e)) from None
        return service.detail(eid)

    @app.get("/api/inbox")
    def inbox(person: str) -> list[dict]:
        return service.inbox(who_is(person))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


__all__ = ["Service", "create_app"]
