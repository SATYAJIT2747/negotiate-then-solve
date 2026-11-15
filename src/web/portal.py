"""Web portal and channel endpoints (proposal L0 and L6), FastAPI.

    uv run uvicorn web.portal:demo_app --factory --reload     # synthetic department, no API key needed

Identity: the ``X-User`` header stands in for the college SSO session
(OAuth/OIDC in production); a request body can never set who is asking.

* POST /requests                 portal request from the signed-in user
* POST /ingest/email             raw RFC 822 message (from the IMAP poller)
* POST /ingest/messaging         Slack-shaped JSON from a mock adapter
* GET  /cases/{id}               status, reply and trace of one request
* GET  /approvals                changes awaiting the coordinator
* POST /approvals/{id}/approve   publish (coordinator only)
* POST /approvals/{id}/reject    back to negotiation (coordinator only)
* GET  /timetable                the published timetable
* GET  /                         a minimal HTML page for trying it out
"""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pipeline.ingest import Intake, UnknownSender
from pipeline.orchestrator import Case, NotAuthorised, Orchestrator


class PortalRequest(BaseModel):
    text: str


def _case_view(o: Orchestrator, case: Case) -> dict:
    return {
        "id": case.id,
        "status": case.status.value,
        "route": case.route,
        "reply": case.reply,
        "constraints": [c.id for c in case.constraints],
        "proposal": case.proposal.version if case.proposal else None,
        "fairness": case.fairness,
        "escalation": case.outcome.escalation.text if case.outcome and case.outcome.escalation else None,
        "trace": [{k: v for k, v in e.items() if k != "case"} for e in o.store.events(case.id)],
    }


def create_app(orchestrator: Orchestrator, intake: Intake) -> FastAPI:
    app = FastAPI(title="Negotiate, Then Solve")
    o = orchestrator

    def _submit(r) -> dict:
        if r is None:
            return {"status": "duplicate"}
        return _case_view(o, o.submit(r))

    @app.post("/requests")
    def portal(body: PortalRequest, x_user: str = Header(...)) -> dict:
        try:
            return _submit(intake.from_portal(x_user, body.text))
        except UnknownSender:
            raise HTTPException(403, "unknown user") from None

    @app.post("/ingest/email")
    async def ingest_email(request: Request) -> dict:
        try:
            return _submit(intake.from_email(await request.body()))
        except UnknownSender:
            raise HTTPException(403, "unknown sender") from None

    @app.post("/ingest/messaging")
    def ingest_messaging(payload: dict) -> dict:
        try:
            return _submit(intake.from_messaging(payload))
        except UnknownSender:
            raise HTTPException(403, "unknown sender") from None

    @app.get("/cases/{case_id}")
    def case(case_id: str) -> dict:
        if case_id not in o.cases:
            raise HTTPException(404)
        return _case_view(o, o.cases[case_id])

    @app.get("/approvals")
    def approvals() -> list[dict]:
        return [_case_view(o, c) for c in o.pending()]

    @app.post("/approvals/{case_id}/approve")
    def approve(case_id: str, x_user: str = Header(...)) -> dict:
        try:
            v = o.approve(case_id, x_user)
        except NotAuthorised as e:
            raise HTTPException(403, str(e)) from None
        except (KeyError, ValueError) as e:
            raise HTTPException(409, str(e)) from None
        return {"published": v.version if v else None, "notified": o.cases[case_id].notices}

    @app.post("/approvals/{case_id}/reject")
    def reject(case_id: str, reason: str = "", x_user: str = Header(...)) -> dict:
        try:
            o.reject(case_id, x_user, reason)
        except NotAuthorised as e:
            raise HTTPException(403, str(e)) from None
        return {"status": o.cases[case_id].status.value}

    @app.get("/timetable")
    def timetable(week: int | None = None) -> dict:
        v = o.store.current_version(week) or o.store.current_version()
        return {"version": v.version, "week": v.week,
                "assignment": {s: p.model_dump() for s, p in v.assignment.items()}} if v else {}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return PAGE

    return app


PAGE = """<!doctype html><meta charset="utf-8"><title>Timetable requests</title>
<style>body{font:15px system-ui;max-width:44rem;margin:2rem auto;padding:0 1rem}textarea,input{width:100%;
font:inherit;margin:.3rem 0}pre{background:#f4f4f4;padding:.8rem;white-space:pre-wrap}</style>
<h1>Timetable requests</h1>
<label>Signed in as (SSO stand-in) <input id=u value="F-301"></label>
<textarea id=t rows=4 placeholder="e.g. I'm at a conference in week 7, Tuesday to Thursday."></textarea>
<button onclick="send()">Send</button><pre id=o></pre>
<script>
async function send(){const r=await fetch('/requests',{method:'POST',headers:{'Content-Type':'application/json',
'X-User':document.getElementById('u').value},body:JSON.stringify({text:document.getElementById('t').value})});
document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}
</script>"""


def demo_app() -> FastAPI:
    """A synthetic department wired with scripted components, so the portal
    runs without an API key. Requests are parsed by a small rule-based stub;
    swap in ``SystemTwoParser`` with a Gemini client for real parsing."""
    from agents.explainer import Explainer
    from agents.negotiation import Negotiator
    from agents.priority import PriorityModel
    from evaluation.benchmark import build
    from pipeline.ingest import Directory
    from pipeline.store import Store

    sc = build("contention", 1, seed=7)
    store = Store()
    pm = PriorityModel(sc.instance)
    neg = Negotiator(sc.instance, priority=pm, explainer=Explainer(sc.instance))
    orch = Orchestrator(sc.instance, store, parser=_EchoParser(sc), negotiator=neg)
    orch.bootstrap(sc.base, sc.baseline)
    return create_app(orch, Intake(Directory.from_instance(sc.instance), store))


class _EchoParser:
    """Demo stand-in: compiles nothing and asks for clarification, so the
    portal flow can be clicked through without an LLM."""

    def __init__(self, sc) -> None:
        self.sc = sc

    def parse(self, request):
        from language.parsing import ParseOutput, postprocess

        out = ParseOutput(request_type="preference", action="clarify", missing=["days"],
                          clarifying_question="Demo mode: which days and times do you mean?")
        return postprocess(self.sc.instance, request, out)
