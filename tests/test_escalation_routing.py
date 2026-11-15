"""Every escalation reaches someone who can decide it: a request the policy check says needs approval goes to whoever
the cited rule names (the head of department, the examination cell, the timetable office), and the demo's exam cell
on autopilot answers what is sent to it, as a teacher on autopilot answers a message."""

import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from agents.negotiation import Reply
from agents.policy import PolicyDecision, Verdict
from core.schemas import Channel, Request, Role

DONE = ("awaiting_approval", "published", "denied", "clarification_requested")


class PicksA:  # a teacher offered more than one way to fit the change
    def respond(self, message):
        return Reply(decision="accept", choice="A")


class NeedsApproval:
    """The policy check, saying every request needs approval under ``cited`` (the real rules kept, so the
    orchestrator reads who approves from their text)."""

    def __init__(self, real, cited):
        self.rules, self.cited = real.rules, cited

    def review(self, text, parse):
        return PolicyDecision(Verdict.NEEDS_APPROVAL, cited=self.cited, explanation="it needs approval.")


@pytest.fixture
def world():
    from web.world import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    w.autopilot_delay = 0.1
    for f in w.instance.faculty:
        w.orch.responders[f.id] = PicksA()
    return w


def _ask(world, text, n=0):
    f = world.instance.faculty[n]
    r = Request(id=f"esc-{abs(hash(text))}", channel=Channel.PORTAL, sender_id=f.id, role=Role.FACULTY,
                raw_text=text, received_at=datetime.now(UTC))
    return world.orch.submit(r)


def _settled(case, seconds=60):
    for _ in range(int(seconds / 0.25)):
        if case.status.value in DONE:
            return case
        time.sleep(0.25)
    return case


@pytest.mark.parametrize(("cited", "to", "by"), [(["P-ONLINE"], "hod", "the Head of Department"),
                                                  (["EXAM-3.2"], "exam_cell", "the examination cell"),
                                                  ([], "coordinator", "the timetable office")])
def test_needs_approval_goes_to_whoever_the_rule_names(world, cited, to, by):
    world.orch.policy = NeedsApproval(world.orch.policy, cited)
    case = _ask(world, f"I'm at a conference in week 10 on Tuesday ({to}).")
    assert case.status.value == "escalated" and case.outcome.escalation.to == to, case.reply
    assert f"sent to {by}" in case.reply


def test_the_one_it_names_can_grant_it_and_nobody_else(world):
    from web.app import create_web_app

    world.orch.policy = NeedsApproval(world.orch.policy, ["P-ONLINE"])
    c = TestClient(create_web_app(world))
    case = _ask(world, "I'm at a conference in week 10 on Wednesday.")
    people = c.get("/api/personas").json()
    hod = {"X-User": next(p["id"] for p in people if p["role"] == "hod")}
    teacher = {"X-User": world.instance.faculty[1].id}
    assert c.get(f"/api/cases/{case.id}", headers=hod).json()["can_decide"]
    assert c.post(f"/api/cases/{case.id}/decide", json={"grant": True}, headers=teacher).status_code == 403
    assert c.post(f"/api/cases/{case.id}/decide", json={"grant": True}, headers=hod).status_code == 200
    assert _settled(case).status.value == "awaiting_approval", (case.status, case.reply)


def test_the_exam_cell_on_autopilot_answers_by_itself(world):
    world.autopilot[world.exam_cell()] = True
    case = _ask(world, "I'm at a conference in week 7 on Thursday, so I can't take my classes then.")
    assert case.outcome.escalation.to == "exam_cell"
    assert _settled(case).status.value == "awaiting_approval", (case.status, case.reply)
    assert case.decision["by"] == world.exam_cell() and "simulated" in case.decision["note"]


def test_switching_the_exam_cells_autopilot_on_answers_what_already_waits(world):
    from web.app import create_web_app

    c = TestClient(create_web_app(world))
    case = _ask(world, "I'm at a conference in week 15 on Wednesday, so I can't take my classes then.")
    time.sleep(0.5)
    assert case.status.value == "escalated"  # autopilot off: it waits for the exam cell
    cell = world.exam_cell()
    assert cell in c.get("/api/autopilot", headers={"X-User": cell}).json()
    assert c.post("/api/autopilot", json={"person": cell, "on": True}, headers={"X-User": cell}).status_code == 200
    assert _settled(case).status.value == "awaiting_approval", (case.status, case.reply)
