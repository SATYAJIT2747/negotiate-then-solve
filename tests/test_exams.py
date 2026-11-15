"""The examination regulations as a rule (pipeline.exams), and the request pipeline applying it.
The regulations come from data/policies/exam-regulations.pdf (EXAM-3.1, 3.2, 3.5)."""

import pytest

from pipeline.exams import exam_ruling, weeks_in

RULES = {
    "EXAM-3.1": "Mid-semester examinations are held in week 8 and end-semester examinations in week 16 of the teaching "
                "calendar. No regular classes are scheduled in these weeks.",
    "EXAM-3.2": "No change to the teaching timetable may take effect in weeks 7, 8, 15 or 16 without the written "
                "approval of the Controller of Examinations. Requests for these weeks are forwarded to the "
                "examination cell.",
    "EXAM-3.5": "Extra revision classes in week 7 or week 15 are allowed only outside the regular timetable.",
}


def test_the_weeks_come_from_the_rules_text():
    assert weeks_in(RULES["EXAM-3.2"]) == {7, 8, 15, 16}
    assert weeks_in(RULES["EXAM-3.1"]) == {8, 16}
    assert weeks_in("changes in weeks 3 and 4, or week 10") == {3, 4, 10}


@pytest.mark.parametrize(("weeks", "action", "first"), [
    ([7], "forward", "EXAM-3.2"), ([15], "forward", "EXAM-3.2"),
    ([8], "refuse", "EXAM-3.1"), ([16], "refuse", "EXAM-3.1"), ([7, 8], "refuse", "EXAM-3.1"),
])
def test_a_change_near_an_examination(weeks, action, first):
    r = exam_ruling(weeks, RULES)
    assert r is not None and r.action == action and r.cited[0] == first
    assert ("examination cell" in r.message) == (action == "forward")


def test_other_weeks_and_departments_without_the_regulations_are_left_alone():
    assert exam_ruling([6, 9, None], RULES) is None
    assert exam_ruling([], RULES) is None
    assert exam_ruling([7, 8], {"P-LUNCH": "no classes 1-2 pm"}) is None


def test_an_extra_class_cites_the_revision_rule():
    r = exam_ruling([15], RULES, extra=True)
    assert r.action == "forward" and r.cited == ["EXAM-3.2", "EXAM-3.5"] and "two hours" in r.message
    assert exam_ruling([15], RULES).cited == ["EXAM-3.2"]


@pytest.fixture(scope="module")
def world():
    from web.world import World

    return World(parser_mode="offline", seed_history=False, deadline=60)


def _ask(world, text):
    from datetime import UTC, datetime

    from core.schemas import Channel, Request, Role

    f = world.instance.faculty[0]
    r = Request(id=f"exam-{abs(hash(text))}", channel=Channel.PORTAL, sender_id=f.id, role=Role.FACULTY,
                raw_text=text, received_at=datetime.now(UTC))
    return world.orch.submit(r)


@pytest.mark.parametrize(("week", "status", "cited"), [(15, "escalated", "EXAM-3.2"), (8, "denied", "EXAM-3.1")])
def test_the_pipeline_applies_the_regulations(world, week, status, cited):
    case = _ask(world, f"I'm at a conference in week {week} on Tuesday.")
    assert "+exams" in case.route and case.status.value == status and case.policy.cited[0] == cited


def test_a_standing_change_is_not_an_examination_week_change():
    assert exam_ruling(range(5, 17), RULES) is None  # "for the rest of the semester"
    assert exam_ruling(range(1, 17), RULES) is None  # "from now on"
    assert exam_ruling([6, 7, 8], RULES).action == "refuse"  # a few weeks that take in an examination week


@pytest.mark.parametrize("text", ["I'm unavailable on Friday afternoons for the rest of the semester.",
                                  "I'm at a conference in week 6 on Tuesday."])
def test_the_pipeline_leaves_ordinary_changes_alone(world, text):
    assert "+exams" not in _ask(world, text).route


def test_the_exam_cell_decides_a_change_next_to_the_examinations(world):
    """Forwarded to the examination cell: the escalation is addressed to it (the exam_cell role), which can open the
    request, sees it waiting for its decision, and granting it solves the change and sends it on for approval."""
    from fastapi.testclient import TestClient

    from web.app import create_web_app

    from agents.negotiation import Reply

    class PicksA:  # the teacher, offered more than one way to fit the change
        def respond(self, message):
            return Reply(decision="accept", choice="A")

    c = TestClient(create_web_app(world))
    world.orch.responders[world.instance.faculty[0].id] = PicksA()
    case = _ask(world, "I'm at a conference in week 7 on Wednesday, so I can't take my classes then.")
    assert case.status.value == "escalated" and case.outcome.escalation.to == "exam_cell", case.reply
    assert case.outcome.escalation.rule == "EXAM-3.2" and "week 7" in case.outcome.escalation.text
    cell = {"X-User": next(p["id"] for p in c.get("/api/personas").json() if p["role"] == "exam_cell")}
    assert c.get(f"/api/cases/{case.id}", headers=cell).status_code == 200
    assert any(i.get("action") == "decide" and i.get("case") == case.id for i in c.get("/api/feed", headers=cell).json())
    teacher = {"X-User": next(p["id"] for p in c.get("/api/personas").json() if p["role"] == "faculty")}
    assert c.post(f"/api/cases/{case.id}/decide", json={"grant": True}, headers=teacher).status_code == 403
    r = c.post(f"/api/cases/{case.id}/decide", json={"grant": True}, headers=cell)
    assert r.status_code == 200, r.text
    import time

    for _ in range(120):  # granted: solved in the background ({"status": "deciding"})
        if case.status.value in ("awaiting_approval", "published", "denied", "clarification_requested"):
            break
        time.sleep(0.5)
    assert case.status.value == "awaiting_approval" and case.decision["granted"], (case.status, case.reply)


def test_the_demo_simulates_the_exam_cells_reply(world):
    """As the inbox simulates a teacher's reply: the office (or the cell) has the exam cell answer; it grants, saying
    why, and the request goes on to approval. A teacher cannot, nor can a request no longer waiting."""
    import time

    from fastapi.testclient import TestClient

    from agents.negotiation import Reply
    from web.app import create_web_app

    class PicksA:
        def respond(self, message):
            return Reply(decision="accept", choice="A")

    c = TestClient(create_web_app(world))
    world.orch.responders[world.instance.faculty[0].id] = PicksA()
    case = _ask(world, "I'm away at a workshop in week 15 on Thursday, so I can't take my classes then.")
    assert case.status.value == "escalated" and case.outcome.escalation.to == "exam_cell", case.reply
    office, teacher = {"X-User": "C-TT"}, {"X-User": world.instance.faculty[1].id}
    shown = c.get(f"/api/cases/{case.id}", headers=office).json()
    assert shown["can_simulate"] and shown["outcome"]["escalation"]["to_name"] == "the examination cell"
    assert c.post(f"/api/cases/{case.id}/simulate-decision", headers=teacher).status_code == 403
    assert c.post(f"/api/cases/{case.id}/simulate-decision", headers=office).json() == {"status": "deciding"}
    for _ in range(120):
        if case.status.value in ("awaiting_approval", "published", "denied", "clarification_requested"):
            break
        time.sleep(0.5)
    assert case.status.value == "awaiting_approval", (case.status, case.reply)
    assert case.decision["by"] == world.exam_cell() and "simulated" in case.decision["note"]
    assert c.post(f"/api/cases/{case.id}/simulate-decision", headers=office).status_code == 409
