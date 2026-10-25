import json

import pytest
from fastapi.testclient import TestClient

from nts.api import Service, create_app
from nts.models import ScriptedModel, call, say
from nts.telemetry import MemorySink, Tracer
from nts.tools.names import FULL_DAY


def client(store, steps, **kw):
    service = Service(store, ScriptedModel(steps), Tracer([MemorySink()]), **kw)
    return TestClient(create_app(service)), service


def poll_until_idle(c, eid):
    out = c.get(f"/api/episodes/{eid}/poll", params={"wait": 10}).json()
    while out["busy"]:
        out = c.get(f"/api/episodes/{eid}/poll", params={"after": len(out["events"]), "wait": 10}).json()
    return c.get(f"/api/episodes/{eid}/poll").json()


def test_department_and_timetable(store):
    c, _ = client(store, [])
    dep = c.get("/api/department").json()
    assert dep["teachers"] and dep["weeks"] == 16 and dep["agent"] == "coordinator"
    t = dep["teachers"][0]
    rows = c.get("/api/timetable", params={"week": 5, "teacher": t}).json()["classes"]
    assert rows and all(r["teacher"] == t for r in rows)
    assert c.get("/api/timetable", params={"teacher": "Dr. Nobody"}).status_code == 404


def test_an_ask_waits_in_the_inbox_until_answered(store):
    teacher = store.world.instance.faculty[2]
    c, service = client(store, [call("ask_requester", question="Which class?"), say("Thanks, noted.")])
    ep = c.post("/api/episodes", json={"request": "move my class", "actor": teacher.name, "week": 5}).json()
    service.wait(ep["id"])
    got = poll_until_idle(c, ep["id"])
    assert got["status"] == "waiting" and got["pending"]["to_requester"] and got["pending"]["text"] == "Which class?"
    assert [e["type"] for e in got["events"]] == ["model", "waiting"]
    inbox = c.get("/api/inbox", params={"person": teacher.name}).json()
    assert [i["id"] for i in inbox] == [ep["id"]]
    assert c.get("/api/inbox", params={"person": "office"}).json() == []

    assert c.post(f"/api/episodes/{ep['id']}/reply", json={"text": "The Monday one."}).status_code == 200
    service.wait(ep["id"])
    done = c.get(f"/api/episodes/{ep['id']}/poll").json()
    assert done["status"] == "answered" and done["text"] == "Thanks, noted."
    kinds = [e["type"] for e in done["events"]]
    assert kinds == ["model", "waiting", "reply", "model", "end"]
    assert done["events"][2]["says"] == "The Monday one."
    assert c.post(f"/api/episodes/{ep['id']}/reply", json={"text": "again"}).status_code == 409


def test_a_submitted_draft_is_approved_through_the_kernel(store):
    inst = store.world.instance
    t = inst.faculty[1]
    day = next(d for d in inst.calendar.days)
    steps = [call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]], from_week=6, to_week=6,
                  reason="a conference"),
             call("check_draft"), call("submit", message="Your classes that day move.")]
    c, service = client(store, steps)
    ep = c.post("/api/episodes", json={"request": "away Monday week 6", "actor": t.name, "week": 5}).json()
    service.wait(ep["id"])
    got = c.get(f"/api/episodes/{ep['id']}").json()
    assert got["status"] == "submitted" and any("not available" in d for d in got["draft"])
    before = store.version
    done = c.post(f"/api/episodes/{ep['id']}/approve").json()
    assert got["decision"] is None and done["status"] == "approved" and store.version != before
    assert done["decision"]["approved"] and done["decision"]["version"] == store.version
    assert c.post(f"/api/episodes/{ep['id']}/approve").status_code == 409


def test_events_stream_as_sse_and_end_when_the_episode_does(store):
    c, service = client(store, [call("describe", what="rooms"), say("Four rooms.")])
    ep = c.post("/api/episodes", json={"request": "rooms?", "actor": "office"}).json()
    service.wait(ep["id"])
    with c.stream("GET", f"/api/episodes/{ep['id']}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        data = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert [e["type"] for e in data] == ["model", "tool", "model", "end"]
    assert data[1]["name"] == "describe" and data[1]["ok"] and data[-1]["status"] == "answered"
    with c.stream("GET", f"/api/episodes/{ep['id']}/events", params={"after": 3}) as r:
        assert [json.loads(x[5:])["type"] for x in r.iter_lines() if x.startswith("data:")] == ["end"]


def test_simulated_people_answer_at_once(store):
    c, service = client(store, [call("ask_requester", question="Which week?"), say("I will ask the office.")],
                        simulate=True)
    ep = c.post("/api/episodes", json={"request": "move it", "actor": store.world.instance.faculty[0].name}).json()
    service.wait(ep["id"])
    got = c.get(f"/api/episodes/{ep['id']}/poll").json()
    assert got["status"] == "answered" and got["events"][1]["answer"] == "no_reply"


@pytest.mark.parametrize("body", [{"request": " "}, {"request": "x", "actor": "Dr. Nobody"}])
def test_bad_starts_are_refused(store, body):
    c, _ = client(store, [])
    assert c.post("/api/episodes", json=body).status_code in (404, 422)
