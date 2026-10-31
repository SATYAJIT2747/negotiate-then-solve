import json

import pytest
from fastapi.testclient import TestClient

from nts.agents import COORDINATOR_V3
from nts.api import Service, Users, create_app
from nts.api.auth import check_password, hash_password
from nts.models import ScriptedModel, call, say
from nts.policy import Handbook
from nts.telemetry import MemorySink, SqliteSink, Tracer
from nts.tools.names import FULL_DAY


def app(store, steps, *, users=None, tmp_path=None, **kw):
    sinks = [MemorySink()] + ([SqliteSink(tmp_path / "index.sqlite")] if tmp_path else [])
    service = Service(store, ScriptedModel(steps), Tracer(sinks), **kw)
    extra = {"telemetry": tmp_path, "evals": tmp_path} if tmp_path else {}
    return create_app({"demo": service}, users, **extra), service


def signed_in(application, name, password=""):
    c = TestClient(application)
    r = c.post("/api/login", json={"dept": "demo", "name": name, "password": password})
    assert r.status_code == 200, r.text
    return c


def test_nothing_without_signing_in(store):
    a, _ = app(store, [])
    c = TestClient(a)
    assert c.get("/api/department").status_code == 401
    assert c.get("/api/departments").json()["departments"][0]["id"] == "demo"
    assert c.post("/api/login", json={"dept": "demo", "name": "Dr. Nobody"}).status_code == 401
    assert c.post("/api/login", json={"dept": "other", "name": "office"}).status_code == 404


def test_department_and_timetable(store):
    a, _ = app(store, [])
    t = store.world.instance.faculty[0].name
    c = signed_in(a, t)
    assert c.get("/api/me").json() == {"dept": "demo", "name": t, "role": "teacher"}
    dep = c.get("/api/department").json()
    assert dep["teachers"] and dep["weeks"] == 16 and dep["agent"] == "coordinator"
    rows = c.get("/api/timetable", params={"week": 5, "teacher": t}).json()["classes"]
    assert rows and all(r["teacher"] == t for r in rows)
    assert c.get("/api/timetable", params={"teacher": "Dr. Nobody"}).status_code == 404
    c.post("/api/logout")
    assert c.get("/api/me").status_code == 401


def test_passwords_when_there_is_a_users_file(store, tmp_path):
    stored = hash_password("correct horse")
    assert check_password("correct horse", stored) and not check_password("wrong", stored)
    users = Users.load(tmp_path / "users.json")
    t = store.world.instance.faculty[0].name
    users.set("demo", t, "correct horse")
    users.save()
    a, _ = app(store, [], users=Users.load(tmp_path / "users.json"))
    c = TestClient(a)
    assert c.post("/api/login", json={"dept": "demo", "name": t, "password": "wrong"}).status_code == 401
    assert c.post("/api/login", json={"dept": "demo", "name": "office", "password": ""}).status_code == 401
    assert c.post("/api/login", json={"dept": "demo", "name": t, "password": "correct horse"}).status_code == 200
    assert c.get("/api/departments").json()["passwords"] is True


def test_an_ask_waits_for_the_person_asked_and_no_one_else(store):
    inst = store.world.instance
    me, other = inst.faculty[2].name, inst.faculty[3].name
    a, service = app(store, [call("ask_requester", question="Which class?"), say("Thanks, noted.")])
    mine, theirs, office = signed_in(a, me), signed_in(a, other), signed_in(a, "office")
    ep = mine.post("/api/episodes", json={"request": "move my class", "week": 5}).json()
    service.wait(ep["id"])
    got = mine.get(f"/api/episodes/{ep['id']}/poll").json()
    assert got["status"] == "waiting" and got["pending"]["to_requester"] and got["can"]["answer"]
    assert [e["type"] for e in got["events"]] == ["model", "waiting"]
    assert [i["id"] for i in mine.get("/api/inbox").json()] == [ep["id"]]

    assert theirs.get(f"/api/episodes/{ep['id']}").status_code == 404  # not theirs: not even shown
    assert theirs.get("/api/episodes").json() == [] and theirs.get("/api/inbox").json() == []
    assert office.get(f"/api/episodes/{ep['id']}").status_code == 404  # a teacher's talk with the agent is theirs
    assert office.post(f"/api/episodes/{ep['id']}/reply", json={"text": "x"}).status_code == 404

    assert mine.post(f"/api/episodes/{ep['id']}/reply", json={"text": "The Monday one."}).status_code == 200
    service.wait(ep["id"])
    done = mine.get(f"/api/episodes/{ep['id']}/poll").json()
    assert done["status"] == "answered" and done["text"] == "Thanks, noted."
    assert [e["type"] for e in done["events"]] == ["model", "waiting", "reply", "model", "end"]
    assert mine.post(f"/api/episodes/{ep['id']}/reply", json={"text": "again"}).status_code == 403


def test_only_the_office_approves(store):
    inst = store.world.instance
    t = inst.faculty[1]
    steps = [call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[inst.calendar.days[0]]],
                  from_week=6, to_week=6, reason="a conference"),
             call("check_draft"), call("submit", message="Your classes that day move.")]
    a, service = app(store, steps)
    mine, office = signed_in(a, t.name), signed_in(a, "office")
    ep = mine.post("/api/episodes", json={"request": "away Monday week 6", "week": 5}).json()
    service.wait(ep["id"])
    got = mine.get(f"/api/episodes/{ep['id']}").json()
    assert got["status"] == "submitted" and any("not available" in d for d in got["draft"]) and not got["can"]["decide"]
    assert mine.post(f"/api/episodes/{ep['id']}/approve").status_code == 403
    assert office.get(f"/api/episodes/{ep['id']}").json()["can"]["decide"]
    before = store.version
    done = office.post(f"/api/episodes/{ep['id']}/approve").json()
    assert done["status"] == "approved" and store.version != before
    assert done["decision"]["approved"] and done["decision"]["version"] == store.version
    assert office.post(f"/api/episodes/{ep['id']}/approve").status_code == 409


def test_events_stream_as_sse_and_end_when_the_episode_does(store):
    a, service = app(store, [call("describe", what="rooms"), say("Four rooms.")])
    c = signed_in(a, "office")
    ep = c.post("/api/episodes", json={"request": "rooms?"}).json()
    service.wait(ep["id"])
    with c.stream("GET", f"/api/episodes/{ep['id']}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        data = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert [e["type"] for e in data] == ["model", "tool", "model", "end"]
    assert data[1]["name"] == "describe" and data[1]["ok"] and data[-1]["status"] == "answered"
    with c.stream("GET", f"/api/episodes/{ep['id']}/events", params={"after": 3}) as r:
        assert [json.loads(x[5:])["type"] for x in r.iter_lines() if x.startswith("data:")] == ["end"]


def test_simulated_people_answer_at_once(store):
    a, service = app(store, [call("ask_requester", question="Which week?"), say("I will ask the office.")],
                     simulate=True)
    c = signed_in(a, store.world.instance.faculty[0].name)
    ep = c.post("/api/episodes", json={"request": "move it"}).json()
    service.wait(ep["id"])
    got = c.get(f"/api/episodes/{ep['id']}/poll").json()
    assert got["status"] == "answered" and got["events"][1]["answer"] == "no_reply"


def test_fairness_rules_and_observability(store, tmp_path):
    a, service = app(store, [call("rules", question="lunch"), say("Rule 2.2 [P-LUNCH].")], tmp_path=tmp_path,
                     agent=COORDINATOR_V3, handbook=Handbook.load())
    teacher, office = signed_in(a, store.world.instance.faculty[0].name), signed_in(a, "office")
    f = teacher.get("/api/fairness").json()
    assert len(f["teachers"]) == len(store.world.instance.faculty) and f["spread"]["gini"] == 0 and f["concessions"] == []
    assert teacher.get("/api/rules", params={"q": "lunch break"}).json()["rules"][0]["id"] == "P-LUNCH"
    assert len(teacher.get("/api/rules").json()["rules"]) == 16

    ep = office.post("/api/episodes", json={"request": "lunch?"}).json()
    service.wait(ep["id"])
    assert teacher.get("/api/obs/overview").status_code == 403
    over = office.get("/api/obs/overview").json()
    assert over["episodes"] == 1 and over["status"] == {"answered": 1} and over["tools"] == [["rules", 1]]
    rows = office.get("/api/obs/episodes").json()
    assert rows[0]["id"] == ep["id"] and rows[0]["request"] == "lunch?"
    kinds = [s["kind"] for s in office.get(f"/api/obs/trace/{ep['id']}").json()]
    assert kinds[0] == "episode" and kinds.count("model_call") == 2
    (tmp_path / "x-s0.json").write_text(json.dumps({"model": "m", "summary": {"all": {"n": 2, "passed": 1},
                                                                             "families": {"away": {"n": 2, "passed": 1}}}}))
    assert office.get("/api/obs/evals").json()[0]["families"] == {"away": "1/2"}


@pytest.mark.parametrize("body", [{"request": " "}])
def test_bad_starts_are_refused(store, body):
    a, _ = app(store, [])
    assert signed_in(a, "office").post("/api/episodes", json=body).status_code == 422
