"""The front end's API over the real pipeline (offline parser and policy)."""

import time

import pytest as _pytest

pytestmark = _pytest.mark.slow  # starts real solves in background threads (~75 s)

import pytest
from fastapi.testclient import TestClient

from web.app import COORDINATOR, World, create_web_app, views_for


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def as_(person):
    return {"X-User": person}


def wait_for(c, case_id, statuses, user=COORDINATOR, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        d = c.get(f"/api/cases/{case_id}", headers=as_(user)).json()
        if d["status"] in statuses:
            return d
        time.sleep(0.3)
    raise AssertionError(f"{case_id} stuck in {d['status']}")


def test_mock_auth(client):
    c, _ = client
    people = c.get("/api/personas").json()
    assert any(p["id"] == COORDINATOR for p in people) and any(p["role"] == "student" for p in people)
    assert c.post("/api/login", json={"person": "F-104"}).json()["name"] == "Dr. Khan"
    assert c.get("/api/me", headers=as_("nobody")).status_code == 401


def test_request_to_publication(client):
    c, _world = client
    rid = c.post("/api/requests", json={"text": "I'm at a conference in week 7, Tuesday and Wednesday."},
                 headers=as_("F-104")).json()["id"]
    d = wait_for(c, rid, {"awaiting_approval"})
    assert d["routing"]["tau"] > 0 and d["policy"]["obligations"] == ["P-MAKEUP"]
    assert d["parse"]["constraints"][0]["tier"] == 3 and d["proposal"]["week"] == 7
    assert c.get(f"/api/cases/{rid}", headers=as_("F-105")).status_code == 403  # not theirs
    assert c.post(f"/api/approvals/{rid}/approve", headers=as_("F-104")).status_code == 403
    v = c.post(f"/api/approvals/{rid}/approve", headers=as_(COORDINATOR)).json()["version"]
    assert c.get("/api/timetable?week=7", headers=as_("F-104")).json()["version"] == v
    kinds = [e["kind"] for e in c.get(f"/api/events?case={rid}", headers=as_(COORDINATOR)).json()]
    assert kinds[0] == "received" and kinds[-1] == "published"

    # the inbox tells the sender it was approved, and everyone whose classes moved what changed
    feed = c.get("/api/feed", headers=as_("F-104")).json()
    mine = next(i for i in feed if i["id"] == f"req:{rid}")
    assert mine["title"] == "Your request was approved" and f"version {v}" in mine["text"] and mine["unread"]
    told = [p for p in world_notices(_world, rid) if p != "F-104"]
    assert told, "someone else's classes moved"
    for person in told:
        change = next(i for i in c.get("/api/feed", headers=as_(person)).json() if i["id"] == f"change:{rid}")
        assert change["kind"] == "change" and change["unread"] and change["version"] == v
    unread = c.get("/api/overview", headers=as_(told[0])).json()["my_inbox"]
    c.post("/api/feed/seen", json={"items": {change["id"]: change["n"]}}, headers=as_(told[0]))
    assert c.get("/api/overview", headers=as_(told[0])).json()["my_inbox"] == unread - 1
    assert not next(i for i in c.get("/api/feed", headers=as_(told[0])).json() if i["id"] == f"change:{rid}")["unread"]


def world_notices(world, case_id):
    return sorted(world.orch.cases[case_id].notices)


def test_interactive_negotiation_through_the_inbox(client):
    c, world = client
    inst = world.instance
    pract = {s.faculty: s for s in inst.sessions if s.kind.value == "practical" and not s.equipment}
    rao, menon = pract["F-101"], pract["F-102"]
    first = c.post("/api/requests", json={"text": f"My {inst.course_title(rao.course)} practical needs the routers, "
                                                  "and it has to be on Tuesday afternoon."}, headers=as_("F-101")).json()
    wait_for(c, first["id"], {"awaiting_approval"})
    c.post(f"/api/approvals/{first['id']}/approve", headers=as_(COORDINATOR))
    second = c.post("/api/requests", json={"text": f"My {inst.course_title(menon.course)} practical needs routers "
                                                   "too; it must be on Tuesday afternoon."}, headers=as_("F-102")).json()
    d = wait_for(c, second["id"], {"negotiating", "escalated"})
    assert d["status"] == "negotiating"
    end = time.time() + 60
    items = []
    while time.time() < end and not items:
        items = [i for i in c.get("/api/inbox", headers=as_(COORDINATOR)).json()
                 if i["case"] == second["id"] and i["reply"] is None]
        time.sleep(0.3)
    item = items[0]
    msg = item["message"]
    assert msg["offers"] and msg["facts"] and all(cl["supported"] for cl in msg["claims"])
    other = "F-102" if item["to"] == "F-101" else "F-101"
    assert c.post(f"/api/inbox/{item['id']}/reply", json={"decision": "accept", "choice": "A"},
                  headers=as_(other)).status_code == 403
    r = c.post(f"/api/inbox/{item['id']}/reply", json={"decision": "accept", "choice": "A"}, headers=as_(item["to"]))
    assert r.status_code == 200
    d = wait_for(c, second["id"], {"awaiting_approval"})
    assert d["outcome"]["status"] == "agreed" and d["outcome"]["concessions"]
    assert d["proposal"]["diff"]


def test_transparency_endpoints(client):
    c, _ = client
    h = as_("F-104")
    s = c.get("/api/handbook/search?q=1pm lunch", headers=h).json()
    assert s["results"][0]["id"] == "P-LUNCH" and "h13" in s["tokens"]
    t = c.get("/api/transparency", headers=h).json()
    assert len(t["tiers"]) == 6 and t["ladder"][2]["name"] == "Negotiate"
    assert "gini" in c.get("/api/ledger", headers=h).json()
    office = as_(COORDINATOR)
    g = c.get("/api/graph", headers=office).json()
    assert any(e["rel"] == "reports_to" for e in g["edges"])
    o = c.get("/api/observability", headers=office).json()
    assert "stages" in o and "llm" in o
    assert isinstance(c.get("/api/experiments", headers=office).json(), list)
    assert c.get("/api/instance", headers=h).json()["slots"][4]["label"] == "1 pm"
    assert c.post("/api/demo/reset", headers=h).status_code == 403


def test_views_are_scoped_by_role(client):
    c, _ = client
    people = {p["role"]: p["id"] for p in c.get("/api/personas").json()}
    views = {role: set(c.post("/api/login", json={"person": pid}).json()["views"]) for role, pid in people.items()}
    assert {"approvals", "graph", "health", "experiments"} <= views["coordinator"]
    assert "inbox" in views["faculty"] and "approvals" not in views["faculty"] and "health" not in views["faculty"]
    assert "inbox" in views["student"] and "fairness" not in views["student"] and "new" in views["student"]
    assert "new" not in views_for("dean") and "fairness" in views_for("dean")  # no Dean in the demo directory
    student = as_(people["student"])
    for path in ("/api/ledger", "/api/graph", "/api/observability", "/api/experiments"):
        assert c.get(path, headers=student).status_code == 403, path
    assert c.get("/api/graph", headers=as_("F-104")).status_code == 403
    assert "documents" in views["coordinator"] and "documents" not in views["hod"]
    for who in ("F-104", people["hod"], people["student"]):  # pending changes are the coordinator's
        assert c.get("/api/approvals", headers=as_(who)).status_code == 403
        assert c.get("/api/handbook/documents", headers=as_(who)).status_code == 403
    assert c.get("/api/approvals", headers=as_(COORDINATOR)).status_code == 200
    # The activity log shows a student only their own requests.
    rid = c.post("/api/requests", json={"text": "Is it allowed to teach more than three hours in a row?"},
                 headers=student).json()["id"]
    wait_for(c, rid, {"answered", "refused", "forwarded", "clarification_requested", "denied"}, user=people["student"])
    cases = {e.get("case") for e in c.get("/api/events", headers=student).json()}
    assert cases == {rid}
    assert len({e.get("case") for e in c.get("/api/events", headers=as_(COORDINATOR)).json()}) > 1


def test_tiers_are_shown_in_plain_words():
    from web.wording import plain

    assert plain("Lab 3 is unavailable in week 9 (Tier 0, physical).") == "Lab 3 is unavailable in week 9 (can't be changed)."
    assert plain("Dr. Rao needs Lab 2 (Tier 4, operational requirement).") == "Dr. Rao needs Lab 2 (teaching need)."
    assert "Tier" not in plain("Escalation to coordinator: every option needs a Tier 0-2 change.")
    assert plain(None) is None


def test_only_the_timetable_office_sees_versions(client):
    c, _ = client
    assert c.get("/api/versions", headers=as_(COORDINATOR)).status_code == 200
    assert c.get("/api/versions", headers=as_("F-104")).status_code == 403
    assert c.get("/api/timetable?version=1", headers=as_("F-104")).status_code == 403  # earlier or proposed
    assert c.get("/api/timetable?version=1", headers=as_(COORDINATOR)).status_code == 200
    assert c.get("/api/timetable", headers=as_("F-104")).json()["entries"]  # the latest published, for everyone
    assert c.get("/api/versions/diff?a=1&b=1", headers=as_("ST-G-01")).status_code == 403
    assert "history" in views_for("coordinator") and "history" not in views_for("faculty")


def test_calendar_marks_my_class_days_per_week(client):
    c, world = client
    cal = c.get("/api/calendar", headers=as_("F-104")).json()
    assert len(cal["weeks"]) == world.instance.calendar.weeks and 1 <= cal["current_week"] <= len(cal["weeks"])
    first = cal["weeks"][0]
    assert first["monday"] == "2026-07-27" and cal["weeks"][1]["monday"] == "2026-08-03"
    teaching = sorted({world.store.current_version().assignment[s.id].day for s in world.instance.sessions
                       if s.faculty == "F-104"}, key=world.instance.calendar.days.index)
    plain = [w for w in cal["weeks"] if not w["own_changes"]]
    assert plain and all(w["class_days"] == teaching for w in plain)
    assert any(w["own_changes"] for w in cal["weeks"])  # week 7, from the conference absence above
    assert all(w["class_days"] == [] for w in c.get("/api/calendar", headers=as_("S-LAB")).json()["weeks"])


def test_weekly_changes_are_listed_for_the_semester_plan(client):
    c, _ = client
    assert c.get("/api/weekly-changes", headers=as_("F-104")).status_code == 403  # the office's page
    rows = c.get("/api/weekly-changes", headers=as_(COORDINATOR)).json()
    week7 = next(r for r in rows if r["week"] == 7)  # the conference absence published above
    assert week7["version"] and week7["moved"] and week7["sender_name"] == "Dr. Khan"


def test_answering_a_clarification_reruns_the_same_case(client):
    c, _ = client
    h = as_("F-107")
    rid = c.post("/api/requests", json={"text": "I'll be away for a few days soon, please adjust my classes."},
                 headers=h).json()["id"]
    d = wait_for(c, rid, {"clarification_requested"}, user="F-107")
    assert "which" in d["reply"]
    assert c.post(f"/api/cases/{rid}/clarify", json={"text": "x"}, headers=as_("F-101")).status_code == 403
    assert c.post(f"/api/cases/{rid}/clarify", json={"text": "On Wednesday and Thursday in week 10."},
                  headers=h).status_code == 200
    d = wait_for(c, rid, {"awaiting_approval", "published"}, user="F-107")
    con = d["parse"]["constraints"][0]
    assert con["type"] == "unavailable" and con["when"]["days"] == ["Wed", "Thu"] and con["when"]["weeks"] == [10]
    assert any(e["kind"] == "clarified" for e in c.get(f"/api/events?case={rid}", headers=as_(COORDINATOR)).json())
    assert c.post(f"/api/cases/{rid}/clarify", json={"text": "x"}, headers=h).status_code == 409


def test_the_office_answers_a_forwarded_message(client):
    c, _ = client
    rid = c.post("/api/requests", json={"text": "Two of our classes clash on Monday morning."},
                 headers=as_("ST-G-01")).json()["id"]
    d = wait_for(c, rid, {"forwarded"}, user="ST-G-01")
    assert d["forwarded_to"] == "coordinator" and not d["can_handle"]  # the sender cannot answer it
    assert c.get("/api/overview", headers=as_(COORDINATOR)).json()["my_forwarded"] >= 1
    assert c.get(f"/api/cases/{rid}", headers=as_(COORDINATOR)).json()["can_handle"]
    assert c.post(f"/api/cases/{rid}/handle", json={"note": "ok"}, headers=as_("ST-G-01")).status_code == 403
    assert c.post(f"/api/cases/{rid}/handle", json={"note": "  "}, headers=as_(COORDINATOR)).status_code == 422
    d = c.post(f"/api/cases/{rid}/handle", json={"note": "Booked Lab 2 at 3 pm."}, headers=as_(COORDINATOR)).json()
    assert d["status"] == "forwarded" and d["answer"]["note"] == "Booked Lab 2 at 3 pm."
    assert d["reply"].endswith("Booked Lab 2 at 3 pm.")
    mine = next(x for x in c.get("/api/cases", headers=as_("ST-G-01")).json() if x["id"] == rid)
    assert mine["handled"] and not c.get(f"/api/cases/{rid}", headers=as_(COORDINATOR)).json()["can_handle"]
    assert c.post(f"/api/cases/{rid}/handle", json={"note": "again"}, headers=as_(COORDINATOR)).status_code == 409


def test_the_sender_withdraws_a_request_that_changed_nothing(client):
    c, _ = client
    text = "Our lab and lecture clash on Friday afternoon."
    rid = c.post("/api/requests", json={"text": text}, headers=as_("ST-G-02")).json()["id"]
    d = wait_for(c, rid, {"forwarded"}, user="ST-G-02")
    assert d["can_withdraw"] and not c.get(f"/api/cases/{rid}", headers=as_(COORDINATOR)).json()["can_withdraw"]
    assert c.post(f"/api/cases/{rid}/withdraw", headers=as_(COORDINATOR)).status_code == 403
    d = c.post(f"/api/cases/{rid}/withdraw", headers=as_("ST-G-02")).json()
    assert d["status"] == "withdrawn" and not d["forwarded_to"]  # off the office's list
    assert c.post(f"/api/cases/{rid}/withdraw", headers=as_("ST-G-02")).status_code == 409
    again = c.post("/api/requests", json={"text": text}, headers=as_("ST-G-02"))
    assert again.status_code == 200 and again.json()["id"] != rid  # withdrawn: not a duplicate any more


def test_an_extra_class_from_request_to_the_week_timetable(client):
    c, world = client
    rid = c.post("/api/requests", json={"text": "On Thursday in week 13 I want an extra class to be scheduled."},
                 headers=as_("F-105")).json()["id"]
    d = wait_for(c, rid, {"clarification_requested", "awaiting_approval"}, user="F-105")
    if d["status"] == "clarification_requested":  # Dr. Das teaches more than one class
        assert "Which class" in d["reply"]
        title = world.instance.course_title(next(s for s in world.instance.sessions if s.faculty == "F-105").course)
        assert c.post(f"/api/cases/{rid}/clarify", json={"text": title}, headers=as_("F-105")).status_code == 200
        d = wait_for(c, rid, {"awaiting_approval"}, user="F-105")
    assert d["extra"]["placement"]["day"] == "Thu" and d["proposal"]["week"] == 13
    row = next(r for r in d["proposal"]["diff"] if r["extra"])
    assert row["before"] is None and "extra" in row["session_name"]
    assert any(a["id"] == rid for a in c.get("/api/approvals", headers=as_(COORDINATOR)).json())
    c.post(f"/api/approvals/{rid}/approve", headers=as_(COORDINATOR)).raise_for_status()
    tt = c.get("/api/timetable?week=13", headers=as_("F-105")).json()
    assert any(e["extra"] and e["faculty"] == "F-105" for e in tt["entries"])
    week = next(w for w in c.get("/api/calendar", headers=as_("F-105")).json()["weeks"] if w["week"] == 13)
    assert any(x["extra"] for x in week["classes"])


def test_every_case_opens(client):
    """Each case page's JSON is plain (numpy values from routing once made some answer 500)."""
    c, world = client
    office = as_(COORDINATOR)
    for row in c.get("/api/cases?scope=all", headers=office).json():
        assert c.get(f"/api/cases/{row['id']}", headers=office).status_code == 200, row["id"]
    assert all(isinstance(x.routing.get("fast_path", False), bool) for x in world.orch.cases.values())


def test_restore_to_the_demo_state_without_replaying(client):
    """Last in this module: it puts the shared world back to its start."""
    c, world = client
    end = time.time() + 60
    while world.snapshot is None and time.time() < end:
        time.sleep(0.2)
    office = as_(COORDINATOR)
    demo_cases = set(world.snapshot["cases"])  # this world replays no history: its demo state is the start
    rid = c.post("/api/requests", json={"text": "I'd prefer no classes before 10 am on Friday, if possible."},
                 headers=as_("F-105")).json()["id"]
    wait_for(c, rid, {"awaiting_approval", "published"})
    history = c.get("/api/versions", headers=office).json()
    assert not any((v["case"] or "").startswith("carry-") for v in history)  # listed under their change
    assert c.post("/api/demo/reset", headers=as_("F-105")).status_code == 403
    assert c.post("/api/demo/reset", headers=office).json() == {"ok": True}
    assert {x["id"] for x in c.get("/api/cases?scope=all", headers=office).json()} == demo_cases
    assert c.get("/api/timetable", headers=office).json()["version"] == 1  # the bootstrap timetable
    assert c.get(f"/api/cases/{rid}", headers=office).status_code == 404
    again = c.post("/api/requests", json={"text": "I'd prefer no classes before 10 am on Friday, if possible."},
                   headers=as_("F-105"))
    assert again.status_code == 200  # the restored store no longer holds it, so it is not a duplicate
    assert c.post("/api/demo/reset", headers=office).status_code == 200  # and it restores again
