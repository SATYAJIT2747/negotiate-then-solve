"""Requests made with clicks (POST /api/requests/form): the intent is given, so no parser reads them, even offline."""

import pytest as _pytest

pytestmark = _pytest.mark.slow  # real solves in background threads

import pytest
from fastapi.testclient import TestClient

from web.app import COORDINATOR, World, create_web_app

from test_web import as_, wait_for

DONE = {"awaiting_approval", "published", "escalated", "denied", "refused", "clarification_requested", "answered", "forwarded"}


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def own_class(c, person, week):
    entries = c.get(f"/api/timetable?week={week}", headers=as_(person)).json()["entries"]
    return entries, next(e for e in entries if e["faculty"] == person and e["kind"] == "lecture")


def test_cancel_with_clicks_is_not_misread(client):
    c, world = client
    week = c.get("/api/calendar", headers=as_("F-104")).json()["current_week"]
    _, e = own_class(c, "F-104", week)
    text = f"Please cancel my {e['title']} lecture in week {week} only."
    rid = c.post("/api/requests/form", headers=as_("F-104"),
                 json={"text": text, "intent": {"kind": "cancel", "which": {"session": e["session"]}, "period": {"weeks": [week]}}}).json()["id"]
    d = wait_for(c, rid, DONE)
    assert d["route"].startswith("form+intents") and d["text"] == text
    assert d["status"] not in ("clarification_requested", "refused"), d["reply"]
    assert world.orch.cases[rid].compiled.cancels[0].session == e["session"]


def test_move_with_clicks_goes_where_it_was_asked(client):
    c, _ = client
    week = c.get("/api/calendar", headers=as_("F-101")).json()["current_week"]
    entries, e = own_class(c, "F-101", week)
    taken = {(o["day"], s) for o in entries if o["faculty"] == "F-101" or set(o["groups"]) & set(e["groups"])
             for s in range(o["slot"], o["slot"] + o["duration"])}
    day, slot = next((d, s) for d in ("Mon", "Tue", "Wed", "Thu", "Fri") for s in (0, 1, 2, 5, 6)
                     if (d, s) not in taken and (d, s) != (e["day"], e["slot"]))
    rid = c.post("/api/requests/form", headers=as_("F-101"), json={
        "text": f"Please move my {e['title']} lecture to {day} at {9 + slot}:00, in week {week} only; it has to be then.",
        "intent": {"kind": "move", "which": {"session": e["session"]}, "to": {"days": [day], "at": [f"{9 + slot:02d}:00"]},
                   "period": {"weeks": [week]}, "hard": True}}).json()["id"]
    d = wait_for(c, rid, DONE)
    assert d["route"].startswith("form+intents")
    con = d["parse"]["constraints"][0]
    assert con["type"] == "prefer" and con["when"] == {"days": [day], "slots": [slot], "weeks": [week]}
    if d["status"] == "awaiting_approval":
        moved = next(r for r in d["proposal"]["diff"] if r["session"] == e["session"])
        assert (moved["after"]["day"], moved["after"]["slot"]) == (day, slot)


def test_a_request_made_with_clicks_is_checked_by_the_rules_not_the_model(client):
    c, world = client

    class Model:  # stands in for the model's policy check: it must not be asked
        asked = []

        def review(self, text, parse=None):
            self.asked.append(text)
            return world.rule_policy.review(text, parse)

    week = c.get("/api/calendar", headers=as_("F-104")).json()["current_week"]
    entries, e = own_class(c, "F-104", week)
    taken = {(o["day"], s) for o in entries if o["faculty"] == "F-104" or set(o["groups"]) & set(e["groups"])
             for s in range(o["slot"], o["slot"] + o["duration"])}
    day, slot = next((d, s) for d in ("Mon", "Tue", "Wed", "Thu", "Fri") for s in (0, 1, 2, 5, 6)
                     if (d, s) not in taken and (d, s) != (e["day"], e["slot"]))
    model, world.orch.policy = world.orch.policy, Model()
    try:
        rid = c.post("/api/requests/form", headers=as_("F-104"), json={
            "text": f"Please move my {e['title']} lecture to {day} at {9 + slot}:00, in week {week} only.",
            "intent": {"kind": "move", "which": {"session": e["session"]},
                       "to": {"days": [day], "at": [f"{9 + slot:02d}:00"]}, "period": {"weeks": [week]}}}).json()["id"]
        d = wait_for(c, rid, DONE)
    finally:
        world.orch.policy = model
    assert d["route"].startswith("form+intents") and Model.asked == []
    assert world.orch.cases[rid].policy is not None  # checked all the same, by the rules


def test_form_only_for_your_own_classes(client):
    c, _ = client
    week = c.get("/api/calendar", headers=as_("F-101")).json()["current_week"]
    _, e = own_class(c, "F-101", week)
    cancel = {"kind": "cancel", "which": {"session": e["session"]}, "period": {"weeks": [week]}}
    assert c.post("/api/requests/form", json={"text": "x", "intent": cancel}, headers=as_("F-102")).status_code == 403
    assert c.post("/api/requests/form", json={"text": "x", "intent": {"kind": "avoid", "time": {"days": ["Fri"]}}},
                  headers=as_("F-101")).status_code == 422
    assert c.post("/api/requests/form", json={"text": "x", "intent": {"kind": "cancel"}}, headers=as_("F-101")).status_code == 422
    assert c.post("/api/requests/form", json={"text": "x", "intent": cancel}, headers=as_(COORDINATOR)).status_code == 403


def test_form_refuses_a_week_that_has_passed(client):
    c, _ = client
    week = c.get("/api/calendar", headers=as_("F-101")).json()["current_week"]
    _, e = own_class(c, "F-101", week)
    if week > 1:
        r = c.post("/api/requests/form", headers=as_("F-101"),
                   json={"text": "x", "intent": {"kind": "cancel", "which": {"session": e["session"]}, "period": {"weeks": [week - 1]}}})
        assert r.status_code == 422 and "has passed" in r.json()["detail"]


# -- a range of dates (the range builder): several classes in one request ------------------------------------


def own_classes(c, person):
    """The person's classes in the weeks from now, from the calendar: (week, class) pairs."""
    cal = c.get("/api/calendar", headers=as_(person)).json()
    now = cal["current_week"]
    return now, [(w["week"], x) for w in cal["weeks"] if w["week"] >= now for x in w["classes"] if not x["cancelled"]]


def test_several_classes_are_cancelled_in_one_request(client):
    c, world = client
    now, classes = own_classes(c, "F-103")
    picked = [x for x in classes if x[0] in (now, now + 1)][:3]
    intents = [{"kind": "cancel", "which": {"session": x["session"]}, "period": {"weeks": [w]}} for w, x in picked]
    rid = c.post("/api/requests/form", headers=as_("F-103"),
                 json={"text": "Please cancel these classes.", "intents": intents, "makeup": True}).json()["id"]
    d = wait_for(c, rid, DONE)
    assert d["route"].startswith("form+intents") and d["status"] == "forwarded"
    assert len(world.orch.cases[rid].compiled.cancels) == len(picked)
    assert world.store.request(rid).makeup is True


def test_classes_of_one_week_are_moved_with_the_sender_away_then(client):
    c, world = client
    now, classes = own_classes(c, "F-105")
    picked = [x for w, x in classes if w == now][:2]
    intents = [{"kind": "unavailable", "time": {"days": [x["day"]], "at": [f"{9 + x['slot']:02d}:00"]},
                "period": {"weeks": [now]}} for x in picked]
    rid = c.post("/api/requests/form", headers=as_("F-105"),
                 json={"text": "I am away; please move these classes.", "intents": intents}).json()["id"]
    d = wait_for(c, rid, DONE)
    assert d["route"].startswith("form+intents") and d["status"] not in ("clarification_requested", "refused"), d["reply"]
    assert world.store.request(rid).makeup is True  # moved, not cancelled


def test_a_range_is_checked(client):
    c, world = client
    now, classes = own_classes(c, "F-106")
    one, other = next(x for w, x in classes if w == now), next(x for w, x in classes if w == now + 1)
    away = lambda x, w: {"kind": "unavailable", "time": {"days": [x["day"]]}, "period": {"weeks": [w]}}  # noqa: E731
    post = lambda intents: c.post("/api/requests/form", headers=as_("F-106"), json={"text": "x", "intents": intents})  # noqa: E731
    assert post([away(one, now), away(other, now + 1)]).status_code == 422  # moved one week per request
    cancel = lambda x, w: {"kind": "cancel", "which": {"session": x["session"]}, "period": {"weeks": [w]}}  # noqa: E731
    assert post([cancel(one, now), away(other, now)]).status_code == 422  # cancelled or moved, not both
    assert c.post("/api/settings", json={"max_range_weeks": 1}, headers=as_("F-106")).status_code == 403
    assert c.post("/api/settings", json={"max_range_weeks": 1}, headers=as_(COORDINATOR)).status_code == 200
    try:
        assert c.get("/api/settings", headers=as_("F-106")).json() == {"max_range_weeks": 1}
        r = post([cancel(one, now), cancel(other, now + 1)])
        assert r.status_code == 422 and "at most 1 week." in r.json()["detail"]
    finally:
        world.store.set_setting("max_range_weeks", 4)


def test_demo_department_keeps_its_own_day(client, monkeypatch):
    """The demo does not follow the real date: it stays on the Monday of DEMO_WEEK, so the weeks its
    history and examples name (6 onwards) are never "passed", whenever the demo is given."""
    from datetime import date, timedelta

    from web.world import DEMO_WEEK

    c, world = client
    monkeypatch.delenv("NTS_DEMO_WEEK", raising=False)
    cal = c.get("/api/calendar", headers=as_("F-101")).json()
    assert cal["current_week"] == DEMO_WEEK
    monday = date.fromisoformat(cal["weeks"][DEMO_WEEK - 1]["monday"])
    assert cal["today"] == monday.isoformat() and world.today() == monday
    assert world.now().date() == monday and world.now().hour == 8
    monkeypatch.setenv("NTS_DEMO_WEEK", "9")
    assert world.current_week() == 9 and world.today() == monday + timedelta(weeks=4)
    monkeypatch.setenv("NTS_DEMO_WEEK", "5")
    _, e = own_class(c, "F-101", 7)
    r = c.post("/api/requests/form", headers=as_("F-101"),
               json={"text": "x", "intent": {"kind": "cancel", "which": {"session": e["session"]}, "period": {"weeks": [7]}}})
    assert r.status_code == 200, r.text
    d = wait_for(c, r.json()["id"], DONE)
    assert "has passed" not in (d.get("reply") or "")


def test_a_published_cancellation_reaches_the_sections_chat(client):
    """Class representatives have no inbox: a change that reaches them is a "Timetable change" card in their
    chat, worded for them (the class by its time, why it is not held); the sender gets none (their request's card)."""
    from web.app import COORDINATOR

    c, world = client
    week = c.get("/api/calendar", headers=as_("F-106")).json()["current_week"] + 1
    _, e = own_class(c, "F-106", week)
    rid = c.post("/api/requests/form", headers=as_("F-106"), json={
        "text": f"Please cancel my {e['title']} lecture in week {week}.", "makeup": False,
        "intent": {"kind": "cancel", "which": {"session": e["session"]}, "period": {"weeks": [week]}}}).json()["id"]
    d = wait_for(c, rid, DONE)
    if d["status"] != "awaiting_approval":
        pytest.skip(f"not waiting for approval here: {d['status']}")
    r = c.post(f"/api/approvals/{rid}/approve", headers=as_(COORDINATOR), json={})
    assert r.status_code == 200, r.text
    rep = f"ST-{e['groups'][0]}"
    cards = [m for m in c.get("/api/chat", headers=as_(rep)).json() if m["kind"] == "change" and m["case"] == rid]
    assert len(cards) == 1 and cards[0]["weeks"] == [week], cards
    assert e["title"] in cards[0]["text"] and "(C-" not in cards[0]["text"]
    assert not [m for m in c.get("/api/chat", headers=as_("F-106")).json() if m["kind"] == "change" and m["case"] == rid]
    tt = c.get(f"/api/timetable?week={week}", headers=as_(rep)).json()
    held = next(x for x in tt["cancelled"] if x["session"] == e["session"])
    assert held["day"] == e["day"] and held["slot"] == e["slot"] and held["why"]
