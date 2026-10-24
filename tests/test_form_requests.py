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
