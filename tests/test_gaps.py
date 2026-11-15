"""Gaps the 30-request demo found: each test is one scenario that went wrong, through the real pipeline
(offline parser, so no model is needed)."""

import pytest as _pytest

pytestmark = _pytest.mark.slow  # real solves in background threads

import pytest
from fastapi.testclient import TestClient

from core.schemas import ConstraintType
from web.app import COORDINATOR, World, create_web_app

from test_web import as_, wait_for

DONE = {"awaiting_approval", "published", "escalated", "denied", "refused", "clarification_requested", "answered",
        "forwarded"}


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def send(c, who, text, makeup=False):
    r = c.post("/api/requests", headers=as_(who), json={"text": text, "makeup": makeup})
    assert r.status_code == 200, r.text
    return wait_for(c, r.json()["id"], DONE)


def approve(c, case_id):
    r = c.post(f"/api/approvals/{case_id}/approve", headers=as_(COORDINATOR), json={})
    assert r.status_code == 200, r.text
    return wait_for(c, case_id, {"published"})


def test_a_change_to_an_earlier_wish_narrows_it_instead_of_withdrawing_it(client):
    c, world = client
    first = send(c, "F-105", "I'd prefer no classes before 10 am on Monday and Wednesday, if possible.")
    if first["status"] == "awaiting_approval":
        approve(c, first["id"])
    # a later constraint of theirs on Monday (as a negotiated placement leaves) must not be taken for the wish
    pin = send(c, "F-105", "My Digital Logic practical has to be on Monday at 2 pm.")
    if pin["status"] == "awaiting_approval":
        approve(c, pin["id"])
    # nor a placement pinned when a negotiation settled, which belongs to no request (the demo's N2-C-033-P-T:
    # amending it turned "keep Monday mornings free" into "the practical needs Monday 11 am")
    wish = next(k for k in world.store.constraints() if k.source.request == first["id"])
    pinned = wish.model_copy(deep=True, update={"id": "N2-C-033-P-T", "type": ConstraintType.PREFER,
                                                "source": wish.source.model_copy(update={"request": None})})
    pinned.when = wish.when.model_copy(update={"days": ["Wed"], "slots": [2, 3]})
    world.store.add_constraints([pinned])
    later = send(c, "F-105", "Actually, Wednesday mornings are fine for me now; only Monday mornings need to stay free.")
    assert f"Your earlier request {first['id']} now covers Monday only" in later["reply"], later["reply"]
    assert next(k for k in world.store.constraints() if k.id == "N2-C-033-P-T").when.days == ["Wed"]  # left as it was
    narrowed = [k for k in world.store.constraints() if k.source.request == later["id"]]
    assert [k.when.days for k in narrowed] == [["Mon"]]
    assert not any(k.source.request == first["id"] for k in world.store.constraints())
    assert later["status"] in {"published", "awaiting_approval"}


def test_an_answered_choice_is_reported_as_chosen(client):
    """A choice answered in words ("A works for me") is taken, and the reply says so."""
    import time

    c, world = client
    r = c.post("/api/requests", headers=as_("F-108"), json={
        "text": "I have a medical procedure on Friday of week 11 and won't be able to take my classes that day.",
        "makeup": True})
    cid = r.json()["id"]
    end, item = time.time() + 120, None
    while time.time() < end and item is None:
        item = next((i for i in world.inbox.items.values() if i.case_id == cid and i.reply is None), None)
        d = c.get(f"/api/cases/{cid}", headers=as_(COORDINATOR)).json()
        if d["status"] in DONE and not d["running"]:
            break
        time.sleep(0.5)
    if item is None:
        pytest.skip("this timetable fits the make-up class only one way, so no choice was offered")
    a = c.post(f"/api/inbox/{item.id}/reply-text", headers=as_("F-108"), json={"text": "A works for me.", "preview": False})
    assert a.status_code == 200, a.text
    d = wait_for(c, cid, DONE)
    assert "No time was chosen" not in d["reply"] and "You chose option A" in d["reply"], d["reply"]


def test_a_question_to_one_person_does_not_hold_up_everyone_else(client):
    """While a teacher had a choice of times unanswered, the next requests waited for its deadline."""
    import time

    c, world = client
    r = c.post("/api/requests", headers=as_("F-108"), json={
        "text": "I have a medical procedure on Friday of week 12 and won't be able to take my classes that day.",
        "makeup": True})
    cid = r.json()["id"]
    end, item = time.time() + 120, None
    while time.time() < end and item is None:
        item = next((i for i in world.inbox.items.values() if i.case_id == cid and i.reply is None), None)
        d = c.get(f"/api/cases/{cid}", headers=as_(COORDINATOR)).json()
        if d["status"] in DONE and not d["running"]:
            break
        time.sleep(0.5)
    if item is None:
        pytest.skip("no choice was offered, so nothing waits")
    t = time.time()
    # Rao's own choice of times, if he is given one, is answered by his simulator; Gupta's still waits
    world.autopilot["F-101"], world.autopilot_delay = True, 0
    other = send(c, "F-101", "I'd prefer no classes on Friday afternoons, if possible.")  # while the choice waits
    assert time.time() - t < 45 and other["status"] in DONE, other
    assert c.get(f"/api/cases/{cid}", headers=as_(COORDINATOR)).json()["running"]  # still waiting for its answer
    c.post(f"/api/inbox/{item.id}/reply-text", headers=as_("F-108"), json={"text": "B please.", "preview": False})
    d = wait_for(c, cid, DONE)
    assert d["status"] in {"awaiting_approval", "published"}, d


def test_the_day_a_class_is_named_by_is_not_reported_unused(client):
    """"shift my Information Security lec frm thursday to wed 11am" said Thursday was not used, though it
    picked out the class."""
    c, world = client
    cal = c.get("/api/calendar", headers=as_("F-102")).json()
    week = cal["weeks"][cal["current_week"]]
    lec = next(x for x in week["classes"] if x["title"] == "Information Security" and x["kind"] == "lecture"
               and x["day"] != "Wed")
    day = cal["day_names"][lec["day"]]
    d = send(c, "F-102", f"Please move my Information Security lecture on {day} at {lec['time']} to Wednesday at 11 am.")
    assert f"You mentioned {day}" not in (d["reply"] or ""), d["reply"]


def test_course_shorthand_is_spelled_out(client):
    """"pls shift my DBMS lec" asked which class was meant; "Database Systems" in the same words did not."""
    c, world = client
    d = send(c, "F-108", "pls shift my DBMS lec frm tuesday to wed 11am, it clashes with my phd viva")
    assert "Which class do you mean" not in (d["reply"] or ""), d["reply"]
    assert any(e["kind"] == "spelled_out" and "Database Systems lecture" in e["text"] for e in d["events"]), d["events"]


def test_a_club_gets_the_room_it_named_or_is_told_why_not(client):
    """"Room 103" was not matched (rooms are R-103 inside), so another room was booked, named "R-101"."""
    from language.intents import ClubRef

    c, world = client
    request = world.intake.from_portal("ST-G-02", "a club booking")

    def ask(room):
        ref = ClubRef(club="Robotics Club", activity="club meeting", date=None, day="Wed", weeks=[10], start="15:00",
                      end="17:00", attendees=40, cohorts=[], room=room)
        return world.club_desk(ref, request)

    reply, outcome = ask("Room 103")
    assert "R-1" not in reply and "Room 103" in reply, reply
    reply, outcome = ask("Room 999")
    assert outcome == "rejected" and "no room called Room 999" in reply, reply


def test_the_exam_cell_taking_a_room_is_not_denied_by_a_misread_rule(client, monkeypatch):
    """"Room 103 is needed for mid-semester exams in week 8" was denied because "week 8 is for exams"."""
    from agents.policy import PolicyDecision, Verdict

    c, world = client
    real = world.orch.policy.review

    def misread(text, parse):
        d = real(text, parse)
        d.verdict, d.explanation = Verdict.FORBIDDEN, "week 8 is reserved for mid-semester examinations"
        return d

    monkeypatch.setattr(world.orch.policy, "review", misread)
    d = send(c, "S-EXAM", "Room 103 is closed for mid-semester exams on Monday and Tuesday of week 8.")
    assert d["status"] != "denied", d["reply"]
    t = send(c, "F-103", "Please schedule my Compiler Design lecture at 1 pm on Thursday.")  # others still checked
    assert t["status"] == "denied", t


def test_a_class_outside_hours_goes_to_the_head_of_department(client):
    """"Thursday at 6 pm" was answered "No class starts at 6 pm": P-HOURS lets the head approve it."""
    c, world = client
    d = send(c, "F-104", "Could my Computer Graphics lecture in week 12 be moved to Thursday at 6 pm instead?")
    assert d["status"] == "escalated" and "Head of Department" in d["reply"], (d["status"], d["reply"])
    heads = [p["id"] for p in c.get("/api/personas").json() if p["role"] == "hod"]
    r = c.post(f"/api/cases/{d['id']}/decide", headers=as_(heads[0]), json={"grant": True, "note": "room 103"})
    assert r.status_code == 200, r.text
    done = wait_for(c, d["id"], {"answered"})
    assert "Approved by" in done["reply"], done["reply"]
