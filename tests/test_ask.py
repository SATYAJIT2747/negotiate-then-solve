"""POST /api/ask: timetable questions answered from the timetable, nothing stored; anything else handed back."""

import pytest
from fastapi.testclient import TestClient

from agents.lookup import WeekTimetable
from agents.questions import _spans
from web.app import World, create_web_app


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def as_(person):
    return {"X-User": person}


def faculty(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def ask(c, person, text):
    r = c.post("/api/ask", json={"text": text}, headers=as_(person))
    assert r.status_code == 200, r.text
    return r.json()


def test_free_hours_are_answered_and_returned_for_the_grid(client):
    c, world = client
    menon = faculty(world, "Dr. Menon")
    d = ask(c, menon, "What are my free slots this week?")
    week = WeekTimetable(world.instance, world.store, d["call"]["week"])
    free = week.free_hours(teachers=[menon])
    assert d["kind"] == "timetable" and d["call"]["tool"] == "free_hours"
    assert d["answer"] == f"You are free on {_spans(free)} (week {d['call']['week']})."
    assert [(h["day"], h["slot"]) for h in d["free"]] == free


def test_a_room_at_an_hour_returns_only_that_hour(client):
    c, world = client
    d = ask(c, faculty(world, "Dr. Menon"), "is lab 2 free on thursday at 3 pm")
    assert d["answer"].startswith(("Yes,", "No,")) and all(h == {"day": "Thu", "slot": 6} for h in d["free"])


def test_classes_are_answered_without_free_hours(client):
    c, world = client
    d = ask(c, faculty(world, "Dr. Menon"), "When does Dr. Iyer teach?")
    assert d["kind"] == "timetable" and d["call"]["tool"] == "classes_of" and "free" not in d
    assert d["answer"].startswith("Dr. Iyer teaches in week")


def test_a_request_is_handed_back_and_nothing_is_stored(client):
    c, world = client
    before = len(world.store.requests())
    for text in ("Can I move my Friday lecture to Monday?", "Please cancel my class tomorrow.", "What is the wifi password?"):
        assert ask(c, faculty(world, "Dr. Menon"), text) == {"kind": "not_a_question"}
    ask(c, faculty(world, "Dr. Menon"), "What are my free slots this week?")
    assert len(world.store.requests()) == before  # questions are not requests


def test_an_unknown_name_is_not_guessed(client):
    c, world = client
    d = ask(c, faculty(world, "Dr. Menon"), "Is Dr. Smith free on Monday?")
    assert d["call"]["confident"] is False and d["answer"] == "I don't know a teacher called Smith." and "free" not in d


def test_anyone_signed_in_may_ask_about_a_teachers_free_hours(client):
    c, world = client
    rep = next(i.person for i in world.directory.by_email.values() if i.person.startswith("ST-"))
    d = ask(c, rep, "Is Dr. Iyer free on Monday?")
    assert d["kind"] == "timetable" and d["answer"].startswith("Dr. Iyer ")
    mine = ask(c, rep, "my free slots this week")  # a class representative's "my" is their section
    assert mine["call"]["args"] == {"sections": [rep[3:]]}


def test_signing_in_is_required(client):
    c, _ = client
    assert c.post("/api/ask", json={"text": "my free slots"}).status_code in (401, 422)
    assert c.post("/api/ask", json={"text": "my free slots"}, headers=as_("nobody")).status_code == 401
