"""The chat (/api/chat): one conversation per person; timetable questions answered in it, everything else
made a request that the chat follows with a card. Offline: the regex reads the questions."""

import pytest
from fastapi.testclient import TestClient

from pipeline.store import Store
from web.app import World, create_web_app


@pytest.fixture(scope="module")
def client():
    world = World(parser_mode="offline", seed_history=False, deadline=60)
    return TestClient(create_web_app(world)), world


def faculty(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def send(c, person, text):
    r = c.post("/api/chat", json={"text": text}, headers={"X-User": person})
    assert r.status_code == 200, r.text
    return r.json()


def chat(c, person, after=0):
    return c.get(f"/api/chat?after={after}", headers={"X-User": person}).json()


def test_a_timetable_question_is_answered_in_the_chat(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    before = len(world.store.requests())
    mine, reply = send(c, me, "What are my free slots this week?")
    assert (mine["role"], mine["kind"], mine["text"]) == ("you", "text", "What are my free slots this week?")
    assert (reply["role"], reply["kind"]) == ("assistant", "lookup")
    assert reply["answer"].startswith("You are free on") and reply["call"]["tool"] == "free_hours" and reply["free"]
    assert len(world.store.requests()) == before  # a question is not a request


def test_a_request_becomes_a_request_and_a_card(client):
    c, world = client
    me = faculty(world, "Dr. Menon")
    mine, card = send(c, me, "I am at a conference on Monday of week 15, please cancel my classes that day.")
    assert card["kind"] == "request" and card["reader"] == "request words"
    assert world.store.request(card["case"]).sender_id == me  # the same request the Requests tab lists


def test_a_rule_question_goes_in_as_a_request_the_card_follows(client):
    c, world = client
    _, card = send(c, faculty(world, "Dr. Menon"), "Is it allowed to hold classes on Saturday?")
    assert card["kind"] == "request" and world.store.request(card["case"]) is not None


def test_the_chat_is_kept_in_order_and_read_after_a_number(client):
    c, world = client
    me = faculty(world, "Dr. Iyer")
    first = send(c, me, "What are my free slots this week?")
    second = send(c, me, "what do I teach on wednesday?")
    everything = chat(c, me)
    assert [m["n"] for m in everything] == sorted(m["n"] for m in everything)
    assert everything[-4:] == first + second
    assert chat(c, me, after=first[-1]["n"]) == second


def test_nobody_sees_anyone_elses_chat(client):
    c, world = client
    rao = faculty(world, "Dr. Rao")
    send(c, faculty(world, "Dr. Khan"), "What are my free slots this week?")
    assert all(m.get("text") != "What are my free slots this week?" for m in chat(c, rao))


def test_clearing_the_chat_keeps_the_requests(client):
    c, world = client
    me = faculty(world, "Dr. Das")
    _, card = send(c, me, "Please cancel my classes on Tuesday of week 15, I am away.")
    r = c.delete("/api/chat", headers={"X-User": me})
    assert r.json()["cleared"] == 2 and chat(c, me) == []
    assert world.store.request(card["case"]) is not None


def test_the_chat_is_saved_with_the_store():
    store = Store()
    store.add_chat("F-102", "you", "text", {"text": "hello"})
    copy = Store()
    copy.load(store.db.serialize())
    assert [m["text"] for m in copy.chat("F-102")] == ["hello"]


def test_signing_in_is_required(client):
    c, _ = client
    assert c.get("/api/chat").status_code in (401, 422)
    assert c.post("/api/chat", json={"text": "hi"}, headers={"X-User": "nobody"}).status_code == 401
