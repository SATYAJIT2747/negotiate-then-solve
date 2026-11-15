"""Memory v2: a request in a message that leans on the answer to the message's own question ("when am I free on
Friday? move my DS lecture there") is filled in from that answer before it is sent, or asked about."""

import pytest
from fastapi.testclient import TestClient

from agents.lookups_v4 import LookupsV4
from agents.memory import Rewrite
from web.app import World, create_web_app

FRIDAY = {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Fri"]}}
THERE = "move my Data Structures lecture there"
# one free span (Dr. Menon's Friday morning in the demo department): memory fills "there" in without asking
FRIDAY_MORNING = {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Fri"], "part": "morning"}}


class Fake:
    """The lookup adapter and the memory adapter in one: answers each from its own queue; keeps the prompts."""

    def __init__(self, lookups: list[list[dict]], memory: list[Rewrite]):
        self.lookups, self.memory, self.prompts = list(lookups), list(memory), []

    def generate(self, system, prompt, schema):
        self.prompts.append(prompt)
        if schema is LookupsV4:
            return LookupsV4.model_validate({"parts": self.lookups.pop(0)})
        return self.memory.pop(0)


@pytest.fixture
def chat_with(monkeypatch):
    from web.fallback import ModelLink

    monkeypatch.delenv("NTS_LOOKUP_READER", raising=False)
    monkeypatch.delenv("NTS_MEMORY", raising=False)

    def use(model, memory=True):
        w = World(parser_mode="offline", seed_history=False, deadline=60)
        w.lookup_adapter = "nts-lookup-v4"
        w.model_link = ModelLink(lambda: model)
        monkeypatch.setattr(World, "memory_on", property(lambda self: memory))
        return TestClient(create_web_app(w)), next(f.id for f in w.instance.faculty if f.name == "Dr. Menon")

    return use


def send(c, me, text):
    r = c.post("/api/chat", json={"text": text}, headers={"X-User": me})
    assert r.status_code == 200, r.text
    return r.json()


def test_there_is_filled_in_from_the_answer(chat_with):
    model = Fake([[FRIDAY_MORNING, {"tool": "request", "text": THERE}]],
                 [Rewrite(message="move my Data Structures lecture to Friday 9–11 am")])
    c, me = chat_with(model)
    _, answer, card = send(c, me, f"When am I free on Friday morning? {THERE}")
    assert answer["kind"] == "lookup" and card["kind"] == "request"
    assert card["request_text"] == "move my Data Structures lecture to Friday 9–11 am" and card["said"] == THERE
    asked = model.prompts[-1]  # memory saw the question alone, its answer, and the request as the newest message
    assert "When am I free on Friday morning?" in asked and answer["answer"] in asked and THERE in asked.split("<message>")[1]


def test_several_times_are_asked_about_and_the_answer_fills_the_request(chat_with):
    back = "Which time do you mean: 9–11 am or 2–4 pm?"
    model = Fake([[FRIDAY, {"tool": "request", "text": THERE}],
                  [{"tool": "request", "text": "move my Data Structures lecture to Friday 2–4 pm"}]],
                 [Rewrite(ask=back), Rewrite(message="move my Data Structures lecture to Friday 2–4 pm")])
    c, me = chat_with(model)
    _, _, question = send(c, me, f"When am I free on Friday? {THERE}")
    assert question["kind"] == "other" and question["text"] == back and question["about"] == THERE  # no card yet
    _, card = send(c, me, "2–4 pm")
    assert card["kind"] == "request" and card["request_text"] == "move my Data Structures lecture to Friday 2–4 pm"
    assert f"About “{THERE}”: {back}" in model.prompts[2]  # memory knew what the answer was about


def test_one_of_several_times_picked_by_memory_is_asked_about_instead(chat_with):
    model = Fake([[{"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Wed"]}},
                   {"tool": "request", "text": THERE}]],
                 [Rewrite(message="move my Data Structures lecture to Wednesday 9 am–12 pm")])
    c, me = chat_with(model)
    _, answer, last = send(c, me, f"When am I free on Wednesday? {THERE}")
    assert " and " in answer["answer"]  # Dr. Menon's Wednesday: several free spans in the demo department
    assert last["kind"] == "other" and last["text"].startswith("Which time do you mean:") and last["about"] == THERE


def test_a_request_that_leans_on_nothing_is_offered_as_said(chat_with):
    model = Fake([[FRIDAY, {"tool": "request", "text": "cancel my Monday lab"}]], [Rewrite(same=True)])
    c, me = chat_with(model)
    *_, card = send(c, me, "When am I free on Friday? cancel my Monday lab")
    assert card["request_text"] == "cancel my Monday lab" and "said" not in card


def test_without_memory_the_request_is_offered_as_said(chat_with):
    model = Fake([[FRIDAY, {"tool": "request", "text": THERE}]], [])
    c, me = chat_with(model, memory=False)
    *_, card = send(c, me, f"When am I free on Friday? {THERE}")
    assert card["request_text"] == THERE
