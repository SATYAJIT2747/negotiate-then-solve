"""A chat message with timetable questions and something else (a request, a rule question): with a v3 model, the
questions answered and the rest after them (``World.read_message``, ``chat_send``'s ``_other_parts``)."""

import pytest
from fastapi.testclient import TestClient

from agents.tool_agent import ChoiceV3, Lookups
from web.app import World, create_web_app

MIXED = "When am I free on Friday, and please move my Data Structures lecture to Thursday."


class FakeV3:
    """Stands in for nts-lookup-v3: answers with ``lookups``, or raises ``error``; counts its calls."""

    def __init__(self, *lookups, error=None):
        self.lookups, self.error, self.calls = lookups, error, 0

    def generate(self, system, prompt, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return Lookups(lookups=list(self.lookups))


@pytest.fixture
def chat_with(monkeypatch):
    """A chat whose questions go to a fake model, as with a lookup adapter on the server (v3 unless said)."""
    from web.fallback import ModelLink

    monkeypatch.delenv("NTS_LOOKUP_READER", raising=False)

    def use(model, adapter="nts-lookup-v3"):
        world = World(parser_mode="offline", seed_history=False, deadline=60)
        world.lookup_adapter = adapter
        world.model_link = ModelLink(lambda: model)
        return TestClient(create_web_app(world)), world

    return use


def menon(world):
    return next(f.id for f in world.instance.faculty if f.name == "Dr. Menon")


def send(c, person, text):
    r = c.post("/api/chat", json={"text": text}, headers={"X-User": person})
    assert r.status_code == 200, r.text
    return r.json()


def free_friday(world):
    return ChoiceV3(tool="free_hours", teachers=["me"], days=["Fri"], week=world.current_week())


def test_a_question_and_a_request_get_an_answer_and_a_card(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    model.lookups = (free_friday(world), ChoiceV3(tool="none", week=world.current_week()))
    before = len(world.store.requests())
    mine, answer, card = send(c, menon(world), MIXED)
    assert mine["text"] == MIXED and model.calls == 1  # request words, yet the model read it
    assert answer["kind"] == "lookup" and answer["call"]["tool"] == "free_hours"
    assert card["kind"] == "request" and card["part"] and card["request_text"] == MIXED  # v3 writes no words
    assert len(world.store.requests()) == before + 1  # sent at once


def test_several_requests_go_in_one_card_with_their_words(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    week = world.current_week()
    model.lookups = (free_friday(world), ChoiceV3(tool="none", request="cancel my Friday lecture", week=week),
                     ChoiceV3(tool="none", request="move my DS practical to Thursday", week=week))
    replies = send(c, menon(world), "When am I free on Friday? Cancel my Friday lecture, and move my DS practical "
                                    "to Thursday.")
    assert [m["kind"] for m in replies] == ["text", "lookup", "request"]
    assert replies[2]["request_text"] == "cancel my Friday lecture; move my DS practical to Thursday"


def test_request_words_beside_the_lookups_still_offer_the_card(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    model.lookups = (ChoiceV3(tool="free_hours", teachers=["Dr. Iyer"], days=["Fri"], week=world.current_week()),)
    replies = send(c, menon(world), "When is Dr. Iyer free on Friday? Please give my Friday class to him.")
    assert [m["kind"] for m in replies] == ["text", "lookup", "request"]  # the model left the request out


def test_a_question_alone_gets_no_card(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    model.lookups = (free_friday(world),)
    assert [m["kind"] for m in send(c, menon(world), "When am I free on Friday?")] == ["text", "lookup"]


def test_a_request_alone_is_a_card_as_before(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    model.lookups = (ChoiceV3(tool="none", week=world.current_week()),)
    text = "Please move my Data Structures lecture from Monday to Tuesday."
    _, card = send(c, menon(world), text)
    assert card["kind"] == "request" and card["request_text"] == text and not card.get("part")


def test_a_question_and_a_rule_question_get_both_answers(chat_with):
    model = FakeV3()
    c, world = chat_with(model)
    model.lookups = (free_friday(world), ChoiceV3(tool="none", week=world.current_week()))
    replies = send(c, menon(world), "When am I free on Friday, and how many online classes are allowed in a week?")
    assert [m["kind"] for m in replies] == ["text", "lookup", "rule"] and replies[2]["rules"]


def test_v3_2s_parts_are_read_in_their_own_words(chat_with):
    model = FakeV3()
    c, world = chat_with(model, adapter="nts-lookup-v3_2")
    week = world.current_week()
    model.lookups = (free_friday(world), ChoiceV3(tool="none", request="how many online classes are allowed in a week", week=week),
                     ChoiceV3(tool="none", request="please move my Data Structures lecture to Thursday", week=week))
    replies = send(c, menon(world), "When am I free on Friday, how many online classes are allowed in a week, and "
                                    "please move my Data Structures lecture to Thursday?")
    assert [m["kind"] for m in replies] == ["text", "lookup", "rule", "request"]
    assert "P-ONLINE" in [x["id"] for x in replies[2]["rules"]]  # the rule question alone, not the whole message
    assert replies[3]["request_text"] == "please move my Data Structures lecture to Thursday"


def test_with_the_model_down_request_words_make_it_a_request(chat_with):
    c, world = chat_with(FakeV3(error=ConnectionError("server gone")))
    _, card = send(c, menon(world), MIXED)  # the regex must not read it as a question
    assert card["kind"] == "request" and card["request_text"] == MIXED


def test_before_v3_request_words_never_reach_the_model(chat_with):
    model = FakeV3(error=AssertionError("the model must not be asked"))
    _, world = chat_with(model, adapter="nts-lookup-v2")
    assert world.read_message(MIXED, menon(world)) == (None, "request words", [])
    assert model.calls == 0
