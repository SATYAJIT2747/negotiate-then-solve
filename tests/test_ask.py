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
    for text, reader in (("Can I move my Friday lecture to Monday?", "request words"),
                         ("Please cancel my class tomorrow.", "request words"), ("What is the wifi password?", "regex")):
        assert ask(c, faculty(world, "Dr. Menon"), text) == {"kind": "not_a_question", "reader": reader}
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


# -- who reads the question: request words, then the model (the lookup adapter), then the regex ---------------


class FakeModel:
    """Stands in for the lookup adapter: answers with ``choice``, or raises ``error``; counts its calls."""

    def __init__(self, choice=None, error=None):
        self.choice, self.error, self.calls = choice, error, 0

    def generate(self, system, prompt, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return self.choice


@pytest.fixture
def with_model(monkeypatch):
    """A world whose questions go to a fake model, as with a lookup adapter on the server."""
    from web.fallback import ModelLink

    world = World(parser_mode="offline", seed_history=False, deadline=60)
    monkeypatch.delenv("NTS_LOOKUP_READER", raising=False)

    def use(model):
        world.lookup_adapter = "nts-lookup-v1"
        world.model_link = ModelLink(lambda: model)
        return world

    return use


def test_the_model_reads_what_the_regex_cannot(with_model):
    from agents.tool_agent import Choice

    model = FakeModel(Choice(tool="free_hours", teachers=["Dr. Iyer"], week=13))
    world = with_model(model)
    call, reader = world.read_question("when's Iyer around this week?", faculty(world, "Dr. Menon"))
    assert reader == "model" and call.tool == "free_hours" and call.args == {"teachers": [faculty(world, "Dr. Iyer")]}


def test_request_words_never_reach_the_model(with_model):
    model = FakeModel(error=AssertionError("the model must not be asked"))
    world = with_model(model)
    assert world.read_question("can I have Friday free?", faculty(world, "Dr. Menon")) == (None, "request words")
    assert model.calls == 0


def test_the_regex_answers_when_the_model_is_down(with_model):
    model = FakeModel(error=ConnectionError("server gone"))
    world = with_model(model)
    call, reader = world.read_question("What are my free slots this week?", faculty(world, "Dr. Menon"))
    assert reader == "regex" and call.tool == "free_hours"
    assert world.model_link.up is False  # marked down: the next questions go straight to the regex


def test_the_regex_answers_when_the_model_cannot_settle_it(with_model):
    from agents.tool_agent import Choice

    model = FakeModel(Choice(tool="free_hours", teachers=["Dr. Nobody"], week=13))  # fails the checks every time
    world = with_model(model)
    call, reader = world.read_question("What are my free slots this week?", faculty(world, "Dr. Menon"))
    assert reader == "regex" and call.confident and model.calls == 3


def test_without_a_lookup_adapter_the_regex_reads(client):
    _, world = client
    assert world.lookup_reader == "regex"
    assert world.read_question("What are my free slots this week?", faculty(world, "Dr. Menon"))[1] == "regex"


def test_signing_in_is_required(client):
    c, _ = client
    assert c.post("/api/ask", json={"text": "my free slots"}).status_code in (401, 422)
    assert c.post("/api/ask", json={"text": "my free slots"}, headers=as_("nobody")).status_code == 401



# -- KNOWN_BUGS.md B6: one dropped connection turned a question into a request -----------------------------------


class ResetOnce(FakeModel):
    """The server resets the first call (WinError 10054), then answers."""

    def generate(self, system, prompt, schema):
        if self.calls == 0:
            self.calls += 1
            raise ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
        return super().generate(system, prompt, schema)


def test_a_reset_connection_is_tried_once_more(with_model):
    from agents.tool_agent import Choice

    model = ResetOnce(Choice(tool="classes_of", teachers=["Dr. Iyer"], days=["Mon", "Tue"], week=13))
    world = with_model(model)
    call, reader = world.read_question("what courses do dr.iyer take ? on monday and tuesday",
                                       faculty(world, "Dr. Menon"))
    assert reader == "model" and call.tool == "classes_of" and model.calls == 2 and world.model_link.up


def test_a_question_is_not_made_a_request_while_the_model_is_away(with_model):
    from web.world import MODEL_AWAY

    world = with_model(FakeModel(error=ConnectionError("server gone")))
    me = faculty(world, "Dr. Menon")
    call, reader = world.read_question("who's around after lunch on wednesday?", me)  # the regex can't read it
    assert reader == "regex" and not call.confident and call.note == MODEL_AWAY
    # the next one finds the model marked down, and is told the same
    assert world.read_question("who's around after lunch on wednesday?", me)[0].note == MODEL_AWAY
    # what is not a question, or is about the rules, still goes on as a request
    assert world.read_question("Dr. Rao will be on leave next week.", me) == (None, "regex")
    assert world.read_question("is a lunch break mandatory under the rules?", me) == (None, "regex")


def test_offline_a_question_the_regex_cannot_read_goes_on_as_before(client):
    _, world = client
    assert world.read_question("What is the wifi password?", faculty(world, "Dr. Menon")) == (None, "regex")
