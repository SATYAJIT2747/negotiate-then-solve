"""Which tools a request is offered (nts.harness.pick): a question gets the tools that read, anything else all."""

import pytest

from nts.agents import COORDINATOR_V6
from nts.harness import Harness
from nts.harness.pick import QUESTION_SCOPES, Pick, ToolPicker
from nts.models import ScriptedModel, call, say
from nts.telemetry import MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry, Scope
from nts.tools.names import FULL_DAY


@pytest.fixture(scope="module")
def picker():
    return ToolPicker.trained()


# wordings no training row has (data/questions.jsonl's test rows and nts.eval's question families are held out too)
QUESTIONS = ["Which classes do I have on Thursday?",
             "When am I free on Monday?", "Is Room 101 booked on Tuesday at 2 pm?"]
CHANGES = ["Please move my Tuesday 10 am lecture to Friday.", "I have a conference next week on Monday and Tuesday.",
           "Could you make sure I never have more than two classes on one day?", "Cancel my Thursday lab this week.",
           "I'd rather not teach Machine Learning and Digital Logic lectures on the same day.",
           "Leave at least an hour free between any two of my classes.", "Can a class of mine go on Thursday this week?"]


@pytest.mark.parametrize("text", QUESTIONS)
def test_a_question_is_offered_only_the_tools_that_read(picker, text):
    assert picker(text, teacher=True).scopes == QUESTION_SCOPES


@pytest.mark.parametrize("text", CHANGES)
def test_a_change_is_offered_every_tool(picker, text):
    """Some of these System One calls a question, but never confidently: offering too few tools would stop the
    change, offering too many only costs tokens."""
    assert picker(text, teacher=True).scopes is None


class _Unsure:
    """System One that calls everything a question, with the given confidence."""

    def __init__(self, p):
        self.p = p

    def decide(self, request):
        from language.corpus import ExpectedAction, RequestType
        from language.system_one import Decision

        return Decision(RequestType.QUESTION, self.p, ExpectedAction.ANSWER, self.p, 0.1)


def test_a_question_said_without_confidence_is_offered_every_tool():
    assert ToolPicker(_Unsure(0.69)).__call__("Have I got anything on Wednesday?", True).scopes is None
    assert ToolPicker(_Unsure(0.71)).__call__("Have I got anything on Wednesday?", True).scopes == QUESTION_SCOPES


def _harness(steps, pick):
    model, sink = ScriptedModel(steps), MemorySink()
    return Harness(Registry(ALL_TOOLS), model, Tracer([sink]), picker=lambda text, teacher: pick), model, sink


def test_the_harness_offers_what_the_picker_says_and_records_it(store):
    t = store.world.instance.faculty[0]
    h, model, sink = _harness([say("You teach nothing then.")], Pick(QUESTION_SCOPES, "question", 0.93))
    out = h.run(COORDINATOR_V6, "What do I teach on Monday?", store, t.id, 5)
    assert out.status == "answered"
    offered = {s["function"]["name"] for s in model.tools[0]}
    assert {"classes_of", "free_hours", "rules", "ask_requester", "decline"} <= offered
    assert not offered & {"move_class", "constrain", "cancel_class", "check_draft", "relate_classes"}
    episode = next(s for s in sink.spans if s.kind == "episode")
    assert episode.attrs["picked"] == {"request_type": "question", "p": 0.93,
                                       "scopes": sorted(s.value for s in QUESTION_SCOPES)}


def test_without_a_pick_every_tool_is_offered(store):
    t = store.world.instance.faculty[0]
    h, model, _ = _harness([say("Done.")], Pick(None, "move", 0.6))
    h.run(COORDINATOR_V6, "Please move my Monday class to Friday.", store, t.id, 5)
    names = {s["function"]["name"] for s in model.tools[0]}
    assert {"move_class", "check_draft", "submit", "relate_classes"} <= names


def test_a_tool_left_out_by_mistake_still_runs(store):
    """The pick only shortens the prompt: a change misread as a question can still be made."""
    t = store.world.instance.faculty[0]
    day = store.world.instance.calendar.days[0]
    h, _, sink = _harness([call("constrain", rule="no_classes", teacher="me", days=[FULL_DAY[day]], from_week=5,
                                to_week=5), say("Noted.")], Pick(QUESTION_SCOPES, "question", 0.9))
    h.run(COORDINATOR_V6, f"No classes for me on {FULL_DAY[day]} in week 5.", store, t.id, 5)
    tool = next(s for s in sink.spans if s.kind == "tool_call")
    assert Scope.DRAFT not in QUESTION_SCOPES
    assert tool.name == "constrain" and tool.attrs["ok"], tool.attrs["result"]


@pytest.mark.parametrize("text", ["When am I free on Tuesday? Is any slot free, I can take an extra class",
                                  "which slots are free on monday so I can take a make-up lecture"])
def test_a_question_that_asks_for_a_change_is_offered_every_tool(picker, text):
    assert picker(text, teacher=True).scopes is None


def test_the_whole_conversation_decides_the_tools(store):
    """ "2 pm" alone reads as nothing; after "I want an extra class" it is part of a change."""
    seen = []
    t = store.world.instance.faculty[0]
    model = ScriptedModel([say("ok")])
    h = Harness(Registry(ALL_TOOLS), model, Tracer([MemorySink()]),
                picker=lambda text, teacher: seen.append(text) or Pick(None, "move", 0.5))
    h.run(COORDINATOR_V6, "2 pm", store, t.id, 5, history=[("I want an extra class on Tuesday", "Which time?")])
    assert seen == ["I want an extra class on Tuesday\n2 pm"]
