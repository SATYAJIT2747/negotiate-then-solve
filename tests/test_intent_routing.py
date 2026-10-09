"""The orchestrator with a parser that reads intents: swaps, extra classes, cancellations and
withdrawals are routed by what it read, not by keywords in the message."""

import os

import pytest

from agents.negotiation import Reply
from language.corpus import FULL_DAY, hour
from language.intent_compiler import clock
from language.intents import (
    Avoid,
    Cancel,
    ClassRef,
    ExtraClass,
    IntentOutput,
    Period,
    Swap,
    TimeSpec,
    compile_intents,
)


class IntentStub:
    """Fixed intents for the messages it knows; for any other message, no intents (``compile`` returns
    None) and the rule parser's plain parse, as a model trained on the older format would give."""

    def __init__(self, instance, plain) -> None:
        self.instance, self.plain, self.known = instance, plain, {}
        self.calls = []

    def compile(self, request, timetable=None, current_week=None):
        out = self.known.get(request.raw_text)
        self.calls.append((request.raw_text, current_week))
        if out is None:
            return None
        return compile_intents(self.instance, request, out, timetable=timetable, current_week=current_week)

    def parse(self, request):
        return self.plain.parse(request)


class Fixed:
    def __init__(self, decision: str) -> None:
        self.decision, self.seen = decision, []

    def respond(self, message):
        self.seen.append(message)
        return Reply(decision=self.decision, choice="A" if self.decision == "accept" else None)


@pytest.fixture(scope="module")
def world():
    os.environ.setdefault("NTS_RETRIEVER", "bm25")
    from web.world import World

    w = World(parser_mode="offline", seed_history=False, deadline=2)
    w.orch.system_one = None  # every request goes to the parser under test
    w.orch.parser = IntentStub(w.instance, w.orch.parser)
    w.orch.current_week = lambda: 5
    return w


def submit(world, sender, text, out):
    world.orch.parser.known[text] = out
    return world.orch.submit(world.intake.from_portal(sender, text))


def where(s, tt):
    return {"day": tt[s.id].day, "time": clock(tt[s.id].slot)}


def _pair(world):
    inst, tt = world.instance, world.store.current_version().assignment
    lect = [s for s in inst.sessions if s.kind.value == "lecture" and s.id in tt]
    a = lect[0]
    b = next(s for s in lect if s.faculty != a.faculty and tt[s.id].day != tt[a.id].day)
    return a, b, tt


def test_a_swap_is_routed_by_its_intent_and_resolved_against_the_timetable(world):
    a, b, tt = _pair(world)
    yes = world.orch.responders[b.faculty] = Fixed("accept")
    text = "Could Dr. X and I trade our classes in week 3?"  # no swap word the old routing looks for
    case = submit(world, a.faculty, text, IntentOutput(action="compile", intents=[Swap(
        mine=ClassRef(**where(a, tt)), theirs=ClassRef(owner=b.faculty, **where(b, tt)), period=Period(weeks=[3]))]))
    assert "+intents" in case.route and "+swap" in case.route
    assert case.status.value == "awaiting_approval", case.reply
    assert yes.seen and yes.seen[0].to == b.faculty
    new = case.proposal.assignment
    assert (new[a.id].day, new[a.id].slot) == (tt[b.id].day, tt[b.id].slot) and case.proposal.week == 3


def test_a_swap_that_names_no_classes_asks(world):
    a, _, _ = _pair(world)
    case = submit(world, a.faculty, "Can I swap with a colleague?",
                  IntentOutput(action="compile", intents=[Swap(mine=ClassRef(), theirs=ClassRef())]))
    assert case.status.value == "clarification_requested" and "Who would you like to swap with" in case.reply


def test_an_extra_class_by_intent_uses_the_current_week(world):
    t = next(s for s in world.instance.sessions if s.kind.value == "lecture")
    case = submit(world, t.faculty, "One more session for my students next week, please.",
                  IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course=t.course),
                                                                    period=Period(relative="next_week"))]))
    assert case.route.endswith("+extra") and case.status.value == "awaiting_approval", case.reply
    assert case.proposal.week == 6  # the current week is 5
    assert world.instance.session_by_id[case.extra["template"]].course == t.course


def test_an_extra_class_without_a_week_asks_for_it(world):
    t = next(s for s in world.instance.sessions if s.kind.value == "lecture")
    case = submit(world, t.faculty, "One more session for my students, please.",
                  IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course=t.course))]))
    assert case.status.value == "clarification_requested" and "Which week" in case.reply


def test_a_cancellation_goes_to_the_coordinator_with_the_class_resolved(world):
    a, _, tt = _pair(world)
    case = submit(world, a.faculty, "Please call off that class in week 9.", IntentOutput(action="compile", intents=[
        Cancel(which=ClassRef(**where(a, tt)), period=Period(weeks=[9]))]))
    assert case.status.value == "forwarded" and "+cancel" in case.route
    assert world.instance.course_title(a.course) in case.reply and "week 9" in case.reply
    assert FULL_DAY[tt[a.id].day] in case.reply and hour(tt[a.id].slot) in case.reply


def test_withdrawing_by_message_takes_back_the_one_open_request(world):
    a, _, _ = _pair(world)
    first = submit(world, a.faculty, "I'd like to keep Friday free if possible, at 3 pm.",
                   IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Fri"], at=["15:00"]))]))
    open_ = [c for c in world.orch.cases.values() if c.request.sender_id == a.faculty
             and world.orch.withdrawable(c) is None]
    case = submit(world, a.faculty, "Please ignore my earlier request.", IntentOutput(action="withdraw"))
    assert "+withdraw" in case.route and case.status.value == "answered"
    if len(open_) == 1:
        assert open_[0].status.value == "withdrawn" and open_[0].id in case.reply
    else:  # several open: nothing is withdrawn on a guess
        assert "open requests" in case.reply and first.status.value != "withdrawn"


def test_messages_the_parser_has_no_intents_for_are_parsed_as_before(world):
    case = world.orch.submit(world.intake.from_portal("F-101", "Can two faculty members swap their slots?"))
    assert "+intents" not in case.route and "+swap" not in case.route
    assert world.orch.parser.calls[-1] == ("Can two faculty members swap their slots?", 5)


def test_the_web_fallback_gives_no_intents_for_a_model_without_them():
    from web.fallback import Fallback, ModelLink

    class Old:  # a model trained on the older format: parse only
        def parse(self, request):
            return "plain"

    class Link(ModelLink):
        def __init__(self):
            self.client, self.up = object(), True

        def usable(self):
            return True

    f = Fallback(Link(), lambda c: Old(), backup=Old(), name="parser")
    assert f.compile("request") is None and f.parse("request") == "plain"
