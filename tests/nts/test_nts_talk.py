import json
import random

import pytest

from nts.agents import COORDINATOR
from nts.data import episodes
from nts.eval import build_suite, run_suite
from nts.eval.scenarios import ambiguous, clarify, negotiate, negotiate_refused
from nts.harness import Harness
from nts.kernel.schemas import Validity
from nts.models import ScriptedModel, call, say
from nts.people import Cast, Requester, Teacher
from nts.protocol import AssistantTurn
from nts.telemetry import JsonlSink, MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry, Reply
from nts.tools.names import FULL_DAY, hour


def last_result(messages):
    return json.loads(next(m.content for m in reversed(messages) if m.role == "tool"))


def blocking_rule(messages):
    """The number check_draft gave the rule another teacher set."""
    return next(r["number"] for r in last_result(messages)["clashing"] if "number" in r)


def on_day(world, session, week):
    """The day a class is on in ``week`` (to pick it out among a course's classes)."""
    return FULL_DAY[world.timetable(week).assignment[session.id].day]


def negotiation_steps(sc, world, *, then):
    inst = world.instance
    a = inst.session_by_id[sc.name.split("-", 1)[1].rsplit("-w", 1)[0]]
    week = int(sc.name.rsplit("-w", 1)[1])
    owner = next(iter(sc.cast.teachers))
    day, slot = sc.setup[0].constraint.when.days[0], sc.setup[0].constraint.when.slots[0]
    return [call("move_class", course=inst.course_title(a.course), kind=a.kind.value, on_day=on_day(world, a, week),
                 to_day=FULL_DAY[day],
                 to_hour=hour(slot), from_week=week, to_week=week),
            call("check_draft"),
            lambda m: call("make_offer", to=inst.faculty_by_id[owner].name, rule=blocking_rule(m),
                           message="Could your class move this one week? A colleague's students need the hour."),
            *then]


def test_the_v1_suite_is_unchanged_and_the_talk_families_come_after(store):
    suite = build_suite(store.world, per_family=1, seed=0)
    assert [s.family for s in suite][-3:] == ["clarify", "negotiate", "negotiate_refused"]
    assert all(s.cast is not None for s in suite[-3:]) and suite[-1].setup


def test_an_agreed_waiver_lets_the_move_through_for_that_week_only(store):
    sc = negotiate(store.world, random.Random(1), 5)
    inst = store.world.instance
    model = ScriptedModel(negotiation_steps(sc, store.world, then=[call("check_draft"), call("submit", message="Moved.")]))
    v = run_suite([sc], model, store, COORDINATOR, Tracer([MemorySink()]))[0]
    assert v.passed, v.failures
    clash = last_result(model.seen[2])
    assert clash["ok"] is False and any(r.get("set_by") == inst.faculty_by_id[next(iter(sc.cast.teachers))].name
                                        for r in clash["clashing"])
    offer = last_result(model.seen[3])
    assert offer["answer"] == "yes" and "except week" in offer["draft"]


def test_a_refused_offer_is_not_made_twice_and_the_request_is_declined(store):
    sc = negotiate_refused(store.world, random.Random(2), 5)
    inst = store.world.instance
    again = lambda m: call("make_offer", to=inst.faculty_by_id[next(iter(sc.cast.teachers))].name,
                           rule=1, message="Please?")
    model = ScriptedModel(negotiation_steps(sc, store.world, then=[again, call("decline", message="They cannot move.")]))
    v = run_suite([sc], model, store, COORDINATOR, Tracer([MemorySink()]))[0]
    assert v.passed, v.failures
    assert last_result(model.seen[3])["answer"] == "no"
    assert last_result(model.seen[4])["error"]["code"] == "already_refused"
    without_asking = run_suite([sc], ScriptedModel([call("decline", message="No.")]), store, COORDINATOR,
                               Tracer([MemorySink()]))[0]
    assert not without_asking.passed and "did not ask" in without_asking.failures[0]


def test_an_offer_must_go_to_the_rule_owner(store):
    sc = negotiate(store.world, random.Random(1), 5)
    inst = store.world.instance
    owner = next(iter(sc.cast.teachers))
    someone = next(f for f in inst.faculty if f.id not in (owner, sc.actor))
    steps = negotiation_steps(sc, store.world, then=[])[:2] + [
        lambda m: call("make_offer", to=someone.name, rule=blocking_rule(m), message="?"),
        call("make_offer", to=someone.name, rule=9, message="?"), say("I could not arrange it.")]
    model = ScriptedModel(steps)
    run_suite([sc], model, store, COORDINATOR, Tracer([MemorySink()]))
    err = last_result(model.seen[3])["error"]
    assert err["code"] == "not_theirs" and err["hint"]["owner"] == inst.faculty_by_id[owner].name
    assert last_result(model.seen[4])["error"]["code"] == "unknown_rule"


def test_clarify_asks_the_requester_then_moves_the_class(store):
    sc = clarify(store.world, random.Random(3), 5)
    inst = store.world.instance
    s = inst.session_by_id[sc.name.removeprefix("clarify-")]
    to_day = sc.cast.requester.answer.split("Any time on ")[1].split()[0]
    steps = [call("ask_requester", question="Which class do you mean?"),
             call("move_class", course=inst.course_title(s.course), kind=s.kind.value, on_day=on_day(store.world, s, 5),
                  to_day=to_day, from_week=5, to_week=5),
             call("check_draft"), call("submit", message="Moved.")]
    model = ScriptedModel(steps)
    v = run_suite([sc], model, store, COORDINATOR, Tracer([MemorySink()]))[0]
    assert v.passed, v.failures
    assert last_result(model.seen[1])["says"].startswith("I mean my")


def test_with_no_one_to_answer_the_episode_waits_and_resumes(store, tmp_path):
    sc = ambiguous(store.world, random.Random(4), 5)
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    ask, look = call("ask_requester", question="Which class?"), call("describe", what="rooms")
    turn = AssistantTurn(tool_calls=ask.tool_calls + look.tool_calls)  # a lookup made in the same turn as the ask
    model = ScriptedModel([turn, say("Thanks, the office will move it.")])
    h = Harness(Registry(ALL_TOOLS), model, tracer)
    out = h.run(COORDINATOR, sc.request, store, sc.actor, sc.week)
    assert out.status == "waiting" and out.pending.text == "Which class?" and len(model.seen) == 1
    assert not sc.judge(out, store.world, None)  # asking is right for an unclear request

    done = h.resume(out, Reply(text="My Monday lecture."))
    assert done.status == "answered" and done.episode_id == out.episode_id
    tools = [m for m in model.seen[1] if m.role == "tool"]
    assert json.loads(tools[0].content)["says"] == "My Monday lecture."
    assert json.loads(tools[1].content)["error"]["code"] == "made_before_answer"
    with pytest.raises(ValueError):
        h.resume(done, Reply(text="again"))
    ep = next(episodes(spans))
    assert ep.label()["status"] == "answered" and len(ep.model_calls) == 2  # the resume's end is the episode's


def test_people_answer_at_once_when_there_are_some(store):
    model = ScriptedModel([call("ask_requester", question="Which week?"),
                           call("message_party", to=store.world.instance.faculty[3].name, message="FYI"),
                           say("ok")])
    cast = Cast(requester=Requester("Week 6."), teachers={store.world.instance.faculty[3].id: Teacher()})
    out = Harness(Registry(ALL_TOOLS), model).run(COORDINATOR, "move my class", store,
                                                  store.world.instance.faculty[0].id, 5, people=cast)
    assert out.status == "answered" and [a.kind for a, _ in out.asks] == ["question", "notice"]
    assert last_result(model.seen[2]) == {"sent_to": store.world.instance.faculty[3].name}


def test_a_validity_without_exceptions_serializes_as_before():
    assert Validity().model_dump() == {"from_week": None, "to_week": None}
    v = Validity(from_week=2, except_weeks=[5])
    assert v.covers(4) and not v.covers(5) and v.model_dump()["except_weeks"] == [5]
    assert Validity.model_validate(v.model_dump()) == v
