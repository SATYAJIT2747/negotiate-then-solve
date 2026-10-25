import json
import random

from test_nts_talk import last_result, negotiation_steps

from nts.agents import COORDINATOR
from nts.eval import FAMILIES, HARD_FAMILIES, V3_FAMILIES, build_suite, run_suite
from nts.eval.rewrite import SOFT, facts, keeps, rewrite
from nts.eval.scenarios import hedged_yes, no_reply
from nts.models import ScriptedModel, call, say
from nts.protocol import AssistantTurn
from nts.telemetry import MemorySink, Tracer


def test_the_harder_families_come_after_and_build(store):
    first = build_suite(store.world, per_family=1, seed=0, families=V3_FAMILIES)
    assert [s.name for s in build_suite(store.world, per_family=1, seed=0)][:len(first)] == [s.name for s in first]
    hard = build_suite(store.world, per_family=2, seed=0, families=HARD_FAMILIES)
    assert {s.family for s in hard} == set(HARD_FAMILIES) and list(FAMILIES)[-len(HARD_FAMILIES):] == HARD_FAMILIES
    assert all(s.cast is not None for s in hard if s.family != "away_days")


def test_a_teacher_who_hedges_is_asked_again(store):
    sc = hedged_yes(store.world, random.Random(1), 5)
    steps = negotiation_steps(sc, store.world, then=[])
    owner = store.world.instance.faculty_by_id[next(iter(sc.cast.teachers))].name
    again = call("make_offer", to=owner, rule=1, message="It is for that week only, so my students can attend.")
    model = ScriptedModel([*steps, again, call("check_draft"), call("submit", message="Moved.")])
    v = run_suite([sc], model, store, COORDINATOR, Tracer([MemorySink()]))[0]
    assert v.passed, v.failures
    first = last_result(model.seen[3])
    assert first["answer"] == "undecided" and "make the offer again" in first["next"]  # not a no: asking again is allowed
    assert last_result(model.seen[4])["answer"] == "yes"

    once = run_suite([sc], ScriptedModel([*steps, say("They are not sure.")]), store, COORDINATOR, Tracer())[0]
    assert not once.passed


def test_no_reply_means_ask_then_decline(store):
    sc = no_reply(store.world, random.Random(1), 5)
    model = ScriptedModel(negotiation_steps(sc, store.world, then=[call("decline", message="They did not answer.")]))
    v = run_suite([sc], model, store, COORDINATOR, Tracer())[0]
    assert v.passed, v.failures
    assert last_result(model.seen[3])["answer"] == "no reply"


def test_a_rewrite_must_keep_every_fact(store):
    w = store.world
    inst = w.instance
    course, teacher = next(iter(inst.course_titles.values())), inst.faculty[0].name
    original = f"If possible, move my {course} lecture on Tuesday at 2 pm to Thursday in week 6. {teacher} agrees."
    pats = facts(original, w)
    kept = f"hey, if poss could my {course.lower()} lecture (tue 2pm) go to thurs, wk 6? {teacher} agrees. if possible!"
    assert keeps(kept, pats) == []
    lost = keeps(f"could my {course} lecture on Tue at 2 pm go to Friday in week 6? {teacher} agrees.", pats)
    assert any("thu" in p for p in lost) and SOFT.pattern in lost


class Rewriter:
    """Rewrites every request with ``fn``, as a model would, in one JSON reply per batch."""

    name = "rewriter"

    def __init__(self, fn):
        self.fn = fn

    def chat(self, messages, tools):
        items = json.loads(messages[-1].content)
        return AssistantTurn(text=json.dumps([{"id": x["id"], "text": self.fn(x["request"])} for x in items]))


def test_a_rewrite_that_loses_a_fact_keeps_the_template(store, tmp_path):
    away, wish = build_suite(store.world, per_family=1, seed=0, families=["away", "preference"])
    out = rewrite([away, wish], "train", Rewriter(lambda r: SOFT.sub("", r) + " thx"), store.world,
                  cache=tmp_path / "rw.json")
    assert out[0].request == away.request + " thx"
    assert out[1].request == wish.request  # the wish lost its "if possible"
    again = rewrite([away], "train", Rewriter(lambda r: "something else"), store.world, cache=tmp_path / "rw.json")
    assert again[0].request == away.request + " thx"  # cached: the same rewrite every time
