import json

from test_nts_eval import away_scenario, good_away_steps

from nts.agents import COORDINATOR
from nts.data import episodes, export, outcome_pairs
from nts.eval import run_suite
from nts.harness import Harness, approve, reject
from nts.models import ScriptedModel, call, say
from nts.telemetry import JsonlSink, Tracer
from nts.tools import ALL_TOOLS, Registry
from nts.tools.names import FULL_DAY


def whole_term(day):
    """The away request drafted for the whole term instead of one week: the scenario fails it."""
    return [call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]], reason="away"),
            call("check_draft"), call("submit", message="Moved.")]


def test_the_draft_most_runs_agree_on_is_kept_and_the_others_become_pairs(store, tmp_path):
    sc, _, day, week = away_scenario(store)
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    script = [*whole_term(day), *good_away_steps(day, week), *good_away_steps(day, week)]  # 1 of 3 wrong, and first
    v = run_suite([sc], ScriptedModel(script), store, COORDINATOR, tracer, best=3)[0]
    assert v.passed, v.failures
    assert v.episode_id.endswith(".2")  # the first of the two that agree

    eps = {e.id: e for e in episodes(spans)}
    assert len(eps) == 3
    sel = {i: e.selections[-1].attrs for i, e in eps.items()}
    assert [s["chosen"] for s in sel.values()].count(True) == 1 and all(s["agree"] == 2 and s["of"] == 3 for s in sel.values())
    labels = {i: e.label() for i, e in eps.items()}
    wrong = next(i for i in eps if "." not in i)
    assert labels[wrong]["good"] is False and labels[wrong]["why"].startswith("not chosen")
    assert labels[v.episode_id]["good"] is True  # judged by the scenario
    assert labels[next(i for i in eps if i.endswith(".3"))]["good"] is None  # the same draft, not chosen: no verdict

    pairs = outcome_pairs(eps.values())
    assert len(pairs) == 1
    p = pairs[0]
    assert [m["role"] for m in p["prompt"]] == ["system", "user"]  # they part at the first turn
    assert p["chosen"]["tool_calls"][0]["function"]["arguments"]["from_week"] == week
    assert "from_week" not in p["rejected"]["tool_calls"][0]["function"]["arguments"]
    assert p["meta"] == {**p["meta"], "episode": v.episode_id, "against": wrong, "source": "outcome"}

    counts = export(eps.values(), tmp_path / "out")
    assert counts["dpo_outcome"] == 1 and counts["bad"] == 1 and counts["sft-train"] + counts["sft-val"] == 1


def test_one_run_is_the_plain_loop(store):
    sc, _, day, week = away_scenario(store)
    v = run_suite([sc], ScriptedModel(good_away_steps(day, week)), store, COORDINATOR, Tracer(), best=1)[0]
    assert v.passed and "." not in v.episode_id


def test_runs_that_break_do_not_outvote_runs_that_end(store):
    sc, _, day, week = away_scenario(store)
    looping = [call("list_draft")] * 3 + [call("list_draft")]  # repeats until stuck
    script = [*looping, *looping, *good_away_steps(day, week)]
    v = run_suite([sc], ScriptedModel(script), store, COORDINATOR, Tracer(), best=3)[0]
    assert v.passed and v.status == "submitted", (v.status, v.failures)


def test_the_office_saying_no_pairs_with_the_office_saying_yes(store, tmp_path):
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    t = store.world.instance.faculty[0].id
    h = Harness(Registry(ALL_TOOLS), ScriptedModel([
        call("constrain", rule="no_classes", teacher="me", days=["Monday"], firm=False),
        call("check_draft"), call("submit", message="ok"),
        call("constrain", rule="unavailable", teacher="me", days=["Monday"]),
        call("check_draft"), call("submit", message="ok")]), tracer)
    yes = h.run(COORDINATOR, "keep Mondays free if you can", store, t, 5)
    no = h.run(COORDINATOR, "keep Mondays free if you can", store, t, 5)
    reject(no, tracer, note="they said if possible: a wish, not a rule")
    assert approve(store, yes, tracer) is not None

    pairs = outcome_pairs(episodes(spans))
    assert len(pairs) == 1 and pairs[0]["meta"]["why"] == "not approved: they said if possible: a wish, not a rule"
    assert pairs[0]["chosen"]["tool_calls"][0]["function"]["arguments"]["firm"] is False
    assert json.dumps(pairs[0]["rejected"]).count('"unavailable"') == 1


def test_different_requests_never_pair(store, tmp_path):
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    t = store.world.instance.faculty[0].id
    h = Harness(Registry(ALL_TOOLS), ScriptedModel([say("You are free then."), say("No.")]), tracer)
    a = h.run(COORDINATOR, "am I free on Monday?", store, t, 5)
    b = h.run(COORDINATOR, "am I free on Tuesday?", store, t, 5)
    for o, ok in ((a, True), (b, False)):
        with tracer.span("evaluation", "q", trace_id=o.episode_id, passed=ok, failures=[]):
            pass
    assert outcome_pairs(episodes(spans)) == []
