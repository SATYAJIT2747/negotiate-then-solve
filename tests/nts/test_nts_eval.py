import json
import random

from nts.agents import COORDINATOR
from nts.data import episodes, export, without_errors
from nts.eval import FAMILIES, build_suite, run_suite, summary
from nts.eval.scenarios import away, free_hours_said, hours_in
from nts.harness import Harness, approve, reject
from nts.models import ScriptedModel, call, say
from nts.protocol import Message
from nts.telemetry import JsonlSink, MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry
from nts.tools.names import FULL_DAY


def away_scenario(store, seed=3):
    """An 'away' scenario and what it is about: its name is away-<teacher id>-<day>-w<week>."""
    sc = away(store.world, random.Random(seed), 5)
    _, f, n, day, w = sc.name.split("-")
    return sc, f"{f}-{n}", day, int(w[1:])


def good_away_steps(day, week, mistake=False):
    steps = [call("constrain", rule="unavailable", teacher="Dr. Nobody", days=[FULL_DAY[day]], from_week=week,
                  to_week=week)] if mistake else []
    return [*steps,
            call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]], from_week=week, to_week=week,
                 reason="away"),
            call("check_draft"), call("submit", message="Moved.")]


def test_suites_are_reproducible_and_cover_every_family(store):
    a = build_suite(store.world, per_family=2, seed=4)
    b = build_suite(store.world, per_family=2, seed=4)
    assert [s.request for s in a] == [s.request for s in b]
    assert {s.family for s in a} == set(FAMILIES)
    assert all(s.actor in store.world.instance.faculty_by_id or s.actor == "coordinator" for s in a)


def test_a_right_episode_passes_and_a_wrong_one_fails(store):
    sc, _, day, week = away_scenario(store)
    sink = MemorySink()
    right = run_suite([sc], ScriptedModel(good_away_steps(day, week)), store, COORDINATOR, Tracer([sink]))
    assert right[0].passed, right[0].failures
    wrong = run_suite([sc], ScriptedModel([say("Sure, done!")]), store, COORDINATOR, Tracer([sink]))
    assert not wrong[0].passed and "status answered" in wrong[0].failures[0]
    week_only = run_suite([sc], ScriptedModel([
        call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]]),  # the whole term, not one week
        call("check_draft"), call("submit", message="Moved.")]), store, COORDINATOR, Tracer([sink]))
    assert not week_only[0].passed and any("changed too" in f for f in week_only[0].failures)
    assert store.world.timetable(week) is store.world.timetable(None)  # the suite ran on forks
    kinds = [s.kind for s in sink.spans if s.trace_id == right[0].episode_id]
    assert "approval" in kinds and "evaluation" in kinds
    assert summary(right + wrong)["all"] == {**summary(right + wrong)["all"], "n": 2, "passed": 1}


def test_labels_and_export(store, tmp_path):
    sc, _, day, week = away_scenario(store)
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    run_suite([sc], ScriptedModel(good_away_steps(day, week, mistake=True)), store, COORDINATOR, tracer)
    run_suite([sc], ScriptedModel([say("Done.")]), store, COORDINATOR, tracer)
    eps = list(episodes(spans))
    labels = sorted((e.label()["good"], e.label()["why"][:15]) for e in eps)
    assert labels == [(False, "scenario failed"), (True, "scenario passed")]

    counts = export(eps, tmp_path / "out")
    assert counts["good"] == 1 and counts["bad"] == 1 and counts["sft-train"] + counts["sft-val"] == 1
    rows = [json.loads(x) for f in ("sft-train", "sft-val") for x in (tmp_path / "out" / f"{f}.jsonl").read_text().splitlines()]
    roles = [m["role"] for m in rows[0]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]  # ends on the submit call
    assert "Dr. Nobody" not in json.dumps(rows[0]["messages"])  # the refused turn is left out
    first = rows[0]["messages"][2]["tool_calls"][0]["function"]
    assert first["name"] == "constrain" and isinstance(first["arguments"], dict)
    assert rows[0]["tools"] and rows[0]["meta"]["good"] is True

    dpo = [json.loads(x) for x in (tmp_path / "out" / "dpo.jsonl").read_text().splitlines()]
    assert len(dpo) == 1 and dpo[0]["meta"]["error"] == "unknown_name"
    assert "Dr. Nobody" in json.dumps(dpo[0]["rejected"]) and dpo[0]["chosen"]["tool_calls"][0]["function"]["arguments"][
        "teacher"] == "me"


def test_approval_spans_label_episodes_without_a_scenario(store, tmp_path):
    spans = tmp_path / "spans"
    tracer = Tracer([JsonlSink(spans)])
    h = Harness(Registry(ALL_TOOLS), ScriptedModel([
        call("constrain", rule="no_classes", teacher="me", days=["Mon"], firm=False),
        call("check_draft"), call("submit", message="ok")]), tracer)
    t = store.world.instance.faculty[0].id
    ok = h.run(COORDINATOR, "keep Mondays free if you can", store.fork(), t, 5)
    fork = store.fork()
    assert approve(fork, ok, tracer) is not None
    h.model = ScriptedModel([call("constrain", rule="no_classes", teacher="me", days=["Tue"], firm=False),
                             call("check_draft"), call("submit", message="ok")])
    no = h.run(COORDINATOR, "keep Tuesdays free if you can", store.fork(), t, 5)
    reject(no, tracer, note="exam week")
    labels = {e.episode.attrs["request"]: e.label() for e in episodes(spans)}
    assert labels["keep Mondays free if you can"]["good"] is True
    assert labels["keep Tuesdays free if you can"] == {**labels["keep Tuesdays free if you can"], "good": False,
                                                       "why": "not approved: exam week"}


def test_without_errors_keeps_mixed_turns():
    a = Message(role="assistant", tool_calls=[call("find", text="x").tool_calls[0], call("find", text="y").tool_calls[0]])
    ok = Message(role="tool", content="[]")
    bad = Message(role="tool", content=json.dumps({"error": {"code": "unknown_name"}}))
    assert without_errors([a, bad, ok]) == [a, bad, ok]
    assert without_errors([a, bad, bad]) == []


def test_hours_in_text():
    assert hours_in("free at 10 am, 2pm and 4 p.m., also at noon") == {1, 5, 7, 3}


def test_hours_said_as_free_skip_busy_sentences():
    text = "You are free at:\n* 9 am\n* 11 am\nYou are busy at 1 pm and 4 pm. Your lecture starts at 10 am."
    assert free_hours_said(text) == {0, 2}
