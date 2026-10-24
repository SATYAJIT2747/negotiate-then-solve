import json
from dataclasses import replace

from nts.agents import READER
from nts.harness import LAST_TURN, Budgets, Harness, fit
from nts.models import ReplayModel, ScriptedModel, call, say
from nts.protocol import AssistantTurn, Message
from nts.telemetry import JsonlSink, MemorySink, SqliteSink, Tracer, load_trace
from nts.tools import READ_TOOLS, Registry


def harness(steps, *sinks):
    model = ScriptedModel(steps)
    return Harness(Registry(READ_TOOLS), model, Tracer(sinks)), model


def tool_results(messages):
    return [json.loads(m.content) for m in messages if m.role == "tool"]


def test_a_question_answered_with_two_lookups(store):
    t = store.world.instance.faculty[2]
    h, model = harness([call("find", text="iyer"), call("classes_of", teacher="Dr. Iyer", days=["Tue"]),
                        say("On Tuesday you teach ...")])
    before = store.version
    out = h.run(READER, "what do I teach on Tuesday?", store, actor=t.id, week=5)
    assert (out.status, out.turns, out.tool_calls, out.refused) == ("answered", 3, 2, 0)
    assert out.text.startswith("On Tuesday")
    assert [m.role for m in out.transcript] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert f"The person asking: {t.name}. It is week 5 of 16." in out.transcript[0].content
    # the model was shown each result, answering its call by id
    third = model.seen[2]
    assert third[-1].role == "tool" and third[-1].call_id == third[-2].tool_calls[0].id
    assert "classes" in tool_results(third)[-1]
    assert store.version == before and out.draft.changes == []


def test_a_refused_name_is_fed_back_and_fixed(store):
    h, model = harness([call("classes_of", teacher="Dr. Iyar"), call("classes_of", teacher="Dr. Iyer"), say("ok")])
    out = h.run(READER, "Iyar's classes?", store, actor="coordinator", week=5)
    assert out.status == "answered" and out.refused == 1
    error = tool_results(model.seen[1])[-1]["error"]
    assert error["code"] == "unknown_name" and "Dr. Iyer" in error["hint"]["did_you_mean"]


def test_repeated_calls_are_not_rerun_and_end_as_stuck(store):
    same = call("describe", what="rooms").tool_calls[0]
    steps = [AssistantTurn(tool_calls=[same.model_copy(update={"id": f"c{i}"})]) for i in range(5)]
    h, _ = harness(steps)
    agent = replace(READER, budgets=Budgets(max_repeats=2))
    out = h.run(agent, "rooms?", store, actor="coordinator")
    assert out.status == "stuck" and out.tool_calls == 3
    assert [r.ok for r in out.calls] == [True, False, False]
    assert out.calls[1].error.code == "repeated_call"


def test_the_last_turn_offers_no_tools(store):
    h, model = harness([call("describe", what="rooms"), call("describe", what="sections"), say("done")])
    agent = replace(READER, budgets=Budgets(max_turns=3))
    out = h.run(agent, "rooms and sections?", store, actor="coordinator")
    assert out.status == "answered"
    assert model.tools[0] and model.tools[1] and model.tools[2] == []
    assert model.seen[2][-1] == Message(role="user", content=LAST_TURN)
    assert all(m.content != LAST_TURN for m in out.transcript)  # the note is sent, not kept

    h, _ = harness([call("describe", what="rooms"), call("describe", what="sections")])
    out = h.run(replace(READER, budgets=Budgets(max_turns=2)), "x", store, actor="coordinator")
    assert out.status == "budget_exhausted" and "last turn" in out.detail


def test_calls_past_the_per_turn_limit_are_refused(store):
    many = AssistantTurn(tool_calls=[call("describe", what=w).tool_calls[0] for w in ("rooms", "sections", "teachers")])
    h, _ = harness([many, say("ok")])
    out = h.run(replace(READER, budgets=Budgets(max_calls_per_turn=2)), "all?", store, actor="coordinator")
    assert [r.ok for r in out.calls] == [True, True, False] and out.calls[2].error.code == "too_many_calls"


def test_tool_call_budget(store):
    steps = [call("describe", what=w) for w in ("rooms", "sections", "teachers", "courses")]
    h, _ = harness(steps)
    out = h.run(replace(READER, budgets=Budgets(max_tool_calls=2)), "x", store, actor="coordinator")
    assert out.status == "budget_exhausted" and out.detail == "2 tool calls" and out.turns == 2


def test_a_failing_model_ends_the_episode_and_is_recorded(store):
    def down(messages):
        raise ConnectionError("server down")

    sink = MemorySink()
    h, _ = harness([down], sink)
    out = h.run(READER, "x", store, actor="coordinator")
    assert out.status == "model_error" and "server down" in out.detail
    mc = next(s for s in sink.spans if s.kind == "model_call")
    assert mc.status == "error" and mc.attrs["error"]["type"] == "ConnectionError"
    ep = next(s for s in sink.spans if s.kind == "episode")
    assert ep.status == "ok" and ep.attrs["status"] == "model_error"


def test_spans_nest_and_hold_what_training_needs(store):
    sink = MemorySink()
    h, _ = harness([call("describe", what="rooms"), say("Four rooms.")], sink)
    out = h.run(READER, "which rooms?", store, actor="coordinator", week=3)
    spans = sink.trace(out.episode_id)
    by_kind = {k: [s for s in spans if s.kind == k] for k in ("episode", "turn", "model_call", "tool_call")}
    assert [len(v) for v in by_kind.values()] == [1, 2, 2, 1]
    ep = by_kind["episode"][0]
    assert ep.parent_id is None and all(t.parent_id == ep.span_id for t in by_kind["turn"])
    turn1 = by_kind["turn"][0]
    assert {s.parent_id for s in spans if s.kind in ("model_call", "tool_call") and s.seq < by_kind["turn"][1].seq} \
        == {turn1.span_id}
    assert ep.attrs["status"] == "answered" and ep.attrs["state_version"] == store.version
    assert ep.attrs["agent_version"] == READER.version(ep.attrs["tools"])
    first = by_kind["model_call"][0]
    assert first.attrs["messages"][1] == {"role": "user", "content": "which rooms?", "tool_calls": [],
                                          "call_id": None, "name": None}
    assert first.attrs["response"]["tool_calls"][0]["name"] == "describe"
    assert by_kind["tool_call"][0].attrs["result"]["ok"] is True


def test_jsonl_and_sqlite_record_the_same_episode(store, tmp_path):
    spans_dir, db = tmp_path / "spans", tmp_path / "index.sqlite"
    h, _ = harness([call("classes_of", teacher="Dr. Iyar"), say("Did you mean Dr. Iyer?")],
                   JsonlSink(spans_dir), SqliteSink(db))
    out = h.run(READER, "Iyar?", store, actor="coordinator")
    trace = load_trace(spans_dir, out.episode_id)
    assert [s.kind for s in trace] == ["episode", "turn", "model_call", "tool_call", "turn", "model_call"]
    rows = SqliteSink(db).query("SELECT json_extract(attrs, '$.result.error.code') FROM spans "
                                "WHERE trace_id = ? AND kind = 'tool_call'", (out.episode_id,))
    assert rows == [("unknown_name",)]


def test_a_recorded_episode_replays_exactly(store, tmp_path):
    spans_dir = tmp_path / "spans"
    h, _ = harness([call("find", text="lab"), call("free_rooms", day="Mon", hour="10 am", kind="lab"),
                    say("Lab 1 is free.")], JsonlSink(spans_dir))
    first = h.run(READER, "a free lab on Monday at 10?", store, actor="coordinator", week=4)

    model = ReplayModel.from_spans(load_trace(spans_dir, first.episode_id))
    again = Harness(Registry(READ_TOOLS), model).run(READER, "a free lab on Monday at 10?", store,
                                                     actor="coordinator", week=4, episode_id=first.episode_id)
    assert again.status == "answered" and again.transcript == first.transcript

    drift = ReplayModel.from_spans(load_trace(spans_dir, first.episode_id))  # the same calls, another week
    moved = Harness(Registry(READ_TOOLS), drift).run(READER, "a free lab on Monday at 10?", store,
                                                     actor="coordinator", week=9)
    assert moved.status == "model_error" and "ReplayDiverged" in moved.detail


def test_fit_trims_old_results_first():
    big = "x" * 4000
    msgs = [Message(role="system", content="s"), Message(role="user", content="q")]
    for i in range(4):
        msgs += [Message(role="assistant", tool_calls=[call("find", text=str(i)).tool_calls[0]]),
                 Message(role="tool", content=big, call_id="c", name="find")]
    out = fit(msgs, budget=2500, keep_recent=1)
    tools = [m.content for m in out if m.role == "tool"]
    assert tools[-1] == big and "omitted" in tools[0]
    assert out[:2] == msgs[:2] and msgs[3].content == big  # the transcript itself is untouched
    assert fit(msgs, budget=10**6, keep_recent=1) == msgs


def test_tools_outside_the_agents_scopes_are_not_offered_or_run(store):
    h, model = harness([call("describe", what="rooms"), say("I cannot look that up.")])
    out = h.run(replace(READER, scopes=frozenset()), "rooms?", store, actor="coordinator")
    assert model.tools[0] == [] and out.calls[0].error.code == "unknown_tool"
