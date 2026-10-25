from dataclasses import replace

import pytest
from test_nts_state import away, busy_teacher

from nts.agents import COORDINATOR, COORDINATOR_V1
from nts.harness import Budgets, Harness
from nts.models import ScriptedModel, call, say
from nts.protocol import UNPARSED, ToolCall
from nts.state import ConstraintAdded, InvalidChange, TimetableSet
from nts.tools import ALL_TOOLS, Context, Registry, Scope
from nts.tools.names import FULL_DAY, hour


def episode(store, steps, actor, week=5, agent=COORDINATOR):
    model = ScriptedModel(steps)
    return Harness(Registry(ALL_TOOLS), model).run(agent, "a request", store, actor, week), model


def results(out):
    return {r.name: r for r in out.calls}


def test_a_week_off_is_drafted_checked_submitted_and_committed(store):
    t = busy_teacher(store, "Tue")
    inst = store.world.instance
    out, _ = episode(store, [
        call("constrain", rule="unavailable", teacher="me", days=["Tuesday"], from_week=7, to_week=7,
             reason="a conference"),
        call("check_draft"),
        call("submit", message="Your Tuesday classes in week 7 move."),
    ], actor=t)
    assert out.status == "submitted" and out.text == "Your Tuesday classes in week 7 move."
    added = results(out)["constrain"].content
    assert added["change"] == 1 and "not available on Tuesday, all day, week 7 (firm)" in added["added"][0]
    checked = results(out)["check_draft"].content
    assert checked["ok"] and checked["applies_to"] == "week 7" and checked["classes_moved"] >= 1
    assert all(m["class"]["day"] == "Tuesday" or m["class"]["teacher"] != inst.faculty_by_id[t].name
               for m in checked["moves"])

    c = next(ch.constraint for ch in out.draft.changes if isinstance(ch, ConstraintAdded))
    assert c.tier.name == "VERIFIED_UNAVAILABILITY" and c.source.request == "a conference" and c.owner == t
    assert [ch.week for ch in out.draft.changes if isinstance(ch, TimetableSet)] == [7]
    world = store.commit(out.draft)
    assert not any(p.day == "Tue" and inst.session_by_id[s].faculty == t for s, p in world.timetable(7).assignment.items())
    assert world.timetable(8) is world.timetable(None)


def test_submit_needs_a_check_of_this_exact_draft(store):
    t = busy_teacher(store, "Tue")
    out, _ = episode(store, [
        call("submit", message="x"),
        call("constrain", rule="no_classes", teacher="me", days=["Fri"], from_hour="2 pm", firm=False),
        call("submit", message="x"),
        call("check_draft"),
        call("constrain", rule="no_classes", teacher="me", days=["Mon"], to_hour="10 am", firm=False),
        call("submit", message="x"),
        say("I need to check again."),
    ], actor=t)
    codes = [r.error.code if r.error else "ok" for r in out.calls]
    assert codes == ["empty_draft", "ok", "not_checked", "ok", "ok", "not_checked"]
    assert out.status == "answered"
    soft = next(ch.constraint for ch in out.draft.changes if isinstance(ch, ConstraintAdded))
    assert not soft.hard and soft.tier.name == "PREFERENCE" and soft.when.slots == [5, 6, 7]


def test_teachers_change_only_their_own_things(store):
    inst = store.world.instance
    t = busy_teacher(store, "Tue")
    other = next(f for f in inst.faculty if f.id != t)
    out, _ = episode(store, [
        call("constrain", rule="unavailable", teacher=other.name, days=["Mon"]),
        call("constrain", rule="unavailable", room="Lab 2", days=["Mon"]),
        call("constrain", rule="unavailable", section="Section 1", days=["Mon"]),
        say("I can't do that."),
    ], actor=t)
    assert [r.error.code for r in out.calls] == ["not_allowed"] * 3

    out, _ = episode(store, [call("constrain", rule="unavailable", room="Lab 2", days=["Mon"], from_week=3, to_week=3),
                             say("ok")], actor="coordinator")
    c = out.draft.changes[0].constraint
    assert out.calls[0].ok and c.scope.room == "L-2" and c.tier.name == "COMMITMENT"
    assert "Lab 2: closed on Monday" in out.calls[0].content["added"][0]


def test_a_clash_names_the_rules_and_blocks_submit(store):
    t = busy_teacher(store, "Tue")
    out, _ = episode(store, [
        call("constrain", rule="unavailable", teacher="me", from_week=6, to_week=6),  # every day, all day
        call("check_draft"),
        call("submit", message="x"),
        call("retract", change=1),
        call("decline", message="You must teach some time in week 6."),
    ], actor=t)
    r = results(out)
    assert not r["check_draft"].content["ok"]
    clashing = r["check_draft"].content["clashing"]
    assert any(c.get("your_change") == 1 for c in clashing)
    assert out.calls[2].error.code == "check_failed"
    assert out.status == "declined" and out.text.startswith("You must teach")
    assert r["retract"].content == {"retracted": 1, "left": []}


def test_move_one_class(store):
    inst, tt = store.world.instance, store.world.timetable()
    s = next(s for s in inst.sessions if s.duration == 1 and s.kind.value == "lecture")
    p = tt.assignment[s.id]
    title, me = inst.course_title(s.course), s.faculty
    ambiguous, _ = episode(store, [call("move_class", course=title, to_day="Fri"), say("which one?")], actor=me)
    n_mine = sum(x.course == s.course and x.faculty == me for x in inst.sessions)
    if n_mine > 1:
        assert ambiguous.calls[0].error.code == "ambiguous_class" and len(ambiguous.calls[0].error.hint["candidates"]) > 1

    to_day = next(d for d in inst.calendar.days if d != p.day)
    out, _ = episode(store, [
        call("move_class", course=title, kind="lecture", on_day=FULL_DAY[p.day], at_hour=hour(p.slot),
             to_day=to_day, from_week=9, to_week=9),
        call("check_draft"),
        call("submit", message="moved"),
    ], actor=me)
    assert out.status == "submitted", out.calls
    world = store.commit(out.draft)
    assert world.timetable(9).assignment[s.id].day == to_day
    assert world.timetable(10).assignment[s.id] == p


def test_list_and_retract(store):
    t = busy_teacher(store, "Tue")
    out, _ = episode(store, [
        call("constrain", rule="no_classes", teacher="me", days=["Mon"], firm=False),
        call("constrain", rule="no_classes", teacher="me", days=["Wed"], firm=False),
        call("retract", change=1),
        call("retract", change=7),
        call("list_draft"),
        say("ok"),
    ], actor=t)
    r = out.calls
    assert r[3].error.code == "unknown_change" and r[3].error.hint["changes"] == [2]
    assert [x["change"] for x in r[4].content] == [2] and "Wednesday" in r[4].content[0]["rules"][0]


def test_unknown_fields_and_bad_json_are_refused(store):
    reg = Registry(ALL_TOOLS)
    ctx = Context(store.world, "coordinator", 5, store.draft("c"))
    r = reg.run(ToolCall(id="1", name="constrain", arguments={"rule": "unavailable", "teacher": "Dr. Iyer", "day": "Mon"}),
                ctx)
    assert r.error.code == "bad_arguments" and "'day'" in r.error.message and "days" in r.error.hint["fields"]
    r = reg.run(ToolCall(id="2", name="constrain", arguments={UNPARSED: '{"rule": unav'}), ctx)
    assert r.error.code == "bad_json"


def test_the_commit_gate_checks_weeks_a_new_rule_touches(store):
    t = busy_teacher(store, "Tue")
    draft = store.draft(actor=t)
    draft.add(ConstraintAdded(constraint=away(t, "Tue", 7)))  # no week-7 timetable: week 7 runs the semester one
    with pytest.raises(InvalidChange, match="week 7"):
        store.commit(draft)


def test_the_last_turn_offers_only_the_end_tools(store):
    t = busy_teacher(store, "Tue")
    out, model = episode(store, [
        call("constrain", rule="no_classes", teacher="me", days=["Mon"], firm=False),
        call("check_draft"),
        call("submit", message="done"),
    ], actor=t, agent=replace(COORDINATOR, budgets=Budgets(max_turns=3)))
    assert out.status == "submitted"
    assert {t["function"]["name"] for t in model.tools[2]} == {"submit", "decline", "escalate"}
    assert len(model.tools[0]) == len(Registry(ALL_TOOLS).specs(COORDINATOR.scopes))


def test_calls_after_an_end_tool_are_not_run(store):
    from nts.protocol import AssistantTurn

    both = AssistantTurn(tool_calls=[call("escalate", message="for the office").tool_calls[0],
                                     call("describe", what="rooms").tool_calls[0]])
    out, _ = episode(store, [both], actor="coordinator")
    assert out.status == "escalated" and out.tool_calls == 1
    assert out.transcript[-1].role == "tool" and "episode_ended" in out.transcript[-1].content


def test_scopes_of_the_coordinator():
    assert COORDINATOR_V1.scopes == {Scope.READ, Scope.DRAFT, Scope.EVALUATE, Scope.END}
    assert COORDINATOR.scopes == COORDINATOR_V1.scopes | {Scope.TALK}
