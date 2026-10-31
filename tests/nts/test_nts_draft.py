from dataclasses import replace

import pytest
from test_nts_state import away, busy_teacher

from nts.agents import COORDINATOR, COORDINATOR_V1, COORDINATOR_V4
from nts.harness import LAST_TURN, Budgets, Harness
from nts.harness.context import UNCHECKED, WRAP_UP
from nts.models import ScriptedModel, call, say
from nts.protocol import UNPARSED, Message, ToolCall
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
    assert COORDINATOR.scopes == COORDINATOR_V1.scopes | {Scope.TALK, Scope.POLICY}


def test_the_turn_before_the_last_says_so_and_an_unchecked_draft_cannot_be_submitted_last(store):
    t = busy_teacher(store, "Tue")
    agent = replace(COORDINATOR, budgets=replace(COORDINATOR.budgets, max_turns=3))
    model = ScriptedModel([
        call("constrain", rule="unavailable", teacher="me", days=["Tuesday"], from_week=7, to_week=7),
        call("constrain", rule="unavailable", teacher="me", days=["Wednesday"], from_week=7, to_week=7),
        call("escalate", message="Away Tuesday and Wednesday of week 7: not checked yet."),
    ])
    out = Harness(Registry(ALL_TOOLS), model).run(agent, "a request", store, t, 5)
    assert model.seen[1][-1] == Message(role="user", content=WRAP_UP)
    assert model.seen[2][-1] == Message(role="user", content=UNCHECKED)
    assert {x["function"]["name"] for x in model.tools[2]} == {"decline", "escalate"}  # no submit: it would be refused
    assert out.status == "escalated"

    model = ScriptedModel([
        call("constrain", rule="unavailable", teacher="me", days=["Tuesday"], from_week=7, to_week=7),
        call("check_draft"),
        call("submit", message="Your Tuesday classes in week 7 move."),
    ])
    out = Harness(Registry(ALL_TOOLS), model).run(agent, "a request", store, t, 5)
    assert model.seen[2][-1] == Message(role="user", content=LAST_TURN) and out.status == "submitted"


def test_a_move_without_the_hour_the_request_named_says_so(store):
    inst, tt = store.world.instance, store.world.timetable()
    s = next(s for s in inst.sessions if s.duration == 1 and s.kind.value == "lecture")
    p = tt.assignment[s.id]
    to_day = next(d for d in inst.calendar.days if d != p.day)
    move = call("move_class", course=inst.course_title(s.course), kind="lecture", on_day=FULL_DAY[p.day],
                at_hour=hour(p.slot), to_day=to_day, from_week=9, to_week=9)
    for request, noted in ((f"Please move it to {FULL_DAY[to_day]} at 2 pm in week 9.", True),
                           (f"Please move it to {FULL_DAY[to_day]} in week 9; it is at {hour(p.slot)} now.", False)):
        out = Harness(Registry(ALL_TOOLS), ScriptedModel([move, say("ok")])).run(COORDINATOR, request, store, s.faculty, 5)
        result = out.calls[0].content
        assert ("note" in result) == noted, result
        if noted:
            assert f"{FULL_DAY[to_day]} at 2 pm" in result["note"] and "to_hour" in result["note"]


def test_a_move_from_a_week_with_no_end_is_for_that_week_unless_the_request_says_from_then_on(store):
    inst, tt = store.world.instance, store.world.timetable()
    s = next(s for s in inst.sessions if s.duration == 1 and s.kind.value == "lecture")
    p = tt.assignment[s.id]
    to_day = next(d for d in inst.calendar.days if d != p.day)
    move = call("move_class", course=inst.course_title(s.course), kind="lecture", on_day=FULL_DAY[p.day],
                at_hour=hour(p.slot), to_day=to_day, from_week=6)
    for request, weeks in ((f"Please move it to {FULL_DAY[to_day]} in week 6 only.", "week 6 (firm)"),
                           (f"Please move it to {FULL_DAY[to_day]} from week 6 onwards.", "from week 6 (firm)")):
        out = Harness(Registry(ALL_TOOLS), ScriptedModel([move, say("ok")])).run(COORDINATOR, request, store, s.faculty, 5)
        added = out.calls[0].content["added"][0]
        assert added.endswith(weeks) and (weeks.startswith("from") or "from week" not in added), added


def _two_of_a_course(store):
    """A teacher's two one-hour classes of the same course, on different days."""
    inst, tt = store.world.instance, store.world.timetable()
    for a in inst.sessions:
        for b in inst.sessions:
            if (a.id < b.id and a.course == b.course and a.faculty == b.faculty and a.kind == b.kind
                    and a.duration == b.duration == 1 and a.id in tt.assignment and b.id in tt.assignment
                    and tt.assignment[a.id].day != tt.assignment[b.id].day):
                return a, b, tt.assignment[a.id], tt.assignment[b.id]
    pytest.skip("no teacher with two classes of one course")


def test_an_ambiguous_move_says_how_to_pick_and_when_to_hour_holds_where_it_is_now(store):
    a, _, pa, pb = _two_of_a_course(store)
    inst = store.world.instance
    title = inst.course_title(a.course)
    to_day = next(d for d in inst.calendar.days if d not in (pa.day, pb.day))
    model = ScriptedModel([call("move_class", course=title, kind=a.kind.value, to_day=FULL_DAY[to_day],
                                to_hour=hour(pa.slot), from_week=6, to_week=6)] * 2 + [say("which one?")])
    out = Harness(Registry(ALL_TOOLS), model).run(COORDINATOR, "a request", store, a.faculty, 5)
    first, again = out.calls[0].error, out.calls[1].error
    assert first.code == "ambiguous_class" and "at_hour" in first.hint["note"]
    assert {"on_day": FULL_DAY[pa.day], "at_hour": hour(pa.slot)} in [c["pick_with"] for c in first.hint["candidates"]]
    assert again.code == "repeated_call" and "refused before" in again.message and "ask_requester" in again.message


def test_a_run_of_weeks_the_change_misses_gets_a_note(store):
    a, _, pa, _ = _two_of_a_course(store)
    inst = store.world.instance
    to_day = next(d for d in inst.calendar.days if d != pa.day)
    move = {"course": inst.course_title(a.course), "kind": a.kind.value, "on_day": FULL_DAY[pa.day],
            "at_hour": hour(pa.slot), "to_day": FULL_DAY[to_day]}
    request = f"Please move it to {FULL_DAY[to_day]} for weeks 6 to 8."
    for weeks, noted in (((6, 6), True), ((6, 8), False)):
        model = ScriptedModel([call("move_class", **move, from_week=weeks[0], to_week=weeks[1]), say("ok")])
        result = Harness(Registry(ALL_TOOLS), model).run(COORDINATOR, request, store, a.faculty, 5).calls[0].content
        assert ("note" in result) == noted, result


def test_swap_classes_puts_each_class_where_the_other_was_and_only_v4_has_it(store):
    a, b, pa, pb = _two_of_a_course(store)
    inst = store.world.instance
    title = inst.course_title(a.course)
    names = {s["function"]["name"] for s in Registry(ALL_TOOLS).specs(COORDINATOR_V4.scopes)}
    assert "swap_classes" in names and "swap_classes" not in {
        s["function"]["name"] for s in Registry(ALL_TOOLS).specs(COORDINATOR.scopes)}
    out, _ = episode(store, [
        call("swap_classes", course=title, on_day=FULL_DAY[pa.day], at_hour=hour(pa.slot), other_course=title,
             other_day=FULL_DAY[pb.day], other_hour=hour(pb.slot), from_week=6),
        call("check_draft"),
        call("submit", message="Swapped."),
    ], actor=a.faculty, agent=COORDINATOR_V4)
    assert out.status == "submitted", out.calls[-1]
    added = out.calls[0].content["added"]
    assert len(added) == 2 and all("week 6 (firm)" in x for x in added)
    after = store.commit(out.draft).timetable(6).assignment
    assert (after[a.id].day, after[a.id].slot) == (pb.day, pb.slot)
    assert (after[b.id].day, after[b.id].slot) == (pa.day, pa.slot)


def _noted(store, request, **arguments):
    from nts.agents import COORDINATOR_V6
    from nts.harness import Harness
    from nts.models import ScriptedModel, call, say
    from nts.tools import ALL_TOOLS, Registry

    t = store.world.instance.faculty[0]
    out = Harness(Registry(ALL_TOOLS), ScriptedModel([call("constrain", teacher="me", **arguments), say("ok")])).run(
        COORDINATOR_V6, request, store, t.id, 5)
    assert out.calls[0].ok, out.calls[0].error
    return out.calls[0].content.get("note") or ""


def test_a_week_typed_loosely_is_still_a_week(store):
    """ "wee 10" named week 10: the agent was told it named no week, and went back and forth until it gave up."""
    for request in ("i am absent on wee 10", "away in wk 10", "absent the 10th week"):
        assert "names no week" not in _noted(store, request, rule="unavailable", days=["Thursday"], from_week=10,
                                             to_week=10), request


def test_an_absence_on_a_day_is_this_week_not_every_week(store):
    """ "absent on Thursday" was approved as every Thursday to the end of term."""
    note = _noted(store, "i am absent on thursday", rule="unavailable", days=["Thursday"], from_week=5)
    assert "this week (week 5)" in note and "to_week=5" in note
    assert _noted(store, "no classes for me on Thursdays", rule="no_classes", days=["Thursday"], from_week=5) == ""
    assert "this week" not in _noted(store, "I am away every Thursday from now on", rule="unavailable",
                                     days=["Thursday"], from_week=5)


def test_cancel_is_not_unavailable(store):
    note = _noted(store, "i am absent in week 10, cancel all my classes", rule="unavailable", from_week=10, to_week=10)
    assert "cancel_class" in note
