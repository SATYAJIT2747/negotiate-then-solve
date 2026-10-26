import json

import pytest
from pydantic import BaseModel
from test_nts_state import away, busy_teacher, repair

from nts.protocol import ToolCall
from nts.state import ConstraintAdded, TimetableSet
from nts.tools import READ_TOOLS, Context, Refusal, Registry, Scope, tool
from nts.tools.names import FULL_DAY, Directory, hour


@pytest.fixture
def registry():
    return Registry(READ_TOOLS)


def run(registry, store, name, actor="coordinator", week=5, **args):
    result = registry.run(ToolCall(id="c1", name=name, arguments=args), Context(store.world, actor, week))
    assert result.call_id == "c1" and result.name == name
    return result


def test_specs_are_function_schemas_without_titles(registry):
    specs = registry.specs()
    assert {s["function"]["name"] for s in specs} == {t.name for t in READ_TOOLS}
    assert all(s["function"]["description"] for s in specs)
    assert "title" not in str(specs).replace("'title': {", "")
    assert registry.specs({Scope.DRAFT}) == []


def test_classes_of_uses_names_and_me(registry, store):
    inst = store.world.instance
    f = inst.faculty[2]
    by_name = run(registry, store, "classes_of", teacher=f.name.split()[-1].lower())
    mine = run(registry, store, "classes_of", actor=f.id, teacher="me")
    assert by_name.ok and mine.content == by_name.content
    assert len(by_name.content["classes"]) == sum(s.faculty == f.id for s in inst.sessions)
    assert all(c["teacher"] == f.name for c in by_name.content["classes"])
    assert "F-" not in str(by_name.content)  # names, never ids


def test_unknown_name_is_refused_with_the_closest_names(registry, store):
    r = run(registry, store, "classes_of", teacher="Dr. Iyar")
    assert not r.ok and r.error.code == "unknown_name"
    assert "Dr. Iyer" in r.error.hint["did_you_mean"]
    r = run(registry, store, "classes_of", teacher="me")  # the coordinator teaches nothing
    assert r.error.code == "not_a_teacher"


def test_bad_arguments_unknown_tools_and_scopes(registry, store):
    r = run(registry, store, "free_hours", teachers=["me"], length=0, actor=store.world.instance.faculty[0].id)
    assert r.error.code == "bad_arguments" and r.error.hint["problems"][0]["field"] == "length"
    assert run(registry, store, "book_room").error.code == "unknown_tool"
    r = registry.run(ToolCall(id="c", name="find", arguments={"text": "x"}), Context(store.world, "c"), {Scope.DRAFT})
    assert r.error.code == "unknown_tool"


def test_free_hours_skip_classes_lunch_and_absences(registry, store):
    inst, tt = store.world.instance, store.world.timetable()
    t = busy_teacher(store, "Tue")
    taught = {(p.day, q) for s, p in tt.assignment.items() if inst.session_by_id[s].faculty == t
              for q in range(p.slot, p.slot + inst.session_by_id[s].duration)}
    r = run(registry, store, "free_hours", teachers=[inst.faculty_by_id[t].name], week=7)
    full = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}
    for day, starts in r.content["free"].items():
        d = next(k for k, v in full.items() if v == day)
        assert hour(4) not in starts  # lunch
        assert not any((d, q) in taught for q in range(8) if hour(q) in starts)

    c = away(t, "Tue", 7)
    draft = store.draft(actor=t)
    draft.add(ConstraintAdded(constraint=c))
    draft.add(repair(store, c, 7))
    store.commit(draft)
    r7 = run(registry, store, "free_hours", teachers=[inst.faculty_by_id[t].name], week=7)
    r8 = run(registry, store, "free_hours", teachers=[inst.faculty_by_id[t].name], week=8)
    assert r7.content["free"]["Tuesday"] == [] and r8.content["free"]["Tuesday"]


def test_two_hour_blocks_never_straddle_a_class(registry, store):
    inst = store.world.instance
    names = [inst.faculty[0].name, inst.faculty[1].name]
    one = run(registry, store, "free_hours", teachers=names).content["free"]
    two = run(registry, store, "free_hours", teachers=names, length=2).content["free"]
    for day, starts in two.items():
        for h in starts:
            q = int(h.split()[0]) + (12 if h.endswith("pm") and not h.startswith("12") else 0) - 9
            assert h in one[day] and hour(q + 1) in one[day]


def test_who_is_free_and_free_rooms(registry, store):
    inst, tt = store.world.instance, store.world.timetable()
    sid, p = next((s, p) for s, p in tt.assignment.items() if inst.session_by_id[s].duration == 1)
    s = inst.session_by_id[sid]
    day = {"Mon": "Monday", "Tue": "tuesday", "Wed": "wed", "Thu": "Thu", "Fri": "friday"}[p.day]
    who = run(registry, store, "who_is_free", day=day, hour=hour(p.slot)).content
    assert inst.faculty_by_id[s.faculty].name not in who["free"]
    rooms = run(registry, store, "free_rooms", day=day, hour=hour(p.slot)).content["rooms"]
    assert inst.room_by_id[p.room].name not in {r["name"] for r in rooms}
    assert run(registry, store, "free_rooms", day=day, hour="2 pm", equipment=["laser"]).error.code == "unknown_equipment"
    assert run(registry, store, "who_is_free", day="Sunday", hour="9 am").error.code == "unknown_day"


def test_changes_show_moved_classes(registry, store):
    t = busy_teacher(store, "Tue")
    assert run(registry, store, "changes", week=7).content == {"week": 7, "moved": [], "not_held": []}
    c = away(t, "Tue", 7)
    draft = store.draft(actor=t)
    draft.add(ConstraintAdded(constraint=c))
    draft.add(repair(store, c, 7))
    store.commit(draft)
    moved = run(registry, store, "changes", week=7).content["moved"]
    assert moved and any(m["class"]["day"] == "Tuesday" for m in moved)


def test_read_tools_see_the_draft(registry, store):
    inst = store.world.instance
    draft = store.draft(actor="agent")
    draft.add(TimetableSet(week=5, assignment={}))  # an (invalid) empty week, visible only in the draft
    ctx = Context(store.world, "coordinator", 5, draft)
    call = ToolCall(id="c", name="classes_of", arguments={"section": inst.groups[0].name})
    assert registry.run(call, ctx).content["classes"] == []
    assert registry.run(call, Context(store.world, "coordinator", 5)).content["classes"]


def test_hours_and_names():
    from nts.kernel.generator import generate_department

    d = Directory(generate_department(seed=1, n_faculty=8, n_groups=3, n_lecture_rooms=4, n_labs=3))
    assert d.slot("2 pm") == d.slot("14:00") == d.slot("2") == d.slot(14) == 5
    assert d.slot("9 am") == 0 and hour(5) == "2 pm"
    with pytest.raises(Refusal):
        d.slot("8 pm")
    assert d.room("lab 2").name == "Lab 2" and d.section("section 3").name == "Section 3"
    assert {"kind": "teacher", "name": "Dr. Iyer"} in d.find("iyer")


def test_a_failing_tool_does_not_raise(store):
    class Nothing(BaseModel):
        pass

    @tool(Scope.READ)
    def broken(ctx: Context, args: Nothing) -> None:
        """Always fails."""
        raise KeyError("x")

    r = Registry([broken]).run(ToolCall(id="c", name="broken"), Context(store.world, "c"))
    assert not r.ok and r.error.code == "internal"
    with pytest.raises(ValueError):
        Registry([broken, broken])


def test_schemas_carry_no_null_unions_or_empty_defaults():
    from nts.tools import ALL_TOOLS

    text = json.dumps(Registry(ALL_TOOLS).specs())
    assert '{"type": "null"}' not in text and '"default": null' not in text and '"default": []' not in text
    move = next(s for s in Registry(ALL_TOOLS).specs() if s["function"]["name"] == "move_class")["function"]
    assert move["parameters"]["properties"]["to_hour"]["type"] == "string"
    assert set(move["parameters"]["required"]) == {"course", "to_day"}


def test_two_courses_under_one_title_are_told_apart(solved):
    """Some departments run two courses under one title: the teacher or the classes picked settle which."""
    from nts.state import Store
    from nts.tools import ALL_TOOLS

    inst, cons, assignment = solved
    a, b = (next(s for s in inst.sessions if s.faculty == f.id) for f in inst.faculty[:2])
    title = inst.course_title(a.course)
    twin = inst.model_copy(update={"course_titles": {**inst.course_titles, b.course: title}})
    store = Store.create(twin, [*(ConstraintAdded(constraint=c) for c in cons), TimetableSet(assignment=assignment)])
    reg = Registry(ALL_TOOLS)
    teacher_b = twin.faculty_by_id[b.faculty]

    r = run(reg, store, "classes_of", course=title, teacher="me", actor=b.faculty)
    assert r.ok and {c["teacher"] for c in r.content["classes"]} == {teacher_b.name}
    r = run(reg, store, "constrain", rule="no_classes", course=title, days=["Monday"])
    assert r.error.code == "ambiguous_name" and f"{title} ({teacher_b.name})" in r.error.hint["candidates"]
    assert Directory(twin).course(f"{title} ({teacher_b.name})") == b.course
    day = "Friday" if FULL_DAY[assignment[b.id].day] != "Friday" else "Thursday"
    args = {"course": title, "kind": b.kind.value, "to_day": day, "on_day": FULL_DAY[assignment[b.id].day],
            "at_hour": hour(assignment[b.id].slot), "from_week": 5, "to_week": 5}
    r = reg.run(ToolCall(id="c", name="move_class", arguments=args), Context(store.world, b.faculty, 5, store.draft("c")))
    assert r.ok, r.error
