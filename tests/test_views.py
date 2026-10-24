"""The part of the timetable drawn beside an answer (agents.views): the same hours and classes as the written
answer, the days and hours that matter, and nothing to draw where there is no picture."""

import pytest

from agents.lookup import WeekTimetable
from agents.questions import Answerer, ToolCall, _spans
from agents.views import free_hours_of, view_of
from pipeline.store import Store
from web.app import World

WEEK = 13


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


@pytest.fixture(scope="module")
def week(world):
    return WeekTimetable(world.instance, world.store, WEEK)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


def test_free_hours_are_the_answers_hours(world, week):
    me, iyer = person(world, "Dr. Menon"), person(world, "Dr. Iyer")
    call = ToolCall("free_hours", {"teachers": [me, iyer], "days": ["Fri"]}, WEEK)
    v = view_of(world.instance, call, week)
    free = [(c["day"], c["slot"]) for c in v["cells"]]
    assert v["layout"] == "week" and v["days"] == ["Fri"] and all(c["kind"] == "free" for c in v["cells"])
    assert free == free_hours_of(world.instance, call, week)
    assert _spans(free) in Answerer(world.instance, me).answer(call, week)  # the words name the same hours
    assert v["slots"] == list(range(min(s for _, s in free), max(s for _, s in free) + 1))  # only those hours


def test_an_hour_asked_about_is_free_or_busy(world, week):
    c = week.classes[0]
    (day, slot), room = c.hours()[0], c.placement.room
    v = view_of(world.instance, ToolCall("free_hours", {"rooms": [room], "days": [day], "slots": [slot]}, WEEK), week)
    assert v["cells"] == [{"day": day, "slot": slot, "span": 1, "kind": "busy"}]


def test_morning_and_afternoon_and_hours_in_a_row_are_applied(world, week):
    me = person(world, "Dr. Menon")
    morning = view_of(world.instance, ToolCall("free_hours", {"teachers": [me], "part": "morning"}, WEEK), week)
    lunch = world.instance.calendar.lunch_slot
    assert morning["cells"] and all(c["slot"] < lunch for c in morning["cells"])
    two = free_hours_of(world.instance, ToolCall("free_hours", {"teachers": [me], "length": 2}, WEEK), week)
    assert set(two) <= set(week.free_hours(teachers=[me]))


def test_classes_are_the_classes_found(world, week):
    me = person(world, "Dr. Menon")
    v = view_of(world.instance, ToolCall("classes_of", {"teacher": me}, WEEK), week)
    assert [c["session"] for c in v["cells"]] == [c.session.id for c in week.classes_of(teacher=me)]
    long = next(c for c in v["cells"] if c["span"] == 2)  # a 2-hour practical covers two hours
    assert long["label"].endswith("practical") and long["sub"]
    wed = view_of(world.instance, ToolCall("classes_of", {"teacher": me, "days": ["Wed"]}, WEEK), week)
    assert wed["days"] == ["Wed"] and all(c["day"] == "Wed" for c in wed["cells"])


def test_the_next_class_is_one_class(world, week):
    me = person(world, "Dr. Menon")
    v = view_of(world.instance, ToolCall("classes_of", {"teacher": me, "after": ["Mon", 0]}, WEEK), week)
    assert len(v["cells"]) == 1 and v["cells"][0]["session"] == week.classes_of(teacher=me)[0].session.id


def test_changes_show_where_a_class_was_and_went(world):
    store = Store()  # a copy of the timetable with one class not held and one moved in WEEK
    semester = world.store.current_version()
    store.publish_version(store.propose_version(dict(semester.assignment), "test").version, "test")
    base = WeekTimetable(world.instance, store, WEEK)
    gone, moved = base.classes[0], base.classes[1]
    free = next(h for h in base.free_hours(teachers=[moved.session.faculty], sections=moved.session.groups,
                                           rooms=[moved.placement.room], length=moved.session.duration))
    plan = dict(semester.assignment)
    plan[moved.session.id] = moved.placement.model_copy(update={"day": free[0], "slot": free[1]})
    store.publish_version(store.propose_version(plan, "test", week=WEEK, cancelled=[gone.session.id]).version, "test")
    v = view_of(world.instance, ToolCall("changes", {}, WEEK), WeekTimetable(world.instance, store, WEEK))
    kinds = {(c["kind"], c["session"]) for c in v["cells"]}
    assert kinds == {("cancelled", gone.session.id), ("moved_from", moved.session.id), ("moved_to", moved.session.id)}


def test_rows_for_rooms_and_people_on_one_day(world, week):
    me = person(world, "Dr. Menon")
    labs = view_of(world.instance, ToolCall("free_rooms", {"kind": "lab", "days": ["Mon"], "part": "morning"}, WEEK), week)
    answer = Answerer(world.instance, me).answer(ToolCall("free_rooms", {"kind": "lab", "days": ["Mon"], "part": "morning"}, WEEK), week)
    assert labs["layout"] == "rows" and labs["day"] == "Mon" and all(r["name"] in answer for r in labs["rows"])
    who = view_of(world.instance, ToolCall("who_is_free", {"days": ["Mon"], "slots": [1]}, WEEK), week, sender=me)
    assert who["slots"] == [1] and all(r["name"] != "Dr. Menon" for r in who["rows"])  # the asker left out


def test_nothing_to_draw(world, week):
    me = person(world, "Dr. Menon")
    assert view_of(world.instance, ToolCall("describe", {"what": "rooms"}, WEEK), week) is None
    assert view_of(world.instance, ToolCall("unclear", week=WEEK, confident=False, note="I don't know X."), week) is None
    several_days = ToolCall("free_rooms", {"kind": "lab", "days": ["Mon", "Tue"]}, WEEK)
    assert view_of(world.instance, several_days, week) is None  # rows for a week don't fit in a chat
    assert view_of(world.instance, ToolCall("free_rooms", {"kind": "lab"}, WEEK), week) is None
    assert view_of(world.instance, ToolCall("classes_of", {"teacher": me, "days": ["Sat"]}, WEEK), week) is None


def test_with_a_room_needed_the_shaded_hours_are_when_a_room_is_free_too(world, week):
    me, section = person(world, "Dr. Menon"), week.classes[0].session.groups[0]
    call = ToolCall("free_hours", {"teachers": [me], "sections": [section], "kind": "lab", "length": 2}, WEEK)
    shaded = set(free_hours_of(world.instance, call, week))
    rooms = week.free_with_room([me], [section], "lab", length=2)
    assert shaded == {h for _, hours in rooms for h in hours}
    assert shaded <= set(week.free_hours(teachers=[me], sections=[section]))
