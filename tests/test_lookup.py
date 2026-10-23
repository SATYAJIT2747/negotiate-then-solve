"""The timetable lookups (agents.lookup): computed answers to "who is free when", "when does X teach Y"."""

import pytest

from agents.lookup import Directory, WeekTimetable
from pipeline.store import Store
from web.app import World

WEEK = 5


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


@pytest.fixture(scope="module")
def week(world):
    return WeekTimetable(world.instance, world.store, WEEK)


def teacher_with_classes(week):
    return week.classes[0].session.faculty


def test_free_hours_are_the_gaps_in_a_teachers_week(world, week):
    teacher = teacher_with_classes(week)
    busy = {h for c in week.classes_of(teacher=teacher) for h in c.hours()}
    free = set(week.free_hours(teachers=[teacher]))
    cal = world.instance.calendar
    every = {(d, s) for d in cal.days for s in range(cal.slots_per_day) if s != cal.lunch_slot}
    assert busy and free == every - busy
    assert not any(s == cal.lunch_slot for _, s in free)  # lunch is never offered


def test_a_two_hour_class_blocks_both_hours(week):
    long = next(c for c in week.classes if c.session.duration == 2)
    free = set(week.free_hours(teachers=[long.session.faculty]))
    assert len(long.hours()) == 2 and not free & set(long.hours())


def test_two_teachers_are_free_only_when_both_are(week):
    a, b = sorted({c.session.faculty for c in week.classes})[:2]
    both = set(week.free_hours(teachers=[a, b]))
    assert both == set(week.free_hours(teachers=[a])) & set(week.free_hours(teachers=[b]))


def test_sections_and_rooms(week):
    c = week.classes[0]
    section, room = c.session.groups[0], c.placement.room
    assert not set(c.hours()) & set(week.free_hours(sections=[section]))
    assert not set(c.hours()) & set(week.free_hours(rooms=[room]))
    assert all(x.placement.room == room for x in week.classes_of(room=room))
    assert all(section in x.session.groups for x in week.classes_of(section=section))


def test_classes_of_filters_and_orders(world, week):
    teacher = teacher_with_classes(week)
    course = week.classes_of(teacher=teacher)[0].session.course
    found = week.classes_of(teacher=teacher, course=course)
    assert found and all(c.session.faculty == teacher and c.session.course == course for c in found)
    days = world.instance.calendar.days
    order = [(days.index(c.placement.day), c.placement.slot) for c in week.classes_of(teacher=teacher)]
    assert order == sorted(order)


def test_a_class_not_held_in_a_week_frees_its_hours_that_week_only(world):
    store = Store()  # a copy of the published timetable, so this test can change it
    semester = world.store.current_version()
    store.publish_version(store.propose_version(dict(semester.assignment), "test").version, "test")
    held = WeekTimetable(world.instance, store, WEEK)
    gone = held.classes[0]
    teacher = gone.session.faculty
    others = {h for c in held.classes_of(teacher=teacher) if c is not gone for h in c.hours()}
    cancelled = store.propose_version(dict(semester.assignment), "test", week=WEEK, cancelled=[gone.session.id])
    store.publish_version(cancelled.version, "test")

    week = WeekTimetable(world.instance, store, WEEK)
    assert gone.session.id not in {c.session.id for c in week.classes}
    assert set(gone.hours()) - others <= set(week.free_hours(teachers=[teacher]))
    other_week = WeekTimetable(world.instance, store, WEEK + 1)
    assert gone.session.id in {c.session.id for c in other_week.classes}


@pytest.mark.parametrize("text, expected", [
    ("iyer", "Dr. Iyer"), ("Dr. Iyer", "Dr. Iyer"), ("lab 2", "Lab 2"), ("Section 4", "Section 4"),
    ("room 101", "Room 101"), ("machine learning", "Machine Learning"), ("ML", "Machine Learning"),
    ("toc", "Theory of Computation"), ("Data Structures", "Data Structures"),
])
def test_directory_finds_what_people_call_things(world, text, expected):
    found = Directory(world.instance).find(text)
    names = [world.instance.course_title(x) if isinstance(x, str) else x.name for x in found]  # courses by title
    assert names == [expected], names


def test_directory_finds_nothing_for_nonsense(world):
    assert Directory(world.instance).find("xyzzy") == []
    assert Directory(world.instance).find("Dr.") == []


def test_free_rooms_by_kind_equipment_and_seats(world, week):
    labs = week.free_rooms(kind="lab")
    assert labs and all(r.type.value == "lab" and hours == week.free_hours(rooms=[r.id]) for r, hours in labs)
    gpu = week.free_rooms(kind="lab", equipment=["gpu"])
    assert gpu and all("gpu" in r.equipment for r, _ in gpu)
    big = max(r.capacity for r in world.instance.rooms)
    assert all(r.capacity >= big for r, _ in week.free_rooms(min_capacity=big))
    assert week.free_rooms(min_capacity=big + 1) == []
