"""The timetable lookups (agents.lookup): computed answers to "who is free when", "when does X teach Y"."""

import pytest

from agents.lookup import Department, Directory, WeekTimetable, in_runs, part_slots
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


# -- free hours in a row, with a room; who is free; the next class; changes; the department; parts of the day --


def test_length_keeps_only_runs_of_that_many_hours(world, week):
    teacher = teacher_with_classes(week)
    free = week.free_hours(teachers=[teacher])
    two = week.free_hours(teachers=[teacher], length=2)
    assert set(two) <= set(free)
    for d, s in two:  # every hour kept has a free neighbour on the same day
        assert (d, s - 1) in two or (d, s + 1) in two
    lunch = world.instance.calendar.lunch_slot
    assert in_runs([("Mon", lunch - 1), ("Mon", lunch + 1)], 2) == []  # lunch breaks a run


def test_free_with_a_room_is_free_for_the_people_and_the_room(world, week):
    c = week.classes[0]
    teacher, section = c.session.faculty, c.session.groups[0]
    found = week.free_with_room(teachers=[teacher], sections=[section], kind="lab", length=2)
    people = set(week.free_hours(teachers=[teacher], sections=[section]))
    assert found
    for room, hours in found:
        assert room.type.value == "lab" and set(hours) <= people & set(week.free_hours(rooms=[room.id]))
        assert hours == week.free_hours(teachers=[teacher], sections=[section], rooms=[room.id], length=2)


def test_who_is_free_at_an_hour(world, week):
    c = week.classes[0]
    (day, slot), busy = c.hours()[0], c.session.faculty
    free = week.who_is_free(days=[day], slots=[slot])
    names = {p.id for p, _ in free}
    assert busy not in names and names
    assert all(hours == [(day, slot)] for _, hours in free)
    for p, _ in week.who_is_free(days=[day], slots=[slot], course=c.session.course):  # substitutes for the course
        assert c.session.course in {s.course for s in world.instance.sessions if s.faculty == p.id}
    assert all(not set(x.hours()) & {(day, slot)} for g, _ in week.who_is_free([day], [slot], sections=True)
               for x in week.classes_of(section=g.id))


def test_next_class(world, week):
    teacher = teacher_with_classes(week)
    mine = week.classes_of(teacher=teacher)
    first, second = mine[0], mine[1]
    assert week.next_class(("Mon", 0), teacher=teacher) is first
    after_first = (first.placement.day, first.placement.slot + 1)
    assert week.next_class(after_first, teacher=teacher) is second
    last = mine[-1]
    assert week.next_class((last.placement.day, last.placement.slot + 1), teacher=teacher) is None


def test_changes_of_a_week(world):
    store = Store()  # a copy of the published timetable with one class moved and one not held in WEEK
    semester = world.store.current_version()
    store.publish_version(store.propose_version(dict(semester.assignment), "test").version, "test")
    base = WeekTimetable(world.instance, store, WEEK)
    gone, moved = base.classes[0], base.classes[1]
    free = next(h for h in base.free_hours(teachers=[moved.session.faculty], sections=moved.session.groups,
                                           rooms=[moved.placement.room], length=moved.session.duration))
    week_plan = dict(semester.assignment)
    week_plan[moved.session.id] = moved.placement.model_copy(update={"day": free[0], "slot": free[1]})
    v = store.propose_version(week_plan, "test", week=WEEK, cancelled=[gone.session.id])
    store.publish_version(v.version, "test")

    found = WeekTimetable(world.instance, store, WEEK).changes()
    assert {(c.kind, c.session.id) for c in found} == {("not_held", gone.session.id), ("moved", moved.session.id)}
    m = next(c for c in found if c.kind == "moved")
    assert (m.before.day, m.before.slot) == (moved.placement.day, moved.placement.slot) and m.after.day == free[0]
    theirs = WeekTimetable(world.instance, store, WEEK).changes(teacher=gone.session.faculty)
    assert ("not_held", gone.session.id) in {(c.kind, c.session.id) for c in theirs}
    assert all(c.session.faculty == gone.session.faculty for c in theirs)
    assert WeekTimetable(world.instance, store, WEEK + 1).changes() == []  # other weeks: as planned


def test_department_facts(world):
    d = Department(world.instance)
    assert [r.name for r in d.rooms("lab")] == [r.name for r in world.instance.rooms if r.type.value == "lab"]
    iyer = next(f for f in world.instance.faculty if f.name == "Dr. Iyer")
    courses = d.courses_of(iyer.id)
    assert courses and all(iyer in d.teachers_of(c) for c in courses)
    assert d.weekly_hours(iyer.id) == sum(s.duration for s in world.instance.sessions if s.faculty == iyer.id)
    assert all(d.sections_of(c) for c in courses)


def test_parts_of_the_day_leave_out_lunch(world):
    lunch = world.instance.calendar.lunch_slot
    morning, afternoon = part_slots(world.instance, "morning"), part_slots(world.instance, "afternoon")
    assert lunch not in morning + afternoon and max(morning) < lunch < min(afternoon)
    assert in_runs([("Mon", 0), ("Mon", 1), ("Mon", 3), ("Tue", 1)], 2) == [("Mon", 0), ("Mon", 1)]
