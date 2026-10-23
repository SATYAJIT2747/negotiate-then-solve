"""Questions about the timetable (agents.questions): what the regex reader makes of them, and the replies."""

import json
from datetime import date
from pathlib import Path

import pytest

from agents.lookup import Directory, WeekTimetable
from agents.questions import Answerer, QuestionReader, ToolCall, _spans
from web.app import World

WEEK = 13
WEDNESDAY = date(2026, 10, 21)  # "today" for the tests: "tomorrow" is Thursday
QUESTIONS = Path(__file__).parents[1] / "data" / "lookup-questions.jsonl"


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


def person(world, name):
    return next(f.id for f in world.instance.faculty if f.name == name)


@pytest.fixture(scope="module")
def reader(world):
    return QuestionReader(world.instance, sender=person(world, "Dr. Menon"), current_week=WEEK, today=WEDNESDAY)


@pytest.mark.parametrize("text, tool, week", [
    ("What are my free slots this week?", "free_hours", WEEK),
    ("any free hours next week?", "free_hours", WEEK + 1),
    ("free slots in week 15 for section 4", "free_hours", 15),
    ("what do I teach tomorrow?", "classes_of", WEEK),
])
def test_reads_the_kind_of_question_and_the_week(reader, text, tool, week):
    call = reader.read(text)
    assert call is not None and call.confident and call.tool == tool and call.week == week


def test_both_free_names_me_and_the_colleague(world, reader):
    call = reader.read("When are Dr. Iyer and I both free on Friday?")
    assert call.args == {"teachers": [person(world, "Dr. Menon"), person(world, "Dr. Iyer")], "days": ["Fri"]}


def test_a_room_at_an_hour(world, reader):
    call = reader.read("is lab 2 free on thursday at 3 pm")
    lab2 = next(r.id for r in world.instance.rooms if r.name == "Lab 2")
    assert call.args == {"rooms": [lab2], "days": ["Thu"], "slots": [6]}  # 3 pm is slot 6; nobody else asked about


def test_tomorrow_is_the_next_teaching_day(world, reader):
    assert reader.read("what do I teach tomorrow?").args["days"] == ["Thu"]
    friday = QuestionReader(world.instance, person(world, "Dr. Menon"), WEEK, today=date(2026, 10, 23))
    call = friday.read("what do I teach tomorrow?")
    assert call.args["days"] == ["Mon"] and call.week == WEEK + 1  # the weekend: Monday of the next week


def test_a_course_and_nobody_named_means_everyones_classes(world, reader):
    call = reader.read("When is Machine Learning this week?")
    assert call.tool == "classes_of" and "teacher" not in call.args and call.args["course"]


@pytest.mark.parametrize("text", [
    "Can I move my Friday lecture to Monday?",
    "Please cancel my class tomorrow.",
    "Can I hold my class before lunch, from 12 to 1?",
    "I want Friday free",
    "What is the wifi password for the library?",
])
def test_requests_and_other_questions_are_not_read_as_timetable_questions(reader, text):
    assert reader.read(text) is None


@pytest.mark.parametrize("text, note", [
    ("Is Dr. Smith free on Monday?", "don't know a teacher called Smith"),
    ("is room 999 free?", "don't know room 999"),
])
def test_unknown_names_are_not_guessed(reader, text, note):
    call = reader.read(text)
    assert not call.confident and note in call.note


def test_spans_merge_hours_into_ranges():
    assert _spans([("Mon", 2), ("Mon", 3), ("Mon", 5), ("Tue", 0)]) == "Monday 11 am–1 pm and 2–3 pm; Tuesday 9–10 am"
    assert _spans([("Mon", 2), ("Mon", 3), ("Mon", 5), ("Mon", 7)]) == "Monday 11 am–1 pm, 2–3 pm and 4–5 pm"


# -- replies ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def week(world):
    return WeekTimetable(world.instance, world.store, WEEK)


def test_free_hours_reply_lists_exactly_the_free_hours(world, week):
    me = person(world, "Dr. Menon")
    reply = Answerer(world.instance, me).answer(ToolCall("free_hours", {"teachers": [me]}, WEEK), week)
    assert reply == f"You are free on {_spans(week.free_hours(teachers=[me]))} (week {WEEK})."


def test_room_at_an_hour_is_yes_or_no(world, week):
    used = week.classes[0]
    room, (day, slot) = used.placement.room, used.hours()[0]
    a = Answerer(world.instance, person(world, "Dr. Menon"))
    assert a.answer(ToolCall("free_hours", {"rooms": [room], "days": [day], "slots": [slot]}, WEEK), week).startswith("No,")
    free = week.free_hours(rooms=[room])[0]
    assert a.answer(ToolCall("free_hours", {"rooms": [room], "days": [free[0]], "slots": [free[1]]}, WEEK), week).startswith("Yes,")


def test_classes_reply_names_every_class(world, week):
    iyer = person(world, "Dr. Iyer")
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(ToolCall("classes_of", {"teacher": iyer}, WEEK), week)
    assert reply.startswith("Dr. Iyer teaches in week 13: ") and reply.count(";") == len(week.classes_of(teacher=iyer)) - 1


def test_a_teacher_who_does_not_teach_the_course_is_told_who_does(world, week):
    c = week.classes[0]
    course, teacher = c.session.course, c.session.faculty
    other = next(f.id for f in world.instance.faculty if f.id != teacher
                 and not week.classes_of(teacher=f.id, course=course))
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(
        ToolCall("classes_of", {"teacher": other, "course": course}, WEEK), week)
    taught_by = world.instance.faculty_by_id[teacher].name
    assert "doesn't teach" in reply and f"taught by {taught_by}" in reply, reply


def test_a_section_question_names_each_teacher(world, week):
    c = week.classes[0]
    section = c.session.groups[0]
    reply = Answerer(world.instance, person(world, "Dr. Menon")).answer(
        ToolCall("classes_of", {"section": section}, WEEK), week)
    assert reply.startswith(f"{world.instance.group_by_id[section].name}'s classes in week {WEEK}: ")
    assert all(world.instance.faculty_by_id[x.session.faculty].name in reply for x in week.classes_of(section=section))


# -- the question set: how many the regex reader gets right (the baseline for a model, ROADMAP.md step 5) --


def expected_call(world, me, row):
    """The question set names people and things; the reader returns IDs."""
    d = Directory(world.instance)

    def ids(names):
        return [me if n == "me" else (x.id if not isinstance(x := d.find(n)[0], str) else x) for n in names]

    out = {}
    for k, v in row.get("args", {}).items():
        out[k] = ids(v) if k in ("teachers", "sections", "rooms") else ids([v])[0] if k in ("teacher", "section", "room", "course") else v
    return out


def test_question_set_score(world):
    me = person(world, "Dr. Menon")
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    reader = QuestionReader(world.instance, sender=me, current_week=WEEK, today=WEDNESDAY)
    right, misses = 0, []
    for row in rows:
        call = reader.read(row["question"])
        if row["tool"] is None:
            ok = call is None
        elif row["tool"] == "unclear":
            ok = call is not None and not call.confident
        else:
            week = {"this": WEEK, "next": WEEK + 1}.get(row["week"], row["week"])
            ok = (call is not None and call.confident and call.tool == row["tool"] and call.week == week
                  and call.args == expected_call(world, me, row))
        right += ok
        if not ok:
            misses.append(f"{row['id']} {row['question']!r} -> {call}")
    print(f"\nregex reader: {right}/{len(rows)} right" + "".join(f"\n  miss {m}" for m in misses))
    assert right >= len(rows) - 2, misses  # the baseline; known misses: wordings the regex has no pattern for
