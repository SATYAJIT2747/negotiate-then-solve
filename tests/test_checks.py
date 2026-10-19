"""Checks on a reading and everyday English: repairs, hedges, typos and readings that contradict the timetable."""

import json
from datetime import datetime

import pytest

from core.generator import generate_department
from core.schemas import Channel, ConstraintType, Placement, Request, Role
from language.checks import check_reading, checked_compile, clock, read_repaired, repair_raw
from language.intents import (Avoid, ClassRef, ClubBooking, IntentOutput, Move, NeedsRoom, Only, Period, TimeSpec,
                              Unavailable)
from language.lexicon import hedged, mentions, names_weeks, necessary, normalise, whole_term
from language.rule_parser import RuleParser


@pytest.fixture(scope="module")
def instance():
    return generate_department(seed=0)


@pytest.fixture(scope="module")
def lecture(instance):
    """A teacher's lecture, whose course has no other title among their classes."""
    for f in instance.faculty:
        mine = [s for s in instance.sessions if s.faculty == f.id]
        titles = [instance.course_title(s.course) for s in mine]
        for s in mine:
            if s.kind.value == "lecture" and len({t for t in titles if t == instance.course_title(s.course)}) == 1:
                return s
    raise AssertionError("no suitable lecture")


@pytest.fixture
def placed(instance, lecture):
    """``lecture`` on Wednesday at 3 pm, the course's other meetings on Monday, the teacher's other classes on
    Tuesday: nothing of theirs on Thursday."""
    mine = [s for s in instance.sessions if s.faculty == lecture.faculty and s.id != lecture.id]
    room = instance.rooms[0].id
    return {lecture.id: Placement(day="Wed", slot=6, room=room),
            **{s.id: Placement(day="Mon" if s.course == lecture.course else "Tue", slot=i % 8, room=room)
               for i, s in enumerate(mine)}}


def same_course(instance, lecture) -> set[str]:
    return {s.id for s in instance.sessions if s.course == lecture.course and s.faculty == lecture.faculty}


def request(sender: str, text: str) -> Request:
    return Request(id="R-0001", channel=Channel.PORTAL, sender_id=sender, role=Role.FACULTY, raw_text=text,
                   received_at=datetime(2026, 9, 1))


def test_everyday_spelling_is_normalised_but_names_and_numbers_are_not():
    assert normalise("no clases b4 11 on thrusday plz") == "no classes before 11 on thursday please"
    assert normalise("move it to fri mornin, 3pm") == "move it to friday morning, 3 pm"
    assert normalise("Room L3, course C-054, Dr. Fried") == "room l3, course c-054, dr. fried"


def test_a_title_is_found_with_a_typo_or_by_acronym():
    assert mentions("my computer architecure lecture", "Computer Architecture")
    assert mentions("can u shift CA to friday", "Computer Architecture", acronyms=True)
    assert not mentions("can u shift CA to friday", "Computer Architecture")  # acronyms only when asked
    assert not mentions("my Data Structures class", "Database Systems")


def test_hedges_and_requirements():
    assert hedged("any chance it could move? no worries if not") and not necessary("no worries if not")
    assert hedged("It would be nice; no problem if it can't.") and not necessary("no problem if it can't.")
    assert necessary("It has to be at 1 pm, no other time works.") and not hedged("It has to be at 1 pm.")
    assert not hedged("Is it possible to move my lecture?")  # politeness, not a hedge


def test_clocks_from_what_a_model_writes():
    assert [clock(x) for x in ("3 pm", "9:00", "15.30", "12 am", "3", "10", "all")] == \
        ["15:00", "09:00", "15:30", "00:00", "15:00", "10:00", None]


def test_an_answer_that_misses_the_schema_on_a_detail_is_repaired_not_lost():
    text = json.dumps({"action": "compile", "intents": [
        {"kind": "unavailable", "time": {"days": ["Wednesday"], "at": ["all"]}, "period": {"relative": "next_week"}}]})
    out, notes = read_repaired(text)
    assert out.intents[0].time == TimeSpec(days=["Wed"]) and out.intents[0].period.relative == "next_week"
    assert notes
    data, notes = repair_raw({"intents": [{"kind": "avoid", "time": {"before": "15:00", "after": "15:00"}}]})
    assert data["intents"][0]["time"] == {"at": ["15:00"], "before": None, "after": None} and notes


def test_a_requirement_in_a_hedged_message_is_a_wish(instance, lecture):
    out = IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Mon"], before="10:00"), hard=True)])
    got = check_reading(instance, request(lecture.faculty, "i'd rather not teach before 10 on mondays"), out)
    assert got.output.intents[0].hard is False and got.notes
    must = check_reading(instance, request(lecture.faculty, "I must not teach before 10 on Mondays"), out)
    assert must.output.intents[0].hard is True and not must.notes


def test_a_time_the_class_is_not_at_is_read_as_where_it_should_go(instance, lecture, placed):
    title = instance.course_title(lecture.course)
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(course=lecture.course, kind="lecture", day="Thu", time="14:00"), hard=True)])
    c = checked_compile(instance, request(lecture.faculty, f"I need my {title} lecture on Thursday at 2 pm"), out,
                        timetable=placed)
    assert c.question is None, c.question
    [k] = c.result.constraints
    assert k.scope.session in same_course(instance, lecture)
    assert (k.when.days, k.when.slots, k.hard) == (["Thu"], [5], True)


def test_a_wish_to_move_a_class_where_it_already_is_is_asked_about(instance, lecture, placed):
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(course=lecture.course, kind="lecture"), to=TimeSpec(days=["Wed"], at=["15:00"]))])
    c = checked_compile(instance, request(lecture.faculty, "can we do it at 3 instead"), out, timetable=placed)
    assert c.question and "already" in c.question and not c.result.constraints
    keep = out.model_copy(update={"intents": [out.intents[0].model_copy(update={"hard": True})]})
    c = checked_compile(instance, request(lecture.faculty, "it has to stay at 3 pm on Wednesday"), keep,
                        timetable=placed)
    assert c.question is None and c.result.constraints  # a hard one keeps it there


def test_a_requirement_on_all_times_in_a_message_about_one_class_is_asked_about(instance, lecture):
    title = instance.course_title(lecture.course)
    out = IntentOutput(action="compile", intents=[Only(time=TimeSpec(days=["Thu"], at=["14:00"]), hard=True)])
    c = checked_compile(instance, request(lecture.faculty, f"I need my {title} lecture on Thursday at 2 pm"), out)
    assert c.question and title in c.question and not c.result.constraints


def test_a_room_need_that_leaves_out_the_time_the_message_gives_gets_it_back(instance, lecture):
    title = instance.course_title(lecture.course)
    which = ClassRef(course=title, kind="lecture")
    out = IntentOutput(action="compile", intents=[NeedsRoom(which=which, equipment=["routers"])])
    text = f"My {title} lecture needs the routers, and it has to be on Tuesday afternoon."
    c = check_reading(instance, request(lecture.faculty, text), out)
    need, move = c.output.intents
    assert c.question is None and need.kind == "needs_room"
    assert (move.kind, move.which, move.to.days, move.to.part, move.hard) == ("move", which, ["Tue"], "afternoon", True)
    c = check_reading(instance, request(lecture.faculty, f"My {title} lecture needs routers; Thursday at 2 pm please."),
                      out)
    move = c.output.intents[1]
    assert (move.to.days, move.to.at, move.hard) == (["Thu"], ["14:00"], False)
    # a room need alone, or for the class that is on a day, is read as it is
    c = checked_compile(instance, request(lecture.faculty, f"My {title} lecture needs the routers."), out)
    assert c.question is None and c.result.constraints
    on_wed = NeedsRoom(which=ClassRef(course=title, kind="lecture", day="Wed"), equipment=["routers"])
    c = check_reading(instance, request(lecture.faculty, f"My Wednesday {title} lecture needs the routers."),
                      IntentOutput(action="compile", intents=[on_wed]))
    assert c.question is None


def test_weeks_and_the_whole_term_are_read_with_a_typo():
    assert names_weeks("away in wek 11, sorry") and names_weeks("for the rst of the trem")
    assert names_weeks("is a room free thhis tuesday?") and names_weeks("on 23 November")
    assert not names_weeks("Looking for a large classroom on Thursday 2-5pm for the poetry reading")
    assert whole_term("unavailable on Thursday mornings for the whole semester")
    assert whole_term("for the duration of the entire semester")
    assert not whole_term("for what's left of the semester") and not whole_term("from now until the term ends")


def test_a_period_the_message_does_not_give_is_every_week(instance, lecture):
    club = ClubBooking(club="Music Society", day="Fri", start="14:00", end="16:00", period=Period(relative="this_week"))
    got = check_reading(instance, request(lecture.faculty, "A room on Friday 14:00 to 16:00 for band practice?"),
                        IntentOutput(action="compile", intents=[club]))
    assert got.output.intents[0].period.is_recurring and got.notes
    off = Unavailable(time=TimeSpec(days=["Thu"], part="morning"), period=Period(relative="rest_of_semester"))
    got = check_reading(instance, request(lecture.faculty, "I'm unavailable on Thursday mornings for the whole "
                                                           "semester (clinical duty)."),
                        IntentOutput(action="compile", intents=[off]))
    assert got.output.intents[0].period.is_recurring
    # what the message does say stays: "the rest of the term", and this Friday
    for text in ("For the rest of the term I can't teach on Thursday mornings.", "Is a room free this Friday?"):
        got = check_reading(instance, request(lecture.faculty, text), IntentOutput(action="compile", intents=[off]))
        assert got.output.intents[0].period == off.period and not got.notes


def test_a_reading_that_leaves_out_the_class_the_message_is_about_is_asked_about(instance, lecture):
    title = instance.course_title(lecture.course)
    out = IntentOutput(action="compile", intents=[Only(time=TimeSpec(days=["Tue"], part="afternoon"))])
    c = checked_compile(instance, request(lecture.faculty, f"Tuesday afternoon would suit my {title} lecture "
                                                            "better."), out)
    assert c.question and title in c.question and not c.result.constraints
    # an acronym counts in capitals only: "this is a must" is not about Information Security
    assert check_reading(instance, request(lecture.faculty, "Keep me free on Fridays; this is a must."),
                         IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Fri"]))])).question is None


def test_checks_leave_a_plain_reading_alone(instance, lecture):
    out = IntentOutput(action="compile", intents=[Unavailable(time=TimeSpec(days=["Fri"]))])
    got = check_reading(instance, request(lecture.faculty, "I can't teach on Friday"), out)
    assert got.output == out and not got.notes and got.question is None


def test_the_rule_parser_reads_typos_and_does_not_pin_every_class(instance, lecture):
    rules = RuleParser(instance)
    title = instance.course_title(lecture.course)
    r = rules.parse(request(lecture.faculty, "no clases b4 11 on thrusday plz"))
    [c] = r.constraints
    assert (c.type, c.when.days, c.hard) == (ConstraintType.AVOID, ["Thu"], False)
    # "from ... to ...": the destination is what follows "to"; nothing says it must be, so a wish
    r = rules.parse(request(lecture.faculty, f"pls move my {title.lower()} lectur from wednesday 3pm to fri mornin"))
    [c] = r.constraints
    assert c.scope.session in same_course(instance, lecture) and (c.when.days, c.hard) == (["Fri"], False)
    # a class named: that class, not every class of theirs
    r = rules.parse(request(lecture.faculty, f"I need my {title} lecture on Thursday at 2 pm"))
    assert r.constraints and all(c.scope.session in same_course(instance, lecture) for c in r.constraints)
    # the reading the corpus labels were made with is kept
    old = RuleParser(instance, robust=False).parse(request(lecture.faculty, "no clases b4 11 on thrusday plz"))
    assert not old.constraints


def test_the_fallback_asks_about_what_it_cannot_read(instance, lecture):
    rules = RuleParser(instance)
    title = instance.course_title(lecture.course)
    for text in ("I won't be around on Mondays over the coming two weeks.",  # weeks counted from now
                 "Could my Friday 11 am lecture move to Monday morning? Separately, I can't teach in week 5.",
                 f"2 pm on Monday would suit my {title} lecture better."):  # no word for free or filled
        r = rules.parse(request(lecture.faculty, text))
        assert r.action.value == "clarify" and not r.constraints, text
    # a move that also says the class can't stay is a move, not an unavailability
    r = rules.parse(request(lecture.faculty, f"I want my {title} lecture moved to Thursday after lunch; "
                                             "it can't stay where it is."))
    assert r.constraints and all(c.type == ConstraintType.PREFER for c in r.constraints)
