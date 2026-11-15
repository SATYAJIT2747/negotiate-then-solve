"""Checks on a reading and everyday English: repairs, hedges, typos and readings that contradict the timetable."""

import json
from datetime import datetime

import pytest

from core.generator import generate_department
from core.schemas import Channel, ConstraintType, Placement, Request, Role
from language.checks import (
    check_reading,
    checked_compile,
    clock,
    read_repaired,
    repair_raw,
)
from language.intents import (
    Avoid,
    ClashReport,
    ClassRef,
    ClubBooking,
    IntentOutput,
    Move,
    NeedsRoom,
    Only,
    Period,
    Relate,
    TimeSpec,
    Unavailable,
    Who,
)
from language.lexicon import (
    firmness_rule,
    hedged,
    mentions,
    names_weeks,
    necessary,
    normalise,
    whole_term,
)
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


def test_the_firmness_rule_by_kind():
    assert firmness_rule("avoid", "Nothing of mine on Tuesday afternoons please, I need that time kept free.")
    assert firmness_rule("avoid", "No teaching on Monday after lunch, treat this as a hard rule.")
    assert firmness_rule("avoid", "Please keep me free on Fridays; this is not negotiable.")
    assert not firmness_rule("avoid", "Could you avoid Fridays for me? It's hard for me.")  # a difficulty
    assert not firmness_rule("avoid", "No classes on Fridays for me, if possible.")
    assert firmness_rule("only", "Only Tuesday and Thursday work for me; I'm away from campus the rest of the week.")
    assert not firmness_rule("only", "Ideally only Tuesday and Thursday, I'm away a lot.")
    assert not firmness_rule("only", "I'd like to teach only on Monday and Tuesday so I can do fieldwork.")
    assert not firmness_rule("avoid", "Please avoid 12 on Tuesdays; that's my only time to eat.")  # not a placement
    assert firmness_rule("unavailable", "I'm off on Friday.")


def test_a_wish_the_message_says_must_be_so_is_a_requirement(instance, lecture):
    out = IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Tue"], part="afternoon"))])
    got = check_reading(instance, request(lecture.faculty, "Nothing of mine on Tuesday afternoons please, I need "
                                                           "that time kept free."), out)
    assert got.output.intents[0].hard is True and got.notes
    plain = check_reading(instance, request(lecture.faculty, "No classes on Tuesday afternoons for me."), out)
    assert plain.output.intents[0].hard is False and not plain.notes
    # in a message of two requests, whose words the must is about is not plain: left as read
    two = IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Tue"], part="afternoon")),
                                                  Avoid(time=TimeSpec(days=["Fri"]))])
    both = check_reading(instance, request(lecture.faculty, "No Tuesday afternoons, and I need Fridays free."), two)
    assert [i.hard for i in both.output.intents] == [False, False]


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


def test_a_must_about_leaving_rules_out_where_the_class_is_and_keeps_the_target_a_wish(instance, lecture, placed):
    now = placed[lecture.id]
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(session=lecture.id), to=TimeSpec(days=["Thu"], part="afternoon"), hard=True)])
    c = checked_compile(instance, request(lecture.faculty, "I want it moved to Thursday after lunch; it can't stay "
                                          "where it is."), out, timetable=placed)
    leave, to = sorted(c.result.constraints, key=lambda k: k.type.value)
    assert (leave.type.value, leave.hard, leave.when.days, leave.when.slots) == ("avoid", True, [now.day], [now.slot])
    assert (to.type.value, to.hard, to.when.days) == ("prefer", False, ["Thu"])


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


def test_what_the_message_cannot_have_asked_for_is_dropped(instance, lecture):
    from language.intents import AnyOf, Cancel, Relate

    ref = ClassRef(day="Wed", time="15:00")
    a = Move(id="a", which=ref, to=TimeSpec(days=["Thu"], part="afternoon"))
    b = Move(id="b", which=ClassRef(same="a"), to=TimeSpec(days=["Thu"], before="13:00"))
    plain = request(lecture.faculty, "I want my Wednesday class (3pm) moved to Thursday after lunch; it can't stay.")
    # a choice the message does not offer is its first item, firm if the choice was; its label goes
    got = check_reading(instance, plain, IntentOutput(action="compile", intents=[a, b, AnyOf(of=["a", "b"], hard=True)]))
    assert got.output.intents == [a.model_copy(update={"id": None, "hard": True})]
    # a second move of the same class outside any choice repeats the first (and, the one request left, is as
    # firm as the labelling rule makes it: "it can't stay")
    assert check_reading(instance, plain, IntentOutput(action="compile", intents=[a, b])).output.intents == \
        [a.model_copy(update={"id": None, "hard": True})]
    # an item said twice, once by naming the other's class, is read once
    cancel = Cancel(id="a", which=ref, period=Period(weeks=[5]))
    twice = Cancel(id="b", which=ClassRef(same="a"), period=Period(weeks=[5]))
    assert check_reading(instance, request(lecture.faculty, "Cancel my Wednesday 3 pm class in week 5."),
                         IntentOutput(action="compile", intents=[cancel, twice])).output.intents == \
        [cancel.model_copy(update={"id": None})]
    # a choice the message offers stays, with the labels it needs
    offered = request(lecture.faculty, "Could my Wednesday 3 pm class move to Thursday afternoon or Friday morning?")
    choice = [a, b.model_copy(update={"to": TimeSpec(days=["Fri"], part="morning")}), AnyOf(of=["a", "b"])]
    assert check_reading(instance, offered, IntentOutput(action="compile", intents=choice)).output.intents == choice
    # the same relation said once per day is said once
    lec = ClassRef(course=lecture.course, kind="lecture")
    days = [Relate(id=f"r{d}", which=lec, relation="different_day") for d in ("Mon", "Tue", "Wed")]
    assert check_reading(instance, request(lecture.faculty, "No two of my lectures on one day."),
                         IntentOutput(action="compile", intents=days)).output.intents == \
        [Relate(which=lec, relation="different_day")]


def test_a_course_named_by_title_has_its_time_read_as_where_it_should_go(instance, lecture, placed):
    # the reading names the course as the message does; the timetable knows it by ID
    title = instance.course_title(lecture.course)
    text = f"4 pm on Thursday would suit my {title} class better."
    out = IntentOutput(action="compile", intents=[Move(which=ClassRef(course=title, day="Thu", time="16:00"))])
    c = check_reading(instance, request(lecture.faculty, text), out, placed)
    [move] = c.output.intents
    assert (move.which.day, move.which.time, move.to.days, move.to.at) == (None, None, ["Thu"], ["16:00"])
    # the same time given as where the class is and where it should go: where it should go
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(course=title, day="Thu", time="16:00"), to=TimeSpec(days=["Thu"], at=["16:00"]))])
    [move] = check_reading(instance, request(lecture.faculty, text), out, placed).output.intents
    assert (move.which.day, move.which.time, move.to.at) == (None, None, ["16:00"])


def test_a_message_for_one_courses_classes_is_read_as_about_them(instance, lecture):
    title = instance.course_title(lecture.course)
    days = instance.calendar.days
    out = IntentOutput(action="compile", intents=[Only(time=TimeSpec(days=[d for d in days if d != "Fri"]), hard=True)])
    c = check_reading(instance, request(lecture.faculty, f"Any day apart from Friday works for my {title} classes."),
                      out)
    assert c.question is None
    [avoid] = c.output.intents
    assert (avoid.kind, avoid.who, avoid.time.days, avoid.hard) == ("avoid", Who(kind="course", id=title), ["Fri"],
                                                                    False)
    # naming the course without saying the times are for it is still asked about
    c = check_reading(instance, request(lecture.faculty, f"About {title}: any day apart from Friday works."), out)
    assert c.question and title in c.question



def test_a_teacher_of_one_course_is_not_asked_which_classes():
    """ "In week 12 I can only come in the mornings; please keep my Cloud Computing lecture before 1 pm", from a
    guest who teaches nothing else: "your Cloud Computing classes only, or all your classes?" asks about one
    set of classes twice."""
    from semester.campus import campus_offerings
    from semester.department import instance_from_offerings

    campus = instance_from_offerings(campus_offerings(), {}, name="Campus")
    titles: dict[str, set[str]] = {}
    for s in campus.sessions:
        titles.setdefault(s.faculty, set()).add(campus.course_title(s.course))
    who, [title] = next((f, sorted(t)) for f, t in titles.items() if len(t) == 1)
    out = IntentOutput(action="compile", intents=[Only(time=TimeSpec(part="morning"), hard=True)])
    c = check_reading(campus, request(who, f"I can only come in the mornings; please keep my {title} lecture "
                                           "before 1 pm."), out)
    assert c.question is None, c.question

def test_the_answer_to_which_classes_is_read(instance, lecture):
    """"Should this apply to your X classes only, or to all your classes?": the answer comes after the request
    (``web.world.joined``) and is read, not asked about again."""
    title = instance.course_title(lecture.course)
    text = f"In week 12 I can only come in the mornings; please keep my {title} lecture before 1 pm that week."
    out = IntentOutput(action="compile", intents=[Only(time=TimeSpec(part="morning"), period=Period(weeks=[12]),
                                                       hard=True)])
    asked = check_reading(instance, request(lecture.faculty, text), out)
    assert asked.question and title in asked.question
    for answer in ("all all classes", "All my classes.", "every class", "everything"):
        c = check_reading(instance, request(lecture.faculty, f"{text} {answer}"), out)
        assert c.question is None and c.output.intents == out.intents, answer
    for answer in (f"{title} only", f"just {title}", f"Only {title}, please."):
        c = check_reading(instance, request(lecture.faculty, f"{text} {answer}"), out)
        assert c.question is None, answer
        assert [i.who for i in c.output.intents] == [Who(kind="course", id=title)], answer


def test_a_clash_that_is_why_a_class_moves_is_not_a_clash_report(instance, lecture):
    text = "My Wednesday 2 pm lecture has to move to Monday morning; where it is now clashes with a meeting."
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(kind="lecture", day="Wed", time="14:00"), to=TimeSpec(days=["Mon"], part="morning")),
        ClashReport(time=TimeSpec(days=["Wed"], at=["14:00"]), justification="stated")])
    c = check_reading(instance, request(lecture.faculty, text), out)
    [move] = c.output.intents
    assert move.kind == "move" and move.hard is True  # "has to move", once it is the message's one request
    # a clash at another time is left as read
    other = out.model_copy(update={"intents": [out.intents[0], ClashReport(time=TimeSpec(days=["Thu"], at=["10:00"]))]})
    assert len(check_reading(instance, request(lecture.faculty, text), other).output.intents) == 2


def test_a_club_booking_has_the_hours_the_message_gives(instance, lecture):
    out = IntentOutput(action="compile", intents=[ClubBooking(club="Robotics Club", day="Tue", start="19:00")])
    c = check_reading(instance, request(lecture.faculty, "Robotics Club session: Tuesday, 7-9pm, 12-odd people."), out)
    assert (c.output.intents[0].start, c.output.intents[0].end) == ("19:00", "21:00")
    wrong = out.model_copy(update={"intents": [ClubBooking(club="Coding Club", day="Fri", start="16:00", end="16:40")]})
    c = check_reading(instance, request(lecture.faculty, "Coding Club contest: Friday, from 2 till 4 pm."), wrong)
    assert (c.output.intents[0].start, c.output.intents[0].end) == ("14:00", "16:00")
    c = check_reading(instance, request(lecture.faculty, "Coding Club contest on Friday afternoon."), wrong)
    assert c.output.intents == wrong.intents  # no range said: as read


def test_a_move_of_a_course_the_message_does_not_name_is_dropped(instance, lecture):
    title = instance.course_title(lecture.course)
    other = next(t for t in sorted(set(instance.course_titles.values())) if t != title and t not in title)
    out = IntentOutput(action="compile", intents=[
        Move(id="a", which=ClassRef(course=title), to=TimeSpec(days=["Wed"], part="morning")),
        Move(id="b", which=ClassRef(course=other), to=TimeSpec(days=["Wed"], part="morning")),
        Relate(which=ClassRef(same="a"), relation="different_day", other=ClassRef(same="b"))])
    c = check_reading(instance, request(lecture.faculty, f"Wednesday morning would suit my {title} class better."), out)
    [move] = c.output.intents
    assert move.which.course == title
    both = check_reading(instance, request(lecture.faculty, f"Move {title} and {other} to Wednesday morning."), out)
    assert len(both.output.intents) == 3


@pytest.mark.parametrize(("text", "period"), [
    ("My lecture has to be on Monday at 11 am in week 10; I have a guest speaker.", {"weeks": [10]}),
    ("Please move my lecture to Monday at 11 am for weeks 10-12.", {"from_week": 10, "to_week": 12}),
    ("Move my lecture to Monday at 11 am in weeks 10 and 12.", {"weeks": [10, 12]}),
    ("From week 10 my lecture should be on Monday at 11 am.", {"from_week": 10}),
    ("Move my lecture to Monday at 11 am every week from now on.", {}),  # every week, as read
])
def test_weeks_the_message_names_by_number_are_kept_when_the_reading_drops_them(instance, lecture, text, period):
    move = Move(which=ClassRef(session=lecture.id), to=TimeSpec(days=["Mon"], at=["11:00"]), hard=True)
    got = check_reading(instance, request(lecture.faculty, text), IntentOutput(action="compile", intents=[move]))
    p = got.output.intents[0].period
    assert {k: v for k, v in p.model_dump().items() if v} == period
    # two requests in one message: which of them a week is about is not plain, so neither is changed
    two = IntentOutput(action="compile", intents=[move, Avoid(time=TimeSpec(days=["Fri"]))])
    got = check_reading(instance, request(lecture.faculty, text + " And no Fridays for me."), two)
    assert all(i.period.is_recurring for i in got.output.intents)
