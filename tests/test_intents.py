"""Intents: the parser says what a message means, deterministic code resolves times, weeks and classes."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from core.generator import generate_department
from core.schemas import Channel, ConstraintType, Placement, Request, Role, Tier
from language.corpus import ExpectedAction
from language.intents import (
    Avoid,
    Cancel,
    ClassRef,
    ExtraClass,
    IntentOutput,
    LoadLimit,
    Move,
    NeedsRoom,
    Only,
    Period,
    RoomClosed,
    Swap,
    TimeSpec,
    Unavailable,
    Unresolved,
    Who,
    compile_intents,
    slots,
    weeks,
)
from language.rule_parser import RuleParser


@pytest.fixture(scope="module")
def instance():
    return generate_department(seed=0)


@pytest.fixture(scope="module")
def teacher(instance):
    """A teacher with lectures of two different courses, one of them taught more than once a week."""
    for f in instance.faculty:
        mine = [s for s in instance.sessions if s.faculty == f.id and s.kind.value == "lecture"]
        courses = {s.course for s in mine}
        if len(courses) > 1 and any(sum(s.course == c for s in mine) > 1 for c in courses):
            return f.id
    raise AssertionError("no suitable teacher in the generated department")


def request(sender: str, role: Role = Role.FACULTY, text: str = "...") -> Request:
    return Request(id="R-0001", channel=Channel.PORTAL, sender_id=sender, role=role, raw_text=text,
                   received_at=datetime(2026, 9, 1))


def compile_(instance, sender, *found, role=Role.FACULTY, **kw):
    return compile_intents(instance, request(sender, role), IntentOutput(action="compile", intents=list(found)), **kw)


def timetable(instance):
    """Every session at a distinct (day, slot), in session order."""
    cal = instance.calendar
    cells = [(d, q) for d in cal.days for q in range(cal.slots_per_day)]
    return {s.id: Placement(day=cells[i % len(cells)][0], slot=cells[i % len(cells)][1], room=instance.rooms[0].id)
            for i, s in enumerate(instance.sessions)}


# -- times and weeks ------------------------------------------------------------------


@pytest.mark.parametrize("spec, expected", [
    (TimeSpec(), None),
    (TimeSpec(days=["Mon"]), None),
    (TimeSpec(before="10:00"), [0]),
    (TimeSpec(before="10:30"), [0, 1]),
    (TimeSpec(after="14:00"), [5, 6, 7]),
    (TimeSpec(after="14:00", before="16:00"), [5, 6]),  # "between 2 and 4 pm"
    (TimeSpec(at=["11:00", "15:00"]), [2, 6]),
    (TimeSpec(part="morning"), [0, 1, 2, 3]),
    (TimeSpec(part="afternoon"), [5, 6, 7]),
    (TimeSpec(part="lunch"), [4]),
])
def test_clock_times_become_slots(instance, spec, expected):
    assert slots(instance, spec) == expected


@pytest.mark.parametrize("spec", [TimeSpec(before="09:00"), TimeSpec(after="17:00"), TimeSpec(at=["13:30"])])
def test_times_with_no_class_are_a_question_not_a_guess(instance, spec):
    with pytest.raises(Unresolved) as e:
        slots(instance, spec)
    assert e.value.missing == "slots" and "classes start on the hour" in e.value.question


def test_a_time_spec_says_one_thing():
    with pytest.raises(ValidationError):
        TimeSpec(part="morning", before="10:00")
    with pytest.raises(ValidationError):
        TimeSpec(before="10am")


@pytest.mark.parametrize("period, current, expected", [
    (Period(), None, None),
    (Period(weeks=[6, 5, 5]), None, [5, 6]),
    (Period(from_week=14), None, [14, 15, 16]),
    (Period(from_week=3, to_week=5), None, [3, 4, 5]),
    (Period(relative="this_week"), 7, [7]),
    (Period(relative="next_week"), 7, [8]),
])
def test_periods_become_weeks(instance, period, current, expected):
    assert weeks(instance, period, current) == expected


@pytest.mark.parametrize("period, current", [(Period(relative="next_week"), None), (Period(weeks=[17]), None),
                                             (Period(relative="next_week"), 16)])
def test_weeks_outside_the_semester_are_a_question(instance, period, current):
    with pytest.raises(Unresolved):
        weeks(instance, period, current)


@pytest.mark.parametrize("period, current, expected", [
    # "from now on, except week 16" in week 13; with nothing else named, a standing request starts now
    (Period(relative="rest_of_semester", except_weeks=[16]), 13, [13, 14, 15]),
    (Period(except_weeks=[15]), 13, [13, 14, 16]),
    (Period(from_week=3, to_week=6, except_weeks=[4, 5]), None, [3, 6]),
    (Period(parity="even"), 11, [12, 14, 16]),
    (Period(parity="odd"), None, [1, 3, 5, 7, 9, 11, 13, 15]),
    (Period(from_week=9, parity="odd", except_weeks=[13]), None, [9, 11, 15]),
])
def test_weeks_left_out_and_odd_or_even_weeks(instance, period, current, expected):
    assert weeks(instance, period, current) == expected
    assert not period.is_recurring


def test_a_period_with_nothing_left_is_a_question(instance):
    with pytest.raises(Unresolved):
        weeks(instance, Period(weeks=[4], except_weeks=[4]), None)


def test_the_rule_reader_reads_weeks_left_out_and_parity(instance):
    rules = RuleParser(instance)
    assert rules.period("I can't teach on Wednesdays from now on, except in week 16.") == Period(except_weeks=[16])
    assert rules.period("I'm away in weeks 5-8 apart from week 7.") == Period(weeks=[5, 6, 7, 8], except_weeks=[7])
    assert rules.period("On even weeks no Friday afternoons.") == Period(parity="even")
    assert rules.period("From week 10 I can only teach Monday to Wednesday.") == Period(from_week=10)
    assert rules.period("From week 6 to week 9 no Monday classes.") == Period(weeks=[6, 7, 8, 9])


def test_the_rule_reader_reads_limits_and_lectures_kept_apart(cse):
    from language.intents import Limit, Relate

    rules = RuleParser(cse)
    read = lambda text: rules.read_intents(_said("F-104", text))  # noqa: E731
    assert read("No more than two classes a day for me, please.").intents == [
        Limit(measure="classes_per_day", n=2, hard=True)]
    assert read("I'd like to teach on at most 3 days a week.").intents[0].measure == "days_per_week"
    assert read("Leave at least an hour free between any two of my classes.").intents[0].measure == "gap"
    assert read("Please keep my total teaching under ten hours a week.").action == "out_of_scope"
    apart = read("Please don't put two Algorithms lectures on the same day.").intents[0]
    assert isinstance(apart, Relate) and apart.relation == "different_day" and apart.which.kind == "lecture"


def test_a_schema_4_reading_is_a_schema_5_reading():
    from language.migrations import migrate

    old = {"action": "compile", "intents": [{"kind": "unavailable", "period": {"relative": "rest_of_semester"}}]}
    assert migrate(old, 4) == old
    assert IntentOutput.model_validate(old).intents[0].period.except_weeks is None


# -- constraint intents ----------------------------------------------------------------


def test_personal_intents_default_to_the_sender(instance, teacher):
    c = compile_intents(instance, request(teacher, text="I'm away at a conference on Tuesday of week 7."), IntentOutput(
        action="compile", intents=[
                 Unavailable(time=TimeSpec(days=["Tue"]), period=Period(weeks=[7]), justification="stated"),
                 Avoid(time=TimeSpec(before="10:00")),
                 Only(time=TimeSpec(days=["Mon", "Tue", "Wed", "Thu"]), hard=True)]))
    unav, avoid, only = c.result.constraints
    assert c.result.action == ExpectedAction.COMPILE and c.question is None
    assert (unav.type, unav.hard, unav.tier, unav.scope.faculty) == (ConstraintType.UNAVAILABLE, True,
                                                                    Tier.VERIFIED_UNAVAILABILITY, teacher)
    assert unav.valid.from_week == unav.valid.to_week == 7
    assert (avoid.type, avoid.hard, avoid.tier, avoid.when.slots) == (ConstraintType.AVOID, False, Tier.PREFERENCE, [0])
    assert (only.type, only.hard, only.when.days) == (ConstraintType.PREFER, True, ["Mon", "Tue", "Wed", "Thu"])


def test_move_names_one_class(instance, teacher):
    sid = next(s.id for s in instance.sessions if s.faculty == teacher)
    c = compile_(instance, teacher, Move(which=ClassRef(session=sid), to=TimeSpec(days=["Fri"], part="morning")))
    (pin,) = c.result.constraints
    assert (pin.type, pin.scope.session, pin.when.days, pin.when.slots) == (ConstraintType.PREFER, sid, ["Fri"],
                                                                           [0, 1, 2, 3])


def test_a_time_for_a_long_class_is_when_it_starts(instance):
    s = next(s for s in instance.sessions if s.duration == 2)
    at_two = compile_(instance, s.faculty, Move(which=ClassRef(session=s.id), to=TimeSpec(days=["Fri"], at=["14:00"])))
    assert at_two.result.constraints[0].when.slots == [5, 6]  # 2 to 4 pm: a 2-hour class starting at 2 pm
    morning = compile_(instance, s.faculty, Move(which=ClassRef(session=s.id), to=TimeSpec(part="morning")))
    assert morning.result.constraints[0].when.slots == [0, 1, 2, 3]  # a start already fits: left alone


def test_moving_a_class_that_meets_several_times_a_week_moves_one_meeting(instance, teacher):
    meetings = [o.id for o in instance.sessions if o.faculty == teacher and o.kind.value == "lecture"]
    s = next(instance.session_by_id[x] for x in meetings
             if sum(instance.session_by_id[y].course == instance.session_by_id[x].course for y in meetings) > 1)
    same = [x for x in meetings if instance.session_by_id[x].course == s.course]
    ref = ClassRef(course=s.course, kind="lecture")
    c = compile_(instance, teacher, Move(which=ref, to=TimeSpec(days=["Fri"])))
    assert [x.scope.session for x in c.result.constraints] == [same[0]]  # any meeting will do: the first

    tt = timetable(instance)
    tt[same[-1]] = tt[same[-1]].model_copy(update={"day": "Fri"})  # one meeting is on Friday already
    c = compile_(instance, teacher, Move(which=ref, to=TimeSpec(days=["Fri"])), timetable=tt)
    assert [x.scope.session for x in c.result.constraints] == [same[-1]]  # so Friday does not get two

    at = tt[s.id]
    ref = ClassRef(course=s.course, kind="lecture", day=at.day, time=f"{9 + at.slot:02d}:00")
    c = compile_(instance, teacher, Move(which=ref, to=TimeSpec(days=["Mon"])), timetable=tt)
    assert [x.scope.session for x in c.result.constraints] == [s.id]  # named by where it is: that one


def test_moving_one_of_several_different_classes_asks_which(instance, teacher):
    c = compile_(instance, teacher, Move(which=ClassRef(kind="lecture"), to=TimeSpec(days=["Fri"])))
    assert c.result.action == ExpectedAction.CLARIFY and c.question.startswith("Which class do you mean")
    assert c.result.output.clarifying_question == c.question and not c.result.constraints


def test_a_class_the_sender_does_not_teach_is_not_found(instance, teacher):
    other_course = next(s.course for s in instance.sessions if s.faculty != teacher and
                        not any(o.course == s.course and o.faculty == teacher for o in instance.sessions))
    c = compile_(instance, teacher, Move(which=ClassRef(course=other_course), to=TimeSpec(days=["Fri"])))
    assert c.missing == ["session"] and "could not find" in c.question


def test_a_move_without_a_destination_is_a_question(instance, teacher):
    sid = next(s.id for s in instance.sessions if s.faculty == teacher)
    c = compile_(instance, teacher, Move(which=ClassRef(session=sid)))
    assert c.missing == ["slots"] and not c.result.constraints


def test_someone_elses_class_is_refused_after_resolution(instance, teacher):
    other = next(f.id for f in instance.faculty if f.id != teacher and any(s.faculty == f.id for s in instance.sessions))
    c = compile_(instance, teacher, Avoid(who=Who(kind="faculty", id=other), time=TimeSpec(days=["Mon"])))
    assert c.result.action == ExpectedAction.REFUSE


def test_rooms(instance, teacher):
    lab = next(r for r in instance.rooms if r.equipment)
    c = compile_(instance, "LAB-1", RoomClosed(room=lab.id, time=TimeSpec(days=["Tue"]), period=Period(weeks=[3])),
                 role=Role.LAB_INCHARGE)
    (closed,) = c.result.constraints
    assert (closed.type, closed.tier, closed.scope.room) == (ConstraintType.UNAVAILABLE, Tier.PHYSICAL, lab.id)

    # the exam cell takes a room for exams, and only rooms
    c = compile_(instance, "S-EXAM", RoomClosed(room=lab.id, period=Period(weeks=[8])), role=Role.EXAM_CELL)
    assert c.result.action == ExpectedAction.COMPILE and c.result.constraints[0].scope.room == lab.id
    c = compile_(instance, "S-EXAM", Avoid(who=Who(kind="faculty", id=teacher), time=TimeSpec(days=["Mon"])),
                 role=Role.EXAM_CELL)
    assert c.result.action == ExpectedAction.REFUSE

    course = next(s.course for s in instance.sessions if s.faculty == teacher)
    c = compile_(instance, teacher, NeedsRoom(which=ClassRef(course=course), equipment=[lab.equipment[0]]))
    named = [s.id for s in instance.sessions if s.faculty == teacher and s.course == course]
    assert [x.scope.session for x in c.result.constraints] == named  # every meeting of the class
    assert all(x.room.equipment == [lab.equipment[0]] for x in c.result.constraints)


def test_load_limit(instance, teacher):
    c = compile_(instance, teacher, Avoid(time=TimeSpec(days=["Fri"])), LoadLimit(max_consecutive=2))
    avoid, limit = c.result.constraints
    assert (limit.type, limit.limit, limit.hard, limit.scope.faculty) == (ConstraintType.MAX_CONSECUTIVE, 2, True,
                                                                         teacher)
    assert avoid.id != limit.id


def test_a_load_limit_is_the_senders(instance, teacher):
    """Whose hours are limited is not the reader's to say: an answer that names a course (an older schema's
    "who") still limits the sender's hours."""
    out = IntentOutput.model_validate({"action": "compile", "intents": [
        {"kind": "load_limit", "who": {"kind": "course", "id": "Dr. Somebody"}, "max_consecutive": 3}]})
    c = compile_intents(instance, request(teacher, text="Please keep my runs to 3."), out)
    assert c.question is None and [(x.limit, x.scope.faculty) for x in c.result.constraints] == [(3, teacher)]


@pytest.mark.parametrize("text, hard, expected", [
    ("No teaching on Mondays, treat this as a hard rule.", True, (ConstraintType.AVOID, Tier.OPERATIONAL)),
    ("I'm at a conference on Monday, so I can't teach then.", True,
     (ConstraintType.UNAVAILABLE, Tier.VERIFIED_UNAVAILABILITY)),
    ("I have a standing meeting on Mondays; no classes then please.", True,
     (ConstraintType.UNAVAILABLE, Tier.VERIFIED_UNAVAILABILITY)),
    ("I'd rather not teach on Mondays, I can't focus.", False, (ConstraintType.AVOID, Tier.PREFERENCE)),
])
def test_the_wording_not_the_reader_decides_an_unavailability(instance, teacher, text, hard, expected):
    for read in (Unavailable(time=TimeSpec(days=["Mon"])), Avoid(time=TimeSpec(days=["Mon"]), hard=True)):
        if not hard:
            read = Avoid(time=TimeSpec(days=["Mon"]))
        c = compile_intents(instance, request(teacher, text=text), IntentOutput(action="compile", intents=[read]))
        (got,) = c.result.constraints
        assert (got.type, got.tier) == expected and got.hard == hard
        assert c.result.output.request_type == ("unavailability" if expected[0] == ConstraintType.UNAVAILABLE
                                                       else "preference")


# -- intents with their own flows ------------------------------------------------------


def test_swap_resolves_both_classes_against_the_timetable(instance, teacher):
    tt = timetable(instance)
    mine = next(s for s in instance.sessions if s.faculty == teacher)
    theirs = next(s for s in instance.sessions if s.faculty != teacher)
    def at(s):
        return {"day": tt[s.id].day, "time": f"{9 + tt[s.id].slot:02d}:00"}

    swap = Swap(mine=ClassRef(**at(mine)), theirs=ClassRef(owner=theirs.faculty, **at(theirs)), period=Period(weeks=[9]))
    c = compile_(instance, teacher, swap, timetable=tt)
    (ref,) = c.swaps
    assert (ref.mine, ref.theirs, ref.weeks) == (mine.id, theirs.id, [9])
    assert c.result.output.request_type == "swap" and c.result.action == ExpectedAction.COMPILE

    c = compile_(instance, teacher, Swap(mine=ClassRef(**at(mine)), theirs=ClassRef()), timetable=tt)
    assert c.missing == ["faculty"] and not c.swaps
    assert c.result.output.request_type == "swap"  # still routed to the swap flow


def test_extra_class_copies_any_meeting_but_needs_a_week(instance, teacher):
    s = next(s for s in instance.sessions if s.faculty == teacher and s.kind.value == "lecture" and
             sum(o.course == s.course and o.kind == s.kind and o.faculty == teacher for o in instance.sessions) > 1)
    ref = ClassRef(course=s.course, kind="lecture")
    c = compile_(instance, teacher, ExtraClass(which=ref, period=Period(weeks=[7]), time=TimeSpec(days=["Sat"])))
    assert c.question == "There are no classes on Saturday. Which day do you mean?"
    c = compile_(instance, teacher, ExtraClass(which=ref, period=Period(weeks=[7])))
    (extra,) = c.extras
    assert instance.session_by_id[extra.template].course == s.course and extra.weeks == [7]
    assert c.result.action == ExpectedAction.INVESTIGATE  # the old format's route to the extra-class flow

    c = compile_(instance, teacher, ExtraClass(which=ref))
    assert c.missing == ["weeks"] and c.result.action == ExpectedAction.INVESTIGATE
    c = compile_(instance, teacher, ExtraClass(which=ClassRef(kind="lecture"), period=Period(weeks=[7])))
    assert c.missing == ["session"] and c.question.startswith("Which class is the extra one for")


def test_cancel(instance, teacher):
    tt = timetable(instance)
    s = next(s for s in instance.sessions if s.faculty == teacher)
    c = compile_(instance, teacher, Cancel(which=ClassRef(day=tt[s.id].day, time=f"{9 + tt[s.id].slot:02d}:00"),
                                           period=Period(relative="next_week")), timetable=tt, current_week=4)
    (ref,) = c.cancels
    assert (ref.session, ref.weeks) == (s.id, [5])


def test_cancelling_classes_in_the_plural_cancels_every_one_they_fit(instance, teacher):
    """"Cancel my <course> lectures in weeks 15-16" is every one of them; "my <course> lecture" without a day
    or time is still asked about."""
    tt = timetable(instance)
    mine = [s for s in instance.sessions if s.faculty == teacher and s.kind.value == "lecture"]
    course = next(c for c in {s.course for s in mine} if sum(s.course == c for s in mine) > 1)
    wanted = sorted(s.id for s in mine if s.course == course)
    ask = Cancel(which=ClassRef(course=course, kind="lecture"), period=Period(from_week=15, to_week=16))

    def run(text):
        return compile_intents(instance, request(teacher, text=text), IntentOutput(action="compile", intents=[ask]),
                               timetable=tt, current_week=14)

    c = run("Cancel my lectures of that course from week 15 to 16")
    assert sorted(r.session for r in c.cancels) == wanted and all(r.weeks == [15, 16] for r in c.cancels)
    c = run("Cancel my lecture of that course in weeks 15-16")
    assert not c.cancels and c.missing == ["session"]


def test_routed_actions_carry_no_intents(instance, teacher):
    for action, rtype in (("answer", "policy_question"), ("investigate", "clash_report"),
                          ("out_of_scope", "out_of_scope")):
        out = IntentOutput(action=action)
        assert out.request_type == rtype
        assert compile_intents(instance, request(teacher), out).result.action == ExpectedAction(action)
    assert IntentOutput(action="clarify", topic="unavailable").request_type == "unavailability"


# -- the rule parser emits intents -------------------------------------------------------


@pytest.mark.parametrize("text, kind", [
    ("I can't teach on Tuesday in week 7 because of a conference", "unavailable"),
    ("I'd prefer no classes before 10 am on Wednesday", "avoid"),
    ("Please limit my lectures to Monday and Tuesday only", "only"),
    ("I would like Friday afternoon free.", "avoid"),
])
def test_rule_parser_reads_intents(instance, teacher, text, kind):
    out = RuleParser(instance).read_intents(request(teacher, text=text))
    assert out.action == "compile" and [i.kind for i in out.intents] == [kind]


def test_rule_parser_reads_times_as_clock_times(instance, teacher):
    rp = RuleParser(instance)
    assert rp.times("no classes before 10 am") == TimeSpec(before="10:00")
    assert rp.times("not after 2 pm") == TimeSpec(after="14:00")
    assert rp.times("move it before lunch as I won't be available after lunch") == TimeSpec(part="morning")
    assert rp.times("at 11 am or 3 pm") == TimeSpec(at=["11:00", "15:00"])
    assert rp.times("at 8 pm") == TimeSpec()  # no class then
    assert rp.slots("before 9 am") is None


@pytest.fixture(scope="module")
def cse():
    """The department the data is made from: every section has its own course of a title."""
    from pathlib import Path

    from core.instance import Instance

    return Instance.model_validate_json((Path(__file__).parents[1] / "data" / "synthetic-cse-s0.json")
                                        .read_text(encoding="utf-8"))


def _said(sender, text, role=Role.FACULTY):
    return Request(id="R-0001", channel=Channel.PORTAL, sender_id=sender, role=role, raw_text=text,
                   received_at=datetime(2026, 9, 1))


def test_course_ids_are_checked_against_the_message(cse):
    from language.intents import ClashReport, ground_references

    # a student in section 1 names Data Structures; the model picked Database Systems, of the same section
    req = _said("ST-G-01", "21 of us in Section 1 also take Computer Graphics with Section 9, and it clashes with "
                           "Data Structures on Tuesday at 2 pm.", Role.STUDENT)
    out = IntentOutput(action="investigate", intents=[ClashReport(courses=["C-095", "C-014"])])
    assert ground_references(cse, req, out).intents[0].courses == ["C-095", "C-011"]
    # each course in the section the message ties it to, else the student's own (the model had them swapped)
    req = _said("ST-G-06", "We have Computer Graphics and Information Security of Section 9 at the same time on "
                           "Wednesday (10 am), please fix this.", Role.STUDENT)
    out = IntentOutput(action="investigate", intents=[ClashReport(courses=["C-064", "C-095"])])
    assert sorted(ground_references(cse, req, out).intents[0].courses) == ["C-063", "C-096"]
    # an abbreviation names no title: the course stays
    out = IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course="C-031"),
                                                             period=Period(weeks=[5]))])
    assert ground_references(cse, _said("F-102", "An extra CG class in week 5."), out) == out


def test_a_title_another_reference_names_does_not_stand_in_for_an_abbreviated_one(cse):
    from language.intents import ground_references

    # F-105 teaches Discrete Mathematics (C-024) and Software Engineering (C-091); Section 2 also takes a
    # Software Engineering (C-023). The named title is the move's, so the abbreviated practical keeps its course.
    req = _said("F-105", "Three small things: the Monday 11 am Software Engineering lecture would be better on "
                         "Thursday at the same time; the Discrete Maths practical needs a lab with electronics "
                         "benches; and I'm away on Friday of week 12.")
    out = IntentOutput(action="compile", intents=[
        Move(which=ClassRef(course="C-023", day="Mon", time="11:00"), to=TimeSpec(days=["Thu"], at=["11:00"])),
        NeedsRoom(which=ClassRef(course="C-024", kind="practical"), equipment=["electronics"])])
    assert ground_references(cse, req, out).intents[1].which.course == "C-024"
    timetable = {"C-091-L1": Placement(day="Mon", slot=2, room="R-101"),
                 "C-024-P": Placement(day="Fri", slot=6, room="L-1")}
    c = compile_intents(cse, req, out, timetable=timetable)
    assert c.question is None and {x.scope.session for x in c.result.constraints} == {"C-091-L1", "C-024-P"}
    # alone, the abbreviated reference is still corrected to the one title the message names
    alone = IntentOutput(action="compile", intents=[out.intents[1]])
    assert ground_references(cse, _said("F-105", "The Software Engineering practical needs electronics benches."),
                             alone).intents[0].which.course == "C-023"


def test_an_absence_beside_an_extra_class_keeps_its_constraints(cse):
    # the old format keeps only the extra class's investigation; a scorer comparing readings must see the absence
    req = _said("F-104", "I'm away on Wednesday of week 6; add an extra Linear Algebra class on the Friday instead.")
    out = IntentOutput(action="compile", intents=[
        Unavailable(time=TimeSpec(days=["Wed"]), period=Period(weeks=[6])),
        ExtraClass(which=ClassRef(course="C-104"), period=Period(weeks=[6]), time=TimeSpec(days=["Fri"]))])
    c = compile_intents(cse, req, out)
    assert c.result.action == ExpectedAction.INVESTIGATE and len(c.extras) == 1
    assert [(x.type, x.when.days, x.when.weeks) for x in c.result.constraints] == [
        (ConstraintType.UNAVAILABLE, ["Wed"], [6])]


def test_a_course_taught_to_two_sections_asks_which(cse):
    # F-102 teaches Computer Graphics to sections 3 (C-031) and 9 (C-095): the title alone does not say which
    req = _said("F-102", "Could you schedule an additional Computer Graphics lecture in week 5?")
    for course in ("C-031", "C-095"):
        out = IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course=course, kind="lecture"),
                                                                 period=Period(weeks=[5]))])
        c = compile_intents(cse, req, out)
        assert c.missing == ["session"] and "Section 3" in c.question and "Section 9" in c.question
    # naming the section settles it
    out = IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course="C-095", kind="lecture",
                                                                            group="G-09"),
                                                             period=Period(weeks=[5]))])
    c = compile_intents(cse, req, out)
    assert c.question is None and cse.session_by_id[c.extras[0].template].course == "C-095"


def test_names_are_looked_up_and_an_id_reads_as_itself(cse):
    from language.names import course_id, faculty_id, room_id

    assert course_id(cse, "Data Structures") == course_id(cse, "data structres") == "C-011"
    assert course_id(cse, "C-043") == "C-043" and course_id(cse, "Underwater Basket Weaving") == "Underwater Basket Weaving"
    sharma = next(f.id for f in cse.faculty if f.name == "Dr. Sharma")
    assert faculty_id(cse, "Dr. Sharma") == faculty_id(cse, "sharma") == faculty_id(cse, "Prof Sharma") == sharma
    assert faculty_id(cse, "Dr. Nobody") == "Dr. Nobody"
    assert room_id(cse, "Room 101") == room_id(cse, "room 101") == "R-101"
    assert room_id(cse, "Lab 5") != room_id(cse, "Room 105")


def test_a_reading_with_names_compiles_as_the_one_with_ids(cse):
    from language.intents import ClashReport, Swap
    from language.names import to_names

    req = _said("ST-G-04", "11 of us in Section 4 also take Information Security wiht Section 6, and it clashes "
                           "with Data Structures on Tuesday at 9 am.", Role.STUDENT)
    ids = IntentOutput(action="investigate", intents=[ClashReport(time=TimeSpec(days=["Tue"], at=["09:00"]),
                                                                  courses=["C-043", "C-064"])])
    names = to_names(cse, ids)
    assert names.intents[0].courses == ["Data Structures", "Information Security"]
    assert compile_intents(cse, req, names).clashes == compile_intents(cse, req, ids).clashes
    # a colleague by name, and an unknown ID or the wrong colleague put right by what the message names
    sharma = next(f.id for f in cse.faculty if f.name == "Dr. Sharma")
    other = next(f.id for f in cse.faculty if f.id != sharma and f.id != "F-102")
    swap = Swap(mine=ClassRef(kind="lecture", day="Tue", time="12:00"),
                theirs=ClassRef(owner=other, day="Fri", time="14:00"))
    from language.intents import ground_references

    req = _said("F-102", "Requesting a swap with Dr. Sharma: my lecture at 12 pm on Tuesday for their class at 2 pm "
                         "on Friday.")
    got = ground_references(cse, req, IntentOutput(action="compile", intents=[swap]))
    assert got.intents[0].theirs.owner == sharma


def test_the_current_prompt_has_no_directory(cse):
    from language.intent_compiler import intent_message

    req = _said("F-102", "No classes before 10 on Monday please.")
    short, full = intent_message(cse, req, current_week=5), intent_message(cse, req, current_week=5, schema=3)
    assert "Faculty:" in full and "Faculty:" not in short and "F-102" not in short
    assert "Current week: 5" in short and req.raw_text in short and len(short) * 5 < len(full)


def test_a_fine_tuned_reader_gets_a_short_prompt_and_gemini_the_whole_vocabulary():
    from language.intent_compiler import INTENT_INSTRUCTIONS, INTENT_PROMPT, read_answer

    class Fake:
        def __init__(self, zero_shot):
            self.zero_shot, self.system = zero_shot, None

        def generate(self, system, prompt, schema):
            self.system = system
            return IntentOutput(action="answer")

    tuned, zero = Fake(False), Fake(True)
    read_answer(tuned, "...")
    read_answer(zero, "...")
    assert tuned.system == INTENT_PROMPT and zero.system == INTENT_INSTRUCTIONS
    assert len(INTENT_PROMPT) * 5 < len(INTENT_INSTRUCTIONS)


@pytest.mark.parametrize("text, intents, notes", [
    ("Away Tuesday and Wednesday of week 15; make-up classes, but not on Friday.",
     [Unavailable(time=TimeSpec(days=["Tue", "Wed"]), period=Period(weeks=[15]))], ["Friday"]),
    ("From week 15 onwards I can only teach in the afternoons.",
     [Only(time=TimeSpec(part="afternoon"), period=Period(from_week=15))], ['"only"']),
    ("Please move my Computer Graphics lecture on Wednesday at 2 pm to Friday morning.",
     [Move(which=ClassRef(course="C-041", kind="lecture", day="Wed", time="14:00"),
           to=TimeSpec(days=["Fri"], part="morning"))], []),
    ("I'm away from week 15 to week 16.", [Unavailable(period=Period(from_week=15, to_week=16))], []),
    ("I'm away in weeks 15 and 16.", [Unavailable(period=Period(weeks=[15]))], ["week 16"]),
    ("If possible no classes before 10 am on Wednesdays.", [Avoid(time=TimeSpec(days=["Wed"], before="10:00"))], []),
    # whole day names are heard too: "Wednesday", "Tuesdays"
    ("Please keep Wednesday and Tuesdays free.", [Avoid(time=TimeSpec(days=["Tue"]))], ["Wednesday"]),
    # an "only" settles every day: the days it leaves out are the days ruled out, not dropped
    ("I can't teach on Mondays, Wednesdays or Fridays; I am only on campus on Tuesdays and Thursdays this semester.",
     [Only(time=TimeSpec(days=["Tue", "Thu"]), hard=True)], []),
    # full day names are heard: "Wednesday" and "Tuesdays" too
    ("Please keep Wednesday and Tuesdays free.", [Avoid(time=TimeSpec(days=["Tue"]))], ["Wednesday"]),
    # an "only" settles every day: the days it leaves out are the days ruled out, not dropped
    ("I can't teach on Mondays, Wednesdays or Fridays; I am only on campus on Tuesdays and Thursdays this semester.",
     [Only(time=TimeSpec(days=["Tue", "Thu"]), hard=True)], []),
    # "before 1 pm" is where the morning ends: read as the morning, it is used
    ("In week 12 I can only come in the mornings; please keep my lecture before 1 pm that week.",
     [Only(time=TimeSpec(part="morning"), period=Period(weeks=[12]), hard=True)], []),
    ("From week 15 I can only teach after 12 pm.",
     [Only(time=TimeSpec(part="afternoon"), period=Period(from_week=15), hard=True)], []),
    # a time in the afternoon is not the morning, whatever word comes before it
    ("In week 12 I can only come in the mornings, and at 1 pm.",
     [Only(time=TimeSpec(part="morning"), period=Period(weeks=[12]), hard=True)], ["1 pm"]),
    ("In week 12 I can only teach in the mornings, before 3 pm.",
     [Only(time=TimeSpec(part="morning"), period=Period(weeks=[12]), hard=True)], ["3 pm"]),
])
def test_what_a_message_says_and_its_reading_does_not_use_is_noted(text, intents, notes):
    from language.intents import unaccounted

    got = unaccounted(text, IntentOutput(action="compile", intents=intents))
    assert len(got) == len(notes) and all(n in g for n, g in zip(notes, got, strict=True))


def test_an_intent_reads_back_as_a_sentence(instance, teacher):
    from language.intents import say

    s = next(s for s in instance.sessions if s.faculty == teacher)
    title = instance.course_title(s.course)
    assert say(instance, Cancel(which=ClassRef(session=s.id), period=Period(from_week=15, to_week=16)), teacher) == \
        f"Cancel my {title} {s.kind.value} from week 15 to week 16."
    assert say(instance, Avoid(time=TimeSpec(days=["Wed"], before="10:00"), hard=True), teacher) == \
        "Do not schedule my classes on Wednesday before 10 am."


def test_the_office_names_a_class_by_what_it_is_whoever_teaches_it(instance):
    """The timetable office teaches nothing: "move the Algorithms lecture on Monday at 9" is anyone's class
    that fits, found by its day and time; a teacher's own reference still means their own classes."""
    tt = timetable(instance)
    s = next(s for s in instance.sessions if s.kind.value == "lecture")
    at = tt[s.id]
    ref = ClassRef(course=s.course, kind="lecture", day=at.day, time=f"{9 + at.slot:02d}:00")
    c = compile_(instance, "C-TT", Move(which=ref, to=TimeSpec(days=["Fri"], part="afternoon")),
                 role=Role.COORDINATOR, timetable=tt)
    assert c.question is None and [p.scope.session for p in c.result.constraints] == [s.id]
    other = next(f.id for f in instance.faculty if f.id != s.faculty and any(x.faculty == f.id for x in instance.sessions))
    c = compile_(instance, other, Move(which=ref, to=TimeSpec(days=["Fri"])), timetable=tt)
    assert c.missing == ["session"] and "could not find" in c.question


def test_equipment_read_as_a_closed_room_is_the_class_need(instance):
    """"My practical needs the routers, and it has to be on Tuesday afternoon" once read as "routers is closed on
    Tuesday afternoon" plus a move: no room is called routers, some room has them, so the class needs them."""
    lab = next(r for r in instance.rooms if r.equipment)
    s = next(x for x in instance.sessions if x.kind.value == "practical")
    move = Move(which=ClassRef(session=s.id), to=TimeSpec(days=["Tue"], part="afternoon"))
    c = compile_(instance, s.faculty, RoomClosed(room=f"the {lab.equipment[0]}", time=TimeSpec(days=["Tue"], part="afternoon")),
                 move)
    assert [type(i).__name__ for i in c.output.intents] == ["NeedsRoom", "Move"]
    assert c.output.intents[0].equipment == [lab.equipment[0]]
    # a real room stays closed
    real = compile_(instance, s.faculty, RoomClosed(room=lab.id, time=TimeSpec(days=["Tue"])), move)
    assert isinstance(real.output.intents[0], RoomClosed)


@pytest.mark.parametrize("answer, picked", [
    ("section a", "Computer Programming practical for B.Tech I year CSE+ECE - Section A"),
    ("B", "Computer Programming practical for B.Tech I year CSE+ECE - Section B"),
    ("the second one", "Computer Programming practical for B.Tech I year CSE+ECE - Section B"),
    ("the practical", None),  # says only what every option says
    ("either", None),
])
def test_an_answer_picks_one_of_the_classes_the_question_listed(answer, picked):
    from language.intents import picked_option

    q = ("Which class do you mean: Computer Programming practical for B.Tech I year CSE+ECE - Section A; "
         "Computer Programming practical for B.Tech I year CSE+ECE - Section B?")
    assert picked_option(q, answer) == picked
    # "section 2" is Section 2 wherever it is listed, not the second option
    q2 = "Which class do you mean: Algorithms lecture for Section 2; Algorithms lecture for Section 1?"
    assert picked_option(q2, "section 2") == "Algorithms lecture for Section 2"
    assert picked_option("Which weeks do you mean?", answer) is None


def test_a_section_the_message_names_picks_the_class(instance):
    """F-102 teaches the Computer Graphics lecture to Section 3 and Section 9: the message (the answer to "which
    class do you mean", in full) names the section, so the reference takes it and the class is found."""
    from language.intents import classes, ground_references

    move = Move(which=ClassRef(course="C-031", kind="lecture"), to=TimeSpec(days=["Fri"]))
    out = IntentOutput(action="compile", intents=[move])
    said = request("F-102", text="Move my Computer Graphics lecture to Friday. Computer Graphics lecture for Section 9")
    ref = ground_references(instance, said, out).intents[0].which
    assert ref.group == "G-09"
    assert {instance.session_by_id[sid].course for sid in classes(instance, ref, "F-102")} == {"C-095"}
    # no section named: left to the question
    assert ground_references(instance, request("F-102", text="Move my Computer Graphics lecture to Friday."),
                             out).intents[0].which.group is None
