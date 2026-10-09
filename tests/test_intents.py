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


# -- constraint intents ----------------------------------------------------------------


def test_personal_intents_default_to_the_sender(instance, teacher):
    c = compile_(instance, teacher,
                 Unavailable(time=TimeSpec(days=["Tue"]), period=Period(weeks=[7]), justification="stated"),
                 Avoid(time=TimeSpec(before="10:00")),
                 Only(time=TimeSpec(days=["Mon", "Tue", "Wed", "Thu"]), hard=True))
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
