"""The orchestrator with a parser that reads intents: swaps, extra classes, cancellations and
withdrawals are routed by what it read, not by keywords in the message."""

import os

import pytest

from agents.negotiation import Reply
from language.corpus import FULL_DAY, hour
from language.intent_compiler import clock
from language.intents import (
    Avoid,
    Cancel,
    ClashReport,
    ClassRef,
    ClubBooking,
    ExtraClass,
    IntentOutput,
    Period,
    Swap,
    TimeSpec,
    compile_intents,
)


class IntentStub:
    """Fixed intents for the messages it knows; for any other message, no intents (``compile`` returns
    None) and the rule parser's plain parse, as a model trained on the older format would give."""

    def __init__(self, instance, plain) -> None:
        self.instance, self.plain, self.known = instance, plain, {}
        self.calls = []

    def compile(self, request, timetable=None, current_week=None, today=None):
        out = self.known.get(request.raw_text)
        self.calls.append((request.raw_text, current_week))
        if out is None:
            return None
        return compile_intents(self.instance, request, out, timetable=timetable, current_week=current_week)

    def parse(self, request):
        return self.plain.parse(request)


class Fixed:
    def __init__(self, decision: str) -> None:
        self.decision, self.seen = decision, []

    def respond(self, message):
        self.seen.append(message)
        return Reply(decision=self.decision, choice="A" if self.decision == "accept" else None)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    os.environ.setdefault("NTS_RETRIEVER", "bm25")
    from web.world import World

    # its own semester folder: not the one the app keeps in runs/ (which a test must neither read nor write);
    # the demo department's semester timetable is built in the background at start-up, quickly here
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NTS_SEMESTER_TIME", "20")
        w = World(parser_mode="offline", seed_history=False, deadline=2,
                  semester_dir=tmp_path_factory.mktemp("semester"))
    w.orch.system_one = None  # every request goes to the parser under test
    w.orch.parser = IntentStub(w.instance, w.orch.parser)
    w.orch.current_week = lambda: 5
    return w


def submit(world, sender, text, out):
    world.orch.parser.known[text] = out
    return world.orch.submit(world.intake.from_portal(sender, text))


def where(s, tt):
    return {"day": tt[s.id].day, "time": clock(tt[s.id].slot)}


def _pair(world):
    inst, tt = world.instance, world.store.current_version().assignment
    lect = [s for s in inst.sessions if s.kind.value == "lecture" and s.id in tt]
    a = lect[0]
    b = next(s for s in lect if s.faculty != a.faculty and tt[s.id].day != tt[a.id].day)
    return a, b, tt


def test_a_swap_is_routed_by_its_intent_and_resolved_against_the_timetable(world):
    a, b, tt = _pair(world)
    yes = world.orch.responders[b.faculty] = Fixed("accept")
    text = "Could Dr. X and I trade our classes in week 3?"  # no swap word the old routing looks for
    case = submit(world, a.faculty, text, IntentOutput(action="compile", intents=[Swap(
        mine=ClassRef(**where(a, tt)), theirs=ClassRef(owner=b.faculty, **where(b, tt)), period=Period(weeks=[3]))]))
    assert "+intents" in case.route and "+swap" in case.route
    assert case.status.value == "awaiting_approval", case.reply
    assert yes.seen and yes.seen[0].to == b.faculty
    new = case.proposal.assignment
    assert (new[a.id].day, new[a.id].slot) == (tt[b.id].day, tt[b.id].slot) and case.proposal.week == 3


def test_a_swap_that_names_no_classes_asks(world):
    a, _, _ = _pair(world)
    case = submit(world, a.faculty, "Can I swap with a colleague?",
                  IntentOutput(action="compile", intents=[Swap(mine=ClassRef(), theirs=ClassRef())]))
    assert case.status.value == "clarification_requested" and "Who would you like to swap with" in case.reply


def test_an_extra_class_by_intent_uses_the_current_week(world):
    t = next(s for s in world.instance.sessions if s.kind.value == "lecture")
    case = submit(world, t.faculty, "One more session for my students next week, please.",
                  IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course=t.course),
                                                                    period=Period(relative="next_week"))]))
    assert case.route.endswith("+extra") and case.status.value == "awaiting_approval", case.reply
    assert case.proposal.week == 6  # the current week is 5
    assert world.instance.session_by_id[case.extra["template"]].course == t.course


def test_an_extra_class_without_a_week_asks_for_it(world):
    t = next(s for s in world.instance.sessions if s.kind.value == "lecture")
    case = submit(world, t.faculty, "One more session for my students, please.",
                  IntentOutput(action="compile", intents=[ExtraClass(which=ClassRef(course=t.course))]))
    assert case.status.value == "clarification_requested" and "Which week" in case.reply


def test_a_cancellation_goes_to_the_coordinator_with_the_class_resolved(world):
    a, _, tt = _pair(world)
    case = submit(world, a.faculty, "Please call off that class in week 9.", IntentOutput(action="compile", intents=[
        Cancel(which=ClassRef(**where(a, tt)), period=Period(weeks=[9]))]))
    assert case.status.value == "forwarded" and "+cancel" in case.route
    assert world.instance.course_title(a.course) in case.reply and "week 9" in case.reply
    assert FULL_DAY[tt[a.id].day] in case.reply and hour(tt[a.id].slot) in case.reply


def test_withdrawing_by_message_takes_back_the_one_open_request(world):
    a, _, _ = _pair(world)
    first = submit(world, a.faculty, "I'd like to keep Friday free if possible, at 3 pm.",
                   IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Fri"], at=["15:00"]))]))
    open_ = [c for c in world.orch.cases.values() if c.request.sender_id == a.faculty
             and world.orch.withdrawable(c) is None]
    case = submit(world, a.faculty, "Please ignore my earlier request.", IntentOutput(action="withdraw"))
    assert "+withdraw" in case.route and case.status.value == "answered"
    if len(open_) == 1:
        assert open_[0].status.value == "withdrawn" and open_[0].id in case.reply
    else:  # several open: nothing is withdrawn on a guess
        assert "open requests" in case.reply and first.status.value != "withdrawn"


def test_messages_the_parser_has_no_intents_for_are_parsed_as_before(world):
    case = world.orch.submit(world.intake.from_portal("F-101", "Can two faculty members swap their slots?"))
    assert "+intents" not in case.route and "+swap" not in case.route
    assert world.orch.parser.calls[-1] == ("Can two faculty members swap their slots?", 5)


def test_the_web_fallback_gives_no_intents_for_a_model_without_them():
    from web.fallback import Fallback, ModelLink

    class Old:  # a model trained on the older format: parse only
        def parse(self, request):
            return "plain"

    class Link(ModelLink):
        def __init__(self):
            self.client, self.up = object(), True

        def usable(self):
            return True

    f = Fallback(Link(), lambda c: Old(), backup=Old(), name="parser")
    assert f.compile("request") is None and f.parse("request") == "plain"


# -- club bookings, clash reports and the office acting on a message -----------------------------


def _future(weekday: int) -> str:
    from datetime import date, timedelta

    d = date.today() + timedelta(days=10)
    return (d + timedelta(days=(weekday - d.weekday()) % 7)).isoformat()


def published_semester(world, wait: float = 180):
    """The semester timetable the club desk checks against, once the start-up build has published it."""
    import time

    world.semester.bootstrap_demo()  # what a World with its start-up history does; this one has none
    end = time.time() + wait
    while world.semester.published() is None and time.time() < end:
        if world.semester.status().get("error"):  # the build ends just before it publishes: wait for either
            break
        time.sleep(0.5)
    v = world.semester.published()
    assert v is not None, f"no semester timetable published: {world.semester.status()}"
    return v


def test_a_student_books_a_room_for_a_club_through_the_desk(world):
    published_semester(world)
    case = submit(world, "ST-G-01", "Room for the Quiz Club, Thursday 7 to 9 pm, 30 of us?", IntentOutput(
        action="compile", intents=[ClubBooking(club="Quiz Club", activity="quiz night", date=_future(3),
                                               start="19:00", end="21:00", attendees=30)]))
    assert "+club" in case.route and case.status.value in ("answered", "forwarded"), case.reply
    assert case.reply.startswith(("Booked:", "Not booked", "Your request")), case.reply
    booked = world.semester.state.bookings[-1]
    assert booked.request.requested_by == "ST-G-01" and booked.request.start == "19:00"


def test_a_club_booking_without_times_asks_and_big_events_go_to_the_office(world):
    case = submit(world, "ST-G-02", "Can the Drama Club get a room next week?", IntentOutput(
        action="compile", intents=[ClubBooking(club="Drama Club", day="Fri", period=Period(relative="next_week"))]))
    assert case.status.value == "clarification_requested" and "what time" in case.reply

    seen = []
    world.orch.club_desk, desk = (lambda ref, r: (seen.append(ref), ("Sent to the office.", "needs_approval"))[1]), \
        world.orch.club_desk
    try:
        case = submit(world, "ST-G-02", "The fest needs the auditorium on Friday next week, 6 to 9 pm, 400 people.",
                      IntentOutput(action="compile", intents=[ClubBooking(
                          activity="fest", day="Fri", period=Period(relative="next_week"), start="18:00", end="21:00",
                          attendees=400, room="auditorium")]))
    finally:
        world.orch.club_desk = desk
    assert case.status.value == "forwarded" and seen[0].weeks == [6] and seen[0].room == "auditorium"


def _overlap(world):
    """Two sessions of different sections' courses at the same time (students taking both would clash)."""
    tt = world.store.current_version().assignment
    inst = world.instance
    placed = [s for s in inst.sessions if s.id in tt]
    for a in placed:
        for b in placed:
            if (a.course < b.course and not set(a.groups) & set(b.groups) and tt[a.id].day == tt[b.id].day
                    and tt[a.id].slot == tt[b.id].slot and a.duration == b.duration == 1):
                return a, b, tt
    raise AssertionError("no two sections' classes at the same time")


def test_a_real_clash_is_forwarded_with_both_classes_and_a_false_one_answered(world):
    a, b, tt = _overlap(world)
    p = tt[a.id]
    real = submit(world, f"ST-{a.groups[0]}", "Our two classes clash on that day.", IntentOutput(
        action="investigate", intents=[ClashReport(time=TimeSpec(days=[p.day], at=[clock(p.slot)]),
                                                   courses=[a.course, b.course])]))
    assert "+clash" in real.route and real.status.value == "forwarded" and "same time" in real.reply

    other = next(s for s in world.instance.sessions if s.id in tt and s.course != a.course
                 and (tt[s.id].day, tt[s.id].slot) != (p.day, p.slot) and s.duration == 1
                 and not any((tt[x.id].day, tt[x.id].slot) == (tt[s.id].day, tt[s.id].slot)
                             for x in world.instance.sessions if x.course == a.course and x.id in tt))
    false = submit(world, f"ST-{a.groups[0]}", "These two clash.", IntentOutput(
        action="investigate", intents=[ClashReport(courses=[a.course, other.course])]))
    assert false.status.value == "answered" and "do not overlap" in false.reply


def test_the_office_acts_on_a_forwarded_message(world):
    a, b, tt = _overlap(world)
    p = tt[a.id]
    report = submit(world, f"ST-{a.groups[0]}", "Clash between our classes, please help.", IntentOutput(
        action="investigate", intents=[ClashReport(time=TimeSpec(days=[p.day], at=[clock(p.slot)]),
                                                   courses=[a.course, b.course])]))
    assert report.status.value == "forwarded"
    r = world.intake.from_portal("C-TT", f"Dr. {a.faculty} is unavailable on {FULL_DAY[p.day]} at {hour(p.slot)}.")
    world.orch.act_on(report.id, "C-TT", r)
    assert report.handled["acting"] == r.id and r.id in report.reply
    with pytest.raises(ValueError):  # answered once
        world.orch.act_on(report.id, "C-TT", r)
    case = world.orch.submit(r)
    assert case.request.role.value == "coordinator" and case.status.value != "refused", case.reply
