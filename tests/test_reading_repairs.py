"""Near misses of the intents reader that the checks put right (language.checks), from the head-to-head
(runs/head_to_head/v1): a move to a part of the day with the day implied, and a cancellation whose day the
reading left out."""

from datetime import UTC, datetime

import pytest

from core.schemas import Channel, Request, Role
from language.checks import check_reading
from language.intents import Cancel, ClassRef, IntentOutput, Move, Period, TimeSpec, Unavailable


@pytest.fixture(scope="module")
def world():
    from web.world import World

    return World(parser_mode="offline", seed_history=False, deadline=60)


def _read(world, sender, text, intent):
    r = Request(id="x", channel=Channel.PORTAL, sender_id=sender, role=Role.FACULTY, raw_text=text,
                received_at=datetime.now(UTC))
    return check_reading(world.instance, r, IntentOutput(action="compile", intents=[intent]),
                         world.store.current_version().assignment)


def test_a_move_to_the_afternoon_stays_on_the_day_the_class_is_on(world):
    who = world.instance.faculty[0].id
    move = Move(which=ClassRef(course="Machine Learning", day="Fri", time="09:00"),
                to=TimeSpec(days=["Tue", "Wed", "Thu"], part="afternoon"))
    c = _read(world, who, "Move my Friday 9 am Machine Learning class to the afternoon in week 6.", move)
    assert c.output.intents[0].to.days == ["Fri"] and c.output.intents[0].to.part == "afternoon"
    named = move.model_copy(update={"to": TimeSpec(days=["Tue"], part="afternoon")})
    c = _read(world, who, "Move my Friday 9 am Machine Learning class to Tuesday afternoon.", named)
    assert c.output.intents[0].to.days == ["Tue"]  # a day the message names is kept


def test_a_cancellation_gets_back_the_day_the_message_names(world):
    inst, tt = world.instance, world.store.current_version().assignment
    lectures = [s for s in inst.sessions if inst.course_title(s.course) == "Artificial Intelligence"
                and s.kind.value == "lecture" and s.id in tt]
    teacher = lectures[0].faculty
    mine = [s for s in lectures if s.faculty == teacher]
    assert len(mine) > 1, "the test needs a course with more than one lecture a week"
    day = tt[mine[0].id].day
    from language.corpus import FULL_DAY

    cancel = Cancel(which=ClassRef(course="Artificial Intelligence", kind="lecture"),
                    period=Period(relative="next_week"))
    c = _read(world, teacher, f"Cancel my AI lecture next {FULL_DAY[day]}.", cancel)
    assert c.output.intents[0].which.day == day
    c = _read(world, teacher, "Cancel my AI lecture next week.", cancel)
    assert c.output.intents[0].which.day is None  # no day named: still asked which


def test_alternate_weeks_from_a_week_keep_their_parity(world):  # intents v13b, suite week_patterns
    who = world.instance.faculty[0].id
    dropped = Unavailable(time=TimeSpec(days=["Wed"]), period=Period(from_week=9))
    c = _read(world, who, "Alternate weeks from week 9: Wednesdays are out for me.", dropped)
    assert c.output.intents[0].period.parity == "odd" and c.output.intents[0].period.from_week == 9
    c = _read(world, who, "Every other week, starting in week 8, I can't do Wednesday.", dropped.model_copy(
        update={"period": Period(from_week=8)}))
    assert c.output.intents[0].period.parity == "even"
    c = _read(world, who, "From week 9 on, Wednesdays are out for me.", dropped)
    assert c.output.intents[0].period.parity is None  # every week: nothing to repair
