"""Messages as programs through the whole pipeline: a choice between times, a relation between classes, a
limit on the week, and a fallback that goes ahead when the first change cannot be made."""

from core.schemas import ConstraintType
from core.semantics import satisfied
from language.intents import (
    AnyOf,
    Cancel,
    ClassRef,
    Fallback,
    IntentOutput,
    Limit,
    Move,
    Period,
    Relate,
    TimeSpec,
    Unavailable,
)
from test_orchestrator import S, req, world  # noqa: F401 - the fixture


def _mine(sc, who):
    return [s for s in sc.instance.sessions if s.faculty == who]


def _program(*intents):
    return IntentOutput(action="compile", intents=list(intents))


def _placed(sc, assignment):
    return {sid: (sc.instance.day_index(p.day), p.slot, p.room) for sid, p in assignment.items()}


def test_a_choice_between_two_times_is_one_constraint_and_one_of_them_is_met(world):
    sc, orch, store, owners = world
    who = owners["R-A"]
    sid = _mine(sc, who)[0].id
    c = orch.submit(req("R-50", who, "choice"), form=_program(
        Move(id="a", which=ClassRef(session=sid), to=TimeSpec(days=["Tue"], part="morning")),
        Move(id="b", which=ClassRef(same="a"), to=TimeSpec(days=["Thu"], part="afternoon")),
        AnyOf(of=["a", "b"], hard=True)))
    assert c.status in (S.AWAITING_APPROVAL, S.PUBLISHED), c.reply
    (choice,) = c.constraints
    assert choice.type == ConstraintType.ANY_OF and len(choice.members) == 2
    timetable = c.proposal.assignment if c.proposal else store.current_version().assignment
    assert satisfied(sc.instance, choice, _placed(sc, timetable))


def test_a_relation_and_a_limit_are_solved_for(world):
    sc, orch, store, owners = world
    who = owners["R-B"]
    mine = _mine(sc, who)
    course = max({s.course for s in mine}, key=lambda k: sum(s.course == k for s in mine))
    c = orch.submit(req("R-51", who, "spread"), form=_program(
        Relate(which=ClassRef(course=course, owner=who), relation="different_day", hard=True),
        Limit(measure="classes_per_day", n=3, hard=True)))
    assert c.status in (S.AWAITING_APPROVAL, S.PUBLISHED), c.reply
    assert {x.type for x in c.constraints} == {ConstraintType.RELATE, ConstraintType.LOAD}
    timetable = c.proposal.assignment if c.proposal else store.current_version().assignment
    assert all(satisfied(sc.instance, x, _placed(sc, timetable)) for x in c.constraints)


def test_a_fallback_goes_ahead_when_the_first_change_cannot_be_made(world):
    sc, orch, store, owners = world
    who = owners["R-A"]
    sid = _mine(sc, who)[0].id
    week = 5
    away = orch.submit(req("R-52", who, "away"), form=_program(
        Unavailable(time=TimeSpec(days=["Thu"]), period=Period(weeks=[week]))))
    if away.status == S.AWAITING_APPROVAL:
        orch.approve(away.id, "C-TT")
    # moving a class to the Thursday they are away cannot be done: it is cancelled that week instead
    c = orch.submit(req("R-53", who, "fallback"), form=_program(
        Move(id="a", which=ClassRef(session=sid), to=TimeSpec(days=["Thu"]), period=Period(weeks=[week])),
        Cancel(id="b", which=ClassRef(same="a"), period=Period(weeks=[week])),
        Fallback(of=["a", "b"])))
    assert c.status == S.ANSWERED and c.parts, (c.status, c.reply)
    child = orch.cases[c.parts[0]]
    assert "fallback" in c.reply and child.routing.get("part_of") == c.id
    assert not any(x.source.request == c.id for x in store.constraints())  # the first change is not in force
