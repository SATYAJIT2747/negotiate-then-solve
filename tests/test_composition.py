"""Load limits, relations between classes and choices between constraints: the solver meets them and the
validator, which reads the same semantics independently of the CP-SAT encoding, agrees."""

import pytest

from core.generator import generate_department
from core.instance import policy_constraints
from core.schemas import Constraint, ConstraintType, Scope, Tier, When
from core.semantics import satisfied
from core.solver import TimetableSolver, neighbourhood
from core.validator import verify_timetable

T = ConstraintType


@pytest.fixture(scope="module")
def small():
    inst = generate_department(seed=1, n_faculty=8, n_groups=3, n_lecture_rooms=4, n_labs=3)
    base = TimetableSolver(inst, policy_constraints(inst), time_limit=20).solve()
    assert base.ok
    return inst, policy_constraints(inst), base.assignment


def _busiest(inst):
    """The teacher with the most classes, and their lecture sessions grouped by course."""
    by = {}
    for s in inst.sessions:
        by.setdefault(s.faculty, []).append(s)
    fid = max(by, key=lambda f: len(by[f]))
    courses = {}
    for s in by[fid]:
        if s.kind.value == "lecture":
            courses.setdefault(s.course, []).append(s.id)
    return fid, by[fid], courses


def _placed(inst, assignment):
    return {sid: (inst.day_index(p.day), p.slot, p.room) for sid, p in assignment.items()}


def _solve(inst, cons, extra, baseline):
    r = TimetableSolver(inst, [*cons, *extra], baseline=baseline, time_limit=30).solve()
    assert r.ok, r.status
    assert verify_timetable(inst, r.assignment, [*cons, *extra]) == []
    return r


def C(cid, type_, hard=True, **kw):
    return Constraint(id=cid, type=type_, hard=hard, tier=Tier.OPERATIONAL if hard else Tier.PREFERENCE, **kw)


@pytest.mark.parametrize("measure", ["classes_per_day", "hours_per_day", "days_per_week", "gap"])
def test_load_limits_are_kept(small, measure):
    """A limit the starting timetable breaks but a timetable can keep: the solver moves classes to keep it."""
    inst, cons, base = small
    fid, mine, _ = _busiest(inst)
    days = {}
    for s in mine:
        days.setdefault(base[s.id].day, []).append(s)
    hours = {d: sum(s.duration for s in v) for d, v in days.items()}
    n_days = len(inst.calendar.days)
    limit = {"classes_per_day": max(-(-len(mine) // n_days), max(map(len, days.values())) - 1),
             "hours_per_day": max(-(-sum(hours.values()) // n_days) + 1, max(hours.values()) - 1),
             "days_per_week": len(days) - 1, "gap": 1}[measure]
    c = C("L", T.LOAD, scope=Scope(faculty=fid), measure=measure, limit=limit)
    r = _solve(inst, cons, [c], base)
    assert satisfied(inst, c, _placed(inst, r.assignment))


@pytest.mark.parametrize("relation, any_other, negate, amount", [
    ("different_day", False, False, None), ("same_day", False, False, None), ("day_gap", False, False, 1),
    ("same_hour", False, False, None), ("same_room", False, False, None), ("right_after", True, False, None),
    ("right_after", False, True, None)])
def test_relations_between_classes_hold(small, relation, any_other, negate, amount):
    inst, cons, base = small
    _, _, courses = _busiest(inst)
    lectures = max(courses.values(), key=len)
    first, rest = lectures[0], lectures[1:]
    if relation in ("different_day", "same_hour", "same_room", "day_gap"):  # each lecture against the others
        c = C("R", T.RELATE, sessions=lectures, relation=relation, amount=amount)
    else:  # the first lecture against the rest
        c = C("R", T.RELATE, sessions=[first], others=rest, relation=relation, any_other=any_other, negate=negate)
    r = _solve(inst, cons, [c], base)
    assert satisfied(inst, c, _placed(inst, r.assignment))


def test_the_validator_sees_a_broken_relation(small):
    inst, _, base = small
    _, _, courses = _busiest(inst)
    lectures = max(courses.values(), key=len)
    same = dict(base)
    for sid in lectures[1:]:  # every lecture on the first one's day
        same[sid] = same[sid].model_copy(update={"day": base[lectures[0]].day})
    c = C("R", T.RELATE, sessions=lectures, relation="different_day")
    assert not satisfied(inst, c, _placed(inst, same))


def test_one_of_two_times_is_met_and_a_ranked_choice_takes_the_first_it_can(small):
    inst, cons, base = small
    fid, mine, _ = _busiest(inst)
    sid = mine[0].id
    tue = C("A1", T.PREFER, scope=Scope(session=sid), when=When(days=["Tue"], slots=[0, 1, 2, 3]))
    thu = C("A2", T.PREFER, scope=Scope(session=sid), when=When(days=["Thu"], slots=[5, 6, 7]))
    either = C("A", T.ANY_OF, members=[tue, thu])
    r = _solve(inst, cons, [either], base)
    p = r.assignment[sid]
    assert (p.day, p.slot in (0, 1, 2, 3)) == ("Tue", True) or (p.day, p.slot in (5, 6, 7)) == ("Thu", True)
    first = tue.model_copy(update={"hard": False, "tier": Tier.PREFERENCE})
    second = thu.model_copy(update={"hard": False, "tier": Tier.PREFERENCE})
    ranked = C("K", T.RANKED, hard=False, members=[first, second])
    r = TimetableSolver(inst, [*cons, ranked], baseline=base, time_limit=30).solve()
    assert r.ok and r.assignment[sid].day == "Tue" and "K" not in r.soft_violations


def test_a_choice_is_one_constraint_negotiation_can_switch_off(small):
    inst, cons, base = small
    _, mine, _ = _busiest(inst)
    sid = mine[0].id
    # both members impossible together with a hard unavailability on every day: the choice is in the core
    away = C("U", T.UNAVAILABLE, scope=Scope(session=sid), when=When(days=["Tue", "Thu"]))
    tue = C("A1", T.PREFER, scope=Scope(session=sid), when=When(days=["Tue"]))
    thu = C("A2", T.PREFER, scope=Scope(session=sid), when=When(days=["Thu"]))
    s = TimetableSolver(inst, [*cons, away, C("A", T.ANY_OF, members=[tue, thu])], baseline=base, time_limit=30)
    core = s.core(s.hard_ids)
    assert core is not None and {"U", "A"} <= set(core)
    assert s.is_feasible([i for i in s.hard_ids if i != "A"])


def test_a_soft_load_limit_costs_only_when_broken(small):
    inst, cons, base = small
    fid, _, _ = _busiest(inst)
    c = C("L", T.LOAD, hard=False, scope=Scope(faculty=fid), measure="days_per_week", limit=4)
    r = TimetableSolver(inst, [*cons, c], baseline=base, time_limit=30).solve()
    assert r.ok and ("L" in r.soft_violations) != satisfied(inst, c, _placed(inst, r.assignment))


def test_a_broken_relation_puts_its_classes_in_the_neighbourhood(small):
    inst, cons, base = small
    _, _, courses = _busiest(inst)
    lectures = max(courses.values(), key=len)
    same = dict(base)
    for sid in lectures[1:]:
        same[sid] = same[sid].model_copy(update={"day": base[lectures[0]].day})
    c = C("R", T.RELATE, sessions=lectures, relation="different_day")
    assert set(lectures) <= neighbourhood(inst, [*cons, c], same)


def test_choices_and_loads_are_checked_when_made():
    with pytest.raises(ValueError):
        C("A", T.ANY_OF, members=[C("A1", T.PREFER, scope=Scope(session="x"))])
    with pytest.raises(ValueError):
        C("L", T.LOAD, scope=Scope(faculty="F"), measure="hours_per_week", limit=8)
    with pytest.raises(ValueError):
        C("K", T.RANKED, members=[C("a", T.PREFER), C("b", T.PREFER)])  # ranked is soft
