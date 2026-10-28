"""Compositional requests: the fourteen hand-written messages of core-rework's ``data/suites/compositional.jsonl``,
reworded for one teacher of the agent's department and judged on what the agent committed.

    python -m nts eval --suite compositional --store runs/agentic/eval.jsonl ...

Each message needs one thing the draft tools do not say in one call: times of day that differ by day, an
exception to a period, alternatives, a first and second choice, a relation between classes, a limit other
than hours in a row, a fallback, a parity of weeks. The checks look at the committed timetable over the weeks
the message covers (a standing request: this week to the end of term). A request the agent cannot do
should not be submitted: a submitted draft that misses what was asked is a silent wrong (``silent_wrong``),
the outcome the suite is for.

The teacher is Dr. Iyer (F-103) of ``runs/agentic/eval.jsonl``, who teaches Database Systems and Artificial
Intelligence (three lectures and a practical each), seen from week 5: nothing on Friday, three classes on
Thursday (11 am, 12 pm and the Database Systems practical at 2 pm), two Database Systems lectures sharing days
with Artificial Intelligence lectures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from nts.harness import Outcome
from nts.kernel.schemas import Constraint, ConstraintType, Tier, Validity, When
from nts.kernel.schemas import Scope as Who
from nts.people import Cast, Requester
from nts.state import World

from .scenarios import Scenario, _solvable, classes

TEACHER, NOW = "F-103", 5
DB, AI = "Database Systems", "Artificial Intelligence"
MORNING, AFTERNOON = range(0, 4), range(5, 8)  # 9 am to 12 pm; 2 pm to 4 pm (1 pm is lunch)
FAMILY = "compositional"

Holds = Callable[[World, int], list[str]]  # (committed world, week) -> what is wrong that week


def _mine(world: World, week: int) -> list[tuple[str, str, set[int]]]:
    """(session, day, hours it takes) of the teacher's classes held in ``week``."""
    inst = world.instance
    return [(sid, day, set(range(q, q + inst.session_by_id[sid].duration))) for sid, day, q in classes(world, TEACHER, week)]


def _of(world: World, course: str, kind: str | None = None) -> list[str]:
    inst = world.instance
    return [s.id for s in inst.sessions if s.faculty == TEACHER and inst.course_title(s.course) == course
            and (kind is None or s.kind.value == kind)]


def _free(days: set[str], hours: range) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        return [f"week {week}: {sid} on {d} at {min(hs) + 9}:00" for sid, d, hs in _mine(world, week)
                if d in days and hs & set(hours)]
    return holds


def _at(sid: str, windows: list[tuple[str, range]]) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        got = next(((d, hs) for s, d, hs in _mine(world, week) if s == sid), None)
        if got and any(got[0] == d and got[1] <= set(hs) for d, hs in windows):
            return []
        return [f"week {week}: {sid} is {f'on {got[0]} at {min(got[1]) + 9}:00' if got else 'not held'}"]
    return holds


def _per_day(cap: int) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        n: dict[str, int] = defaultdict(int)
        for _, d, _hs in _mine(world, week):
            n[d] += 1
        return [f"week {week}: {k} classes on {d}" for d, k in n.items() if k > cap]
    return holds


def _days(cap: int) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        days = {d for _, d, _hs in _mine(world, week)}
        return [f"week {week}: teaches on {len(days)} days"] if len(days) > cap else []
    return holds


def _gap(world: World, week: int) -> list[str]:
    taken: dict[str, list[set[int]]] = defaultdict(list)
    for _, d, hs in _mine(world, week):
        taken[d].append(hs)
    bad = []
    for d, xs in taken.items():
        hours = sorted(q for hs in xs for q in hs)
        ends = [max(hs) for hs in xs]
        if any(e + 1 in hours for e in ends):
            bad.append(f"week {week}: back-to-back classes on {d}")
    return bad


def _right_after(practical: str, lectures: list[str]) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        at = {s: (d, hs) for s, d, hs in _mine(world, week)}
        if practical not in at:
            return [f"week {week}: {practical} not held"]
        d, hs = at[practical]
        if any(lec in at and at[lec][0] == d and max(at[lec][1]) + 1 == min(hs) for lec in lectures):
            return []
        return [f"week {week}: the practical is not right after a lecture"]
    return holds


def _apart(xs: list[str], ys: list[str]) -> Holds:
    def holds(world: World, week: int) -> list[str]:
        at = {s: d for s, d, _hs in _mine(world, week)}
        both = sorted({at[x] for x in xs if x in at} & {at[y] for y in ys if y in at})
        return [f"week {week}: both on {', '.join(both)}"] if both else []
    return holds


def _and(*hs: Holds) -> Holds:
    return lambda world, week: [x for h in hs for x in h(world, week)]


def _check(weeks: list[int], holds: Holds, *, same: list[int] = ()) -> Callable:
    """``holds`` in every one of ``weeks``; the teacher's classes in ``same`` as they were."""
    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        bad = [x for w in weeks for x in holds(after, w)]
        bad += [f"week {w} changed too" for w in same if classes(after, TEACHER, w) != classes(before, TEACHER, w)]
        return bad[:4] + ([f"... {len(bad) - 4} more"] if len(bad) > 4 else [])
    return check


def _want(sid: str, day: str, hours: range, owner: str) -> Constraint:
    return Constraint(id=f"want-{sid}", type=ConstraintType.PREFER, hard=True, tier=Tier.OPERATIONAL, owner=owner,
                      scope=Who(session=sid), when=When(days=[day], slots=list(hours)),
                      valid=Validity(from_week=NOW, to_week=None))


def compositional_suite(world: World) -> list[Scenario]:
    inst = world.instance
    weeks = inst.calendar.weeks
    rest = list(range(NOW, weeks + 1))
    db_l, db_p, ai_l = _of(world, DB, "lecture"), _of(world, DB, "practical")[0], _of(world, AI, "lecture")
    at = {sid: (d, q) for sid, d, q in classes(world, TEACHER, NOW)}
    wed = next(s for s in db_l if at[s][0] == "Wed")  # the Wednesday 11 am Database Systems lecture
    ai_tue = next(s for s in ai_l if at[s][0] == "Tue")  # the Tuesday 2 pm Artificial Intelligence lecture
    first = _solvable(world, [_want(wed, "Fri", MORNING, TEACHER)], NOW)
    moved = _solvable(world, [_want(wed, "Fri", range(8), TEACHER)], NOW + 1)

    cases: list[tuple[str, str, Callable]] = [
        ("union_of_windows", "No classes on Monday mornings or Thursday afternoons, please.",
         _check(rest, _and(_free({"Mon"}, MORNING), _free({"Thu"}, AFTERNOON)))),
        ("except_weeks", "I can't teach on Wednesdays from now on, except in week 16.",
         _check(rest[:-1], _free({"Wed"}, range(8)))),
        ("or_over_days", f"Could you move my {DB} practical to Tuesday or Friday afternoon?",
         _check(rest, _at(db_p, [("Tue", AFTERNOON), ("Fri", AFTERNOON)]))),
        ("or_over_windows", f"Could you move my {DB} practical to Tuesday morning or Friday afternoon?",
         _check(rest, _at(db_p, [("Tue", MORNING), ("Fri", AFTERNOON)]))),
        ("ranked", "Ideally move my Wednesday 11 am lecture to Friday morning; if that doesn't work, Monday "
                   "afternoon is fine.",
         _check(rest, _at(wed, [("Fri", MORNING)] if first else [("Mon", AFTERNOON)]))),
        ("relation_after", f"The {DB} practical should be on the same day as one of its lectures, right after it.",
         _check(rest, _right_after(db_p, db_l))),
        ("relation_apart", f"Please don't put my {DB} lectures on the same day as my {AI} lectures.",
         _check(rest, _apart(db_l, ai_l))),
        ("limit_per_day", "No more than two classes a day for me, please.", _check(rest, _per_day(2))),
        ("limit_days_per_week", "I'd like to teach on at most three days a week.", _check(rest, _days(3))),
        ("limit_gap", "Leave at least an hour free between any two of my classes.", _check(rest, _gap)),
        ("fallback", f"Move my Wednesday lecture in week {NOW + 1} to Friday; if that isn't possible, cancel it.",
         _check([NOW + 1], _at(wed, [("Fri", range(8))]) if moved else
                (lambda w, k: [] if wed in w.timetable(k).cancelled else [f"week {k}: {wed} not cancelled"]),
                same=[NOW, NOW + 2])),
        ("day_per_week", f"I'm away on Tuesday in week {NOW + 1} and on Thursday in week {NOW + 2}.",
         _check([NOW + 1, NOW + 2], lambda w, k: _free({"Tue" if k == NOW + 1 else "Thu"}, range(8))(w, k),
                same=[NOW + 3])),
        ("parity", "On even weeks I can't teach on Thursday afternoons.",
         _check([w for w in rest if w % 2 == 0], _free({"Thu"}, AFTERNOON))),
        ("binding_relation", f"Move my {AI} Tuesday 2 pm lecture to the morning, and keep it on a different day "
                             f"from my other {AI} lectures.",
         _check(rest, _and(_at(ai_tue, [(d, MORNING) for d in inst.calendar.days]),
                           _apart([ai_tue], [s for s in ai_l if s != ai_tue])))),
    ]
    cast = Cast(requester=Requester("Yes, please do exactly what my message says."))
    return [Scenario(f"{FAMILY}-{kind}", FAMILY, TEACHER, NOW, text, frozenset({"submitted"}), [check], cast=cast)
            for kind, text, check in cases]
