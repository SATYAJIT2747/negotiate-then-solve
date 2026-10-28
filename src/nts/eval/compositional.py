"""Compositional requests: what one rule or one move cannot say, as a held-out suite and as families.

    python -m nts eval --suite compositional --store runs/agentic/eval.jsonl ...
    python -m nts eval --families window_move or_windows ... (V6_FAMILIES)

Each request needs one thing the v1-v5 draft tools do not say in one call: times of day that differ by day,
an exception to a period, alternatives, a first and second choice, a relation between classes, a limit other
than hours in a row, a parity of weeks. The checks look at the committed timetable over the weeks the request
covers (a standing request: this week to the end of term). A request the agent cannot do should not be
submitted: a submitted draft that misses what was asked is a silent wrong, the outcome these are for.

The suite (``compositional_suite``) is core-rework's fourteen hand-written messages
(``data/suites/compositional.jsonl``), reworded for Dr. Iyer (F-103) of ``runs/agentic/eval.jsonl`` in week
5: nothing on Friday, three classes on Thursday (11 am, 12 pm and the Database Systems practical at 2 pm), two
Database Systems lectures sharing days with Artificial Intelligence lectures. It is held out: the families
(``V6``, registered in ``scenarios.FAMILIES``) make the same kinds of request for any teacher, worded from
their own templates, and each is built only when the solver says what it asks can be done.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable

from nts.harness import Outcome
from nts.kernel.instance import Session
from nts.kernel.schemas import Constraint, ConstraintType, Tier, Validity, When
from nts.kernel.schemas import Scope as Who
from nts.people import Cast, Requester
from nts.state import World
from nts.tools.names import FULL_DAY, hour

from .scenarios import FAMILIES, Scenario, _solvable, classes

TEACHER, NOW = "F-103", 5
DB, AI = "Database Systems", "Artificial Intelligence"
MORNING, AFTERNOON = range(0, 4), range(5, 8)  # 9 am to 12 pm; 2 pm to 4 pm (1 pm is lunch)
PARTS = {"morning": MORNING, "afternoon": AFTERNOON}
FAMILY = "compositional"

Holds = Callable[[World, str, int], list[str]]  # (committed world, teacher, week) -> what is wrong that week


def _mine(world: World, teacher: str, week: int) -> list[tuple[str, str, set[int]]]:
    """(session, day, hours it takes) of the teacher's classes held in ``week``."""
    inst = world.instance
    return [(sid, day, set(range(q, q + inst.session_by_id[sid].duration))) for sid, day, q in classes(world, teacher, week)]


def _of(world: World, course: str, kind: str | None = None, teacher: str = TEACHER) -> list[str]:
    inst = world.instance
    return [s.id for s in inst.sessions if s.faculty == teacher and inst.course_title(s.course) == course
            and (kind is None or s.kind.value == kind)]


def _free(days: set[str], hours: range) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        return [f"week {week}: {sid} on {d} at {min(hs) + 9}:00" for sid, d, hs in _mine(world, teacher, week)
                if d in days and hs & set(hours)]
    return holds


def _at(sid: str, windows: list[tuple[str, range]]) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        got = next(((d, hs) for s, d, hs in _mine(world, teacher, week) if s == sid), None)
        if got and any(got[0] == d and got[1] <= set(hs) for d, hs in windows):
            return []
        return [f"week {week}: {sid} is {f'on {got[0]} at {min(got[1]) + 9}:00' if got else 'not held'}"]
    return holds


def _per_day(cap: int) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        n: dict[str, int] = defaultdict(int)
        for _, d, _hs in _mine(world, teacher, week):
            n[d] += 1
        return [f"week {week}: {k} classes on {d}" for d, k in n.items() if k > cap]
    return holds


def _days(cap: int) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        days = {d for _, d, _hs in _mine(world, teacher, week)}
        return [f"week {week}: teaches on {len(days)} days"] if len(days) > cap else []
    return holds


def _gap(world: World, teacher: str, week: int) -> list[str]:
    taken: dict[str, list[set[int]]] = defaultdict(list)
    for _, d, hs in _mine(world, teacher, week):
        taken[d].append(hs)
    bad = []
    for d, xs in taken.items():
        hours = sorted(q for hs in xs for q in hs)
        if any(max(hs) + 1 in hours for hs in xs):
            bad.append(f"week {week}: back-to-back classes on {d}")
    return bad


def _right_after(practical: str, lectures: list[str]) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        at = {s: (d, hs) for s, d, hs in _mine(world, teacher, week)}
        if practical not in at:
            return [f"week {week}: {practical} not held"]
        d, hs = at[practical]
        if any(lec in at and at[lec][0] == d and max(at[lec][1]) + 1 == min(hs) for lec in lectures):
            return []
        return [f"week {week}: the practical is not right after a lecture"]
    return holds


def _apart(xs: list[str], ys: list[str]) -> Holds:
    def holds(world: World, teacher: str, week: int) -> list[str]:
        at = {s: d for s, d, _hs in _mine(world, teacher, week)}
        both = sorted({at[x] for x in xs if x in at} & {at[y] for y in ys if y in at})
        return [f"week {week}: both on {', '.join(both)}"] if both else []
    return holds


def _cancelled(sid: str) -> Holds:
    return lambda world, teacher, week: [] if sid in world.timetable(week).cancelled else [f"week {week}: {sid} not cancelled"]


def _and(*hs: Holds) -> Holds:
    return lambda world, teacher, week: [x for h in hs for x in h(world, teacher, week)]


def _by_week(each: dict[int, Holds]) -> Holds:
    return lambda world, teacher, week: each[week](world, teacher, week)


def _check(teacher: str, weeks: list[int], holds: Holds, *, same: list[int] = ()) -> Callable:
    """``holds`` in every one of ``weeks``; the teacher's classes in ``same`` as they were."""
    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        bad = [x for w in weeks for x in holds(after, teacher, w)]
        bad += [f"week {w} changed too" for w in same if classes(after, teacher, w) != classes(before, teacher, w)]
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
    t = TEACHER

    cases: list[tuple[str, str, Callable]] = [
        ("union_of_windows", "No classes on Monday mornings or Thursday afternoons, please.",
         _check(t, rest, _and(_free({"Mon"}, MORNING), _free({"Thu"}, AFTERNOON)))),
        ("except_weeks", "I can't teach on Wednesdays from now on, except in week 16.",
         _check(t, rest[:-1], _free({"Wed"}, range(8)))),
        ("or_over_days", f"Could you move my {DB} practical to Tuesday or Friday afternoon?",
         _check(t, rest, _at(db_p, [("Tue", AFTERNOON), ("Fri", AFTERNOON)]))),
        ("or_over_windows", f"Could you move my {DB} practical to Tuesday morning or Friday afternoon?",
         _check(t, rest, _at(db_p, [("Tue", MORNING), ("Fri", AFTERNOON)]))),
        ("ranked", "Ideally move my Wednesday 11 am lecture to Friday morning; if that doesn't work, Monday "
                   "afternoon is fine.",
         _check(t, rest, _at(wed, [("Fri", MORNING)] if first else [("Mon", AFTERNOON)]))),
        ("relation_after", f"The {DB} practical should be on the same day as one of its lectures, right after it.",
         _check(t, rest, _right_after(db_p, db_l))),
        ("relation_apart", f"Please don't put my {DB} lectures on the same day as my {AI} lectures.",
         _check(t, rest, _apart(db_l, ai_l))),
        ("limit_per_day", "No more than two classes a day for me, please.", _check(t, rest, _per_day(2))),
        ("limit_days_per_week", "I'd like to teach on at most three days a week.", _check(t, rest, _days(3))),
        ("limit_gap", "Leave at least an hour free between any two of my classes.", _check(t, rest, _gap)),
        ("fallback", f"Move my Wednesday lecture in week {NOW + 1} to Friday; if that isn't possible, cancel it.",
         _check(t, [NOW + 1], _at(wed, [("Fri", range(8))]) if moved else _cancelled(wed), same=[NOW, NOW + 2])),
        ("day_per_week", f"I'm away on Tuesday in week {NOW + 1} and on Thursday in week {NOW + 2}.",
         _check(t, [NOW + 1, NOW + 2], _by_week({NOW + 1: _free({"Tue"}, range(8)), NOW + 2: _free({"Thu"}, range(8))}),
                same=[NOW + 3])),
        ("parity", "On even weeks I can't teach on Thursday afternoons.",
         _check(t, [w for w in rest if w % 2 == 0], _free({"Thu"}, AFTERNOON))),
        ("binding_relation", f"Move my {AI} Tuesday 2 pm lecture to the morning, and keep it on a different day "
                             f"from my other {AI} lectures.",
         _check(t, rest, _and(_at(ai_tue, [(d, MORNING) for d in inst.calendar.days]),
                              _apart([ai_tue], [s for s in ai_l if s != ai_tue])))),
    ]
    cast = Cast(requester=Requester("Yes, please do exactly what my message says."))
    return [Scenario(f"{FAMILY}-{kind}", FAMILY, TEACHER, NOW, text, frozenset({"submitted"}), [check], cast=cast)
            for kind, text, check in cases]


# -- v6 families: the same kinds of request for any teacher (named only) ----------------------------------

CAST = Cast(requester=Requester("Yes, exactly as I wrote it, from now on unless I named the weeks."))


def _rule(kind: ConstraintType, teacher: str, *, scope: Who | None = None, when: When | None = None, **kw) -> Constraint:
    return Constraint(id=f"gold-{kind.value}", type=kind, hard=True, tier=Tier.OPERATIONAL, owner=teacher,
                      scope=scope or Who(faculty=teacher), when=when or When(), **kw)


def _ok(world: World, now: int, *rules: Constraint) -> bool:
    """What the request asks can be done this week and next (the solver's word, with every rule in force)."""
    rules = [r.model_copy(update={"id": f"gold-{i}"}) for i, r in enumerate(rules)]
    return all(_solvable(world, rules, w) for w in (now, now + 1))


def _rest(world: World, now: int) -> list[int]:
    return list(range(now, world.instance.calendar.weeks + 1))


def _label(world: World, s: Session, now: int) -> tuple[str, str, str, str]:
    """(course, kind, day, hour) of a class as it is this week."""
    p = world.timetable(now).assignment[s.id]
    return world.instance.course_title(s.course), s.kind.value, FULL_DAY[p.day], hour(p.slot)


def _days_text(days: list[str]) -> str:
    return " or ".join(FULL_DAY[d] for d in days)


def _teachers(world: World, rng: random.Random) -> list[str]:
    ids = [f.id for f in world.instance.faculty]
    rng.shuffle(ids)
    return ids


def _scenario(name: str, family: str, teacher: str, now: int, text: str, check: Callable) -> Scenario:
    return Scenario(name, family, teacher, now, text[0].upper() + text[1:], frozenset({"submitted"}), [check], cast=CAST)


def _a_class(world: World, rng: random.Random, now: int, teacher: str | None = None) -> Session:
    tt = world.timetable(now)
    pool = [s for s in world.instance.sessions if s.id in tt.assignment and (teacher is None or s.faculty == teacher)]
    if not pool:
        raise LookupError("no class")
    return rng.choice(pool)


def _windows(world: World, s: Session, now: int, rng: random.Random, n: int) -> list[tuple[str, str]]:
    """``n`` (day, part) windows the class is not in now and fits into."""
    p = world.timetable(now).assignment[s.id]
    out = [(d, part) for d in world.instance.calendar.days for part, hs in PARTS.items()
           if len(hs) >= s.duration and not (d == p.day and p.slot in hs)]
    rng.shuffle(out)
    return out[:n]


def window_move(world: World, rng: random.Random, now: int) -> Scenario:
    """A class to a part of a day, or to that part of either of two days."""
    for _ in range(20):
        s = _a_class(world, rng, now)
        p = world.timetable(now).assignment[s.id]
        part = rng.choice(list(PARTS))
        days = rng.sample([d for d in world.instance.calendar.days if d != p.day], rng.choice([1, 2]))
        gold = _rule(ConstraintType.PREFER, s.faculty, scope=Who(session=s.id), when=When(days=days, slots=list(PARTS[part])))
        if _ok(world, now, gold):
            break
    else:
        raise LookupError("no class that fits a window")
    c, k, d, h = _label(world, s, now)
    text = rng.choice([
        "Could you move my {c} {k} ({d}, {h}) to {days} {part}?",
        "{days} {part} would suit my {d} {h} {c} {k} better. Can it move there?",
        "please shift my {c} {k} from {d} {h} to the {part} on {days}",
    ]).format(c=c, k=k, d=d, h=h, days=_days_text(days), part=part)
    return _scenario(f"window_move-{s.id}-{''.join(days)}-{part}", "window_move", s.faculty, now, text,
                     _check(s.faculty, _rest(world, now), _at(s.id, [(x, PARTS[part]) for x in days])))


def or_windows(world: World, rng: random.Random, now: int) -> Scenario:
    """A class to one of two windows on different days and parts ("Tuesday morning or Friday afternoon")."""
    for _ in range(20):
        s = _a_class(world, rng, now)
        ws = _windows(world, s, now, rng, 6)
        pair = next(((a, b) for a in ws for b in ws if a[0] != b[0] and a[1] != b[1]), None)
        if pair is None:
            continue
        either = Constraint(id="gold-any", type=ConstraintType.ANY_OF, hard=True, tier=Tier.OPERATIONAL,
                            owner=s.faculty, scope=Who(faculty=s.faculty),
                            members=[_rule(ConstraintType.PREFER, s.faculty, scope=Who(session=s.id),
                                           when=When(days=[d], slots=list(PARTS[part]))).model_copy(update={"id": f"g{i}"})
                                     for i, (d, part) in enumerate(pair)])
        if _ok(world, now, either):
            break
    else:
        raise LookupError("no class with two windows")
    (d1, p1), (d2, p2) = pair
    c, k, d, h = _label(world, s, now)
    text = rng.choice([
        "Could you move my {c} {k} to {a} {pa} or {b} {pb}?",
        "my {d} {h} {c} {k}: either {a} {pa} or {b} {pb} works for me, whichever fits",
    ]).format(c=c, k=k, d=d, h=h, a=FULL_DAY[d1], pa=p1, b=FULL_DAY[d2], pb=p2)
    return _scenario(f"or_windows-{s.id}-{d1}{p1}-{d2}{p2}", "or_windows", s.faculty, now, text,
                     _check(s.faculty, _rest(world, now), _at(s.id, [(d1, PARTS[p1]), (d2, PARTS[p2])])))


def ranked_move(world: World, rng: random.Random, now: int) -> Scenario:
    """Ideally one window, else another: the first if the solver can, else the second."""
    for _ in range(20):
        s = _a_class(world, rng, now)
        ws = _windows(world, s, now, rng, 6)
        pair = next(((a, b) for a in ws for b in ws if a[0] != b[0]), None)
        if pair is None:
            continue
        gold = [_rule(ConstraintType.PREFER, s.faculty, scope=Who(session=s.id), when=When(days=[d], slots=list(PARTS[pt])))
                for d, pt in pair]
        first = _ok(world, now, gold[0])
        if first or _ok(world, now, gold[1]):
            break
    else:
        raise LookupError("no class with a first or second choice")
    (d1, p1), (d2, p2) = pair
    c, k, d, h = _label(world, s, now)
    text = rng.choice([
        "Ideally move my {d} {h} {c} {k} to {a} {pa}; if that doesn't work, {b} {pb} is fine.",
        "I'd most like my {c} {k} on {a} {pa}. Failing that, {b} {pb}.",
    ]).format(c=c, k=k, d=d, h=h, a=FULL_DAY[d1], pa=p1, b=FULL_DAY[d2], pb=p2)
    want = [(d1, PARTS[p1])] if first else [(d2, PARTS[p2])]
    return _scenario(f"ranked_move-{s.id}-{d1}{p1}-{d2}{p2}", "ranked_move", s.faculty, now, text,
                     _check(s.faculty, _rest(world, now), _at(s.id, want)))


def _per_day_now(world: World, teacher: str, now: int) -> dict[str, int]:
    n: dict[str, int] = defaultdict(int)
    for _, d, _q in classes(world, teacher, now):
        n[d] += 1
    return n


def day_cap(world: World, rng: random.Random, now: int) -> Scenario:
    """At most n classes a day, where some day has more now."""
    for t in _teachers(world, rng):
        n = _per_day_now(world, t, now)
        cap = max(n.values(), default=0) - 1
        if cap >= 1 and _ok(world, now, _rule(ConstraintType.LOAD, t, measure="classes_per_day", limit=cap)):
            break
    else:
        raise LookupError("no teacher to limit")
    word = {1: "one", 2: "two", 3: "three", 4: "four"}.get(cap, str(cap))
    text = rng.choice(["No more than {n} classes a day for me, please.",
                       "could you make sure I never have more than {n} classes on one day?",
                       "I can manage at most {n} classes in a day from now on."]).format(n=word)
    return _scenario(f"day_cap-{t}-{cap}", "day_cap", t, now, text, _check(t, _rest(world, now), _per_day(cap)))


def days_cap(world: World, rng: random.Random, now: int) -> Scenario:
    """At most n teaching days a week, one fewer than now."""
    for t in _teachers(world, rng):
        cap = len(_per_day_now(world, t, now)) - 1
        if cap >= 2 and _ok(world, now, _rule(ConstraintType.LOAD, t, measure="days_per_week", limit=cap)):
            break
    else:
        raise LookupError("no teacher to limit")
    word = {2: "two", 3: "three", 4: "four"}.get(cap, str(cap))
    text = rng.choice(["I'd like to teach on at most {n} days a week.",
                       "Can my classes be packed into {n} days a week, at most?",
                       "please keep my teaching to no more than {n} days each week"]).format(n=word)
    return _scenario(f"days_cap-{t}-{cap}", "days_cap", t, now, text, _check(t, _rest(world, now), _days(cap)))


def free_gap(world: World, rng: random.Random, now: int) -> Scenario:
    """An hour free between any two classes, where two are back to back now."""
    for t in _teachers(world, rng):
        if _gap(world, t, now) and _ok(world, now, _rule(ConstraintType.LOAD, t, measure="gap", limit=1)):
            break
    else:
        raise LookupError("no teacher with back-to-back classes")
    text = rng.choice(["Leave at least an hour free between any two of my classes.",
                       "I need a break of an hour or more between my classes, please; no back-to-back teaching.",
                       "could my classes never run back to back? an hour's gap between them would help"])
    return _scenario(f"free_gap-{t}", "free_gap", t, now, text, _check(t, _rest(world, now), _gap))


def after_lecture(world: World, rng: random.Random, now: int) -> Scenario:
    """A practical on the same day as one of its course's lectures, right after it."""
    inst = world.instance
    tt = world.timetable(now)
    pracs = [s for s in inst.sessions if s.kind.value == "practical" and s.id in tt.assignment]
    rng.shuffle(pracs)
    for p in pracs:
        lectures = [s.id for s in inst.sessions if s.course == p.course and s.kind.value == "lecture"
                    and s.faculty == p.faculty]
        gold = _rule(ConstraintType.RELATE, p.faculty, sessions=[p.id], others=lectures, relation="right_after",
                     any_other=True)
        if lectures and _right_after(p.id, lectures)(world, p.faculty, now) and _ok(world, now, gold):
            break
    else:
        raise LookupError("no practical to place")
    c = inst.course_title(p.course)
    text = rng.choice(["The {c} practical should be on the same day as one of its lectures, right after it.",
                       "Could my {c} practical follow straight on from one of the {c} lectures, same day?",
                       "please schedule the {c} practical immediately after one of its lectures"]).format(c=c)
    return _scenario(f"after_lecture-{p.id}", "after_lecture", p.faculty, now, text,
                     _check(p.faculty, _rest(world, now), _right_after(p.id, lectures)))


def courses_apart(world: World, rng: random.Random, now: int) -> Scenario:
    """One course's lectures never on a day with another course's lectures (of the same teacher)."""
    inst = world.instance
    for t in _teachers(world, rng):
        courses = sorted({s.course for s in inst.sessions if s.faculty == t})
        rng.shuffle(courses)
        found = None
        for a in courses:
            for b in courses:
                if a == b:
                    continue
                xs = [s.id for s in inst.sessions if s.course == a and s.faculty == t and s.kind.value == "lecture"]
                ys = [s.id for s in inst.sessions if s.course == b and s.faculty == t and s.kind.value == "lecture"]
                gold = _rule(ConstraintType.RELATE, t, sessions=xs, others=ys, relation="same_day", negate=True)
                if xs and ys and _apart(xs, ys)(world, t, now) and _ok(world, now, gold):
                    found = a, b, xs, ys
                    break
            if found:
                break
        if found:
            break
    else:
        raise LookupError("no two courses to keep apart")
    a, b, xs, ys = found
    text = rng.choice(["Please don't put my {a} lectures on the same day as my {b} lectures.",
                       "I'd rather not teach {a} and {b} lectures on the same day. Can they be on separate days?",
                       "keep my {a} lectures and my {b} lectures on different days please"]
                      ).format(a=inst.course_title(a), b=inst.course_title(b))
    return _scenario(f"courses_apart-{a}-{b}", "courses_apart", t, now, text, _check(t, _rest(world, now), _apart(xs, ys)))


def _busy_window(world: World, rng: random.Random, now: int, *, whole: bool = False):
    """(teacher, day, part or None) where the teacher has a class now."""
    for t in _teachers(world, rng):
        mine = _mine(world, t, now)
        rng.shuffle(mine)
        for _, d, hs in mine:
            for part, window in PARTS.items():
                if hs & set(window):
                    return t, d, (None if whole else part)
    raise LookupError("no teacher with classes")


def even_weeks(world: World, rng: random.Random, now: int) -> Scenario:
    """Unavailable at a part of a day (or all day) in even or odd weeks only."""
    weeks = world.instance.calendar.weeks
    for _ in range(10):
        t, d, part = _busy_window(world, rng, now, whole=rng.random() < 0.3)
        parity = rng.choice(["even", "odd"])
        hit = [w for w in range(now, weeks + 1) if w % 2 == (parity == "odd")]
        hours = PARTS[part] if part else range(8)
        gold = _rule(ConstraintType.UNAVAILABLE, t, when=When(days=[d], slots=list(hours), weeks=hit))
        if all(_solvable(world, [gold], w) for w in hit[:2]):
            break
    else:
        raise LookupError("no window to keep free")
    other = next(w for w in range(now, weeks + 1) if w not in hit)
    when = f"{FULL_DAY[d]} {part}s" if part else f"{FULL_DAY[d]}s"
    text = rng.choice(["On {p} weeks I can't teach on {w}.", "I'm not available on {w} in {p}-numbered weeks.",
                       "{w} are out for me every {p} week from now on"]).format(p=parity, w=when)
    return _scenario(f"even_weeks-{t}-{d}{part or ''}-{parity}", "even_weeks", t, now, text,
                     _check(t, hit, _free({d}, hours), same=[other]))


def except_week(world: World, rng: random.Random, now: int) -> Scenario:
    """Unavailable on a day from now on, except one week."""
    weeks = world.instance.calendar.weeks
    for _ in range(10):
        t, d, _part = _busy_window(world, rng, now, whole=True)
        skip = rng.randint(now + 2, weeks)
        gold = _rule(ConstraintType.UNAVAILABLE, t, when=When(days=[d]), valid=Validity(from_week=now))
        if _ok(world, now, gold):
            break
    else:
        raise LookupError("no day to keep free")
    text = rng.choice(["I can't teach on {d}s from now on, except in week {x}.",
                       "From now on {d}s are out for me (week {x} is the one exception).",
                       "please keep my {d}s free for the rest of term, apart from week {x}"]).format(d=FULL_DAY[d], x=skip)
    return _scenario(f"except_week-{t}-{d}-{skip}", "except_week", t, now, text,
                     _check(t, [w for w in _rest(world, now) if w != skip], _free({d}, range(8)), same=[skip]))


def two_windows_off(world: World, rng: random.Random, now: int) -> Scenario:
    """No classes in two different windows ("Monday mornings or Friday afternoons")."""
    for _ in range(10):
        t, d1, p1 = _busy_window(world, rng, now)
        d2 = rng.choice([d for d in world.instance.calendar.days if d != d1])
        p2 = "afternoon" if p1 == "morning" else "morning"
        gold = [_rule(ConstraintType.AVOID, t, when=When(days=[d], slots=list(PARTS[p]))) for d, p in ((d1, p1), (d2, p2))]
        if _ok(world, now, *gold):
            break
    else:
        raise LookupError("no windows to keep free")
    text = rng.choice(["No classes on {a} {pa}s or {b} {pb}s, please.",
                       "Could you keep my {a} {pa}s and {b} {pb}s free of classes from now on?"]
                      ).format(a=FULL_DAY[d1], pa=p1, b=FULL_DAY[d2], pb=p2)
    return _scenario(f"two_windows_off-{t}-{d1}{p1}-{d2}{p2}", "two_windows_off", t, now, text,
                     _check(t, _rest(world, now), _and(_free({d1}, PARTS[p1]), _free({d2}, PARTS[p2]))))


def move_apart(world: World, rng: random.Random, now: int) -> Scenario:
    """A lecture to a part of the day (any day), kept on a different day from its course's other lectures."""
    inst = world.instance
    for _ in range(20):
        s = _a_class(world, rng, now)
        if s.kind.value != "lecture":
            continue
        part = rng.choice(list(PARTS))
        sibs = [x.id for x in inst.sessions if x.course == s.course and x.faculty == s.faculty and x.kind == s.kind
                and x.id != s.id]
        p = world.timetable(now).assignment[s.id]
        if not sibs or p.slot in PARTS[part]:
            continue
        gold = [_rule(ConstraintType.PREFER, s.faculty, scope=Who(session=s.id), when=When(slots=list(PARTS[part]))),
                _rule(ConstraintType.RELATE, s.faculty, sessions=[s.id], others=sibs, relation="different_day")]
        if _ok(world, now, *gold):
            break
    else:
        raise LookupError("no lecture to move")
    c, k, d, h = _label(world, s, now)
    text = rng.choice(["Move my {c} {d} {h} lecture to the {part}, and keep it on a different day from my other {c} lectures.",
                       "Could my {d} {h} {c} lecture move to the {part}? Not on a day I already teach {c}, though."]
                      ).format(c=c, d=d, h=h, part=part)
    return _scenario(f"move_apart-{s.id}-{part}", "move_apart", s.faculty, now, text,
                     _check(s.faculty, _rest(world, now),
                            _and(_at(s.id, [(x, PARTS[part]) for x in inst.calendar.days]), _apart([s.id], sibs))))


V6: dict[str, Callable[[World, random.Random, int], Scenario]] = {
    "window_move": window_move, "or_windows": or_windows, "ranked_move": ranked_move, "day_cap": day_cap,
    "days_cap": days_cap, "free_gap": free_gap, "after_lecture": after_lecture, "courses_apart": courses_apart,
    "even_weeks": even_weeks, "except_week": except_week, "two_windows_off": two_windows_off, "move_apart": move_apart,
}
FAMILIES.update(V6)
V6_FAMILIES = list(V6)
