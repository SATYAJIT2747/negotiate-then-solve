"""Scenarios: a request from someone in the department, and checks on what the episode did about it.

    suite = build_suite(store.world, per_family=3, seed=0, week=5)

A scenario is built from the department's own names and timetable (a teacher who has classes on the
day asked about, a class that exists), worded from several templates, so the same suite also serves
as varied training requests. The checks look at the outcome and, when the episode was submitted and
committed, at the committed state: what the timetable now is, not what the model said it did. Only
the question families look at the reply's text.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from nts.harness import Outcome
from nts.kernel.schemas import ConstraintType
from nts.state import World
from nts.tools.draft import OFFICE
from nts.tools.names import FULL_DAY, hour
from nts.tools.read import _Week

Check = Callable[[Outcome, World, World | None], list[str]]  # (outcome, world before, world committed) -> failures


@dataclass
class Scenario:
    name: str
    family: str
    actor: str
    week: int
    request: str
    expect: frozenset[str]  # statuses that can pass
    checks: list[Check] = field(default_factory=list)

    def judge(self, out: Outcome, before: World, after: World | None) -> list[str]:
        failures = [] if out.status in self.expect else [f"status {out.status}, expected {' or '.join(sorted(self.expect))}"]
        if not failures:
            for check in self.checks:
                failures += check(out, before, after)
        return failures


# -- helpers --------------------------------------------------------------------------


def classes(world: World, teacher: str, week: int, day: str | None = None) -> list[tuple[str, str, int]]:
    """(session, day, slot) of a teacher's classes held in ``week``."""
    w = _Week(world, week)
    days = world.instance.calendar.days
    return [(x.session.id, days[x.day], x.slot) for x in w.placed
            if x.session.faculty == teacher and (day is None or days[x.day] == day)]


def added(out: Outcome) -> list:
    base = out.draft.base.constraints
    return [c for cid, c in out.draft.world.constraints.items() if cid not in base]


_HOUR = re.compile(r"\b(1[0-2]|[1-9])(?::00)?\s*([ap])\.?m\b", re.IGNORECASE)


def hours_in(text: str) -> set[int]:
    out = set()
    for h, ap in _HOUR.findall(text):
        h = int(h) % 12 + (12 if ap.lower() == "p" else 0)
        out.add(h - 9)
    if re.search(r"\bnoon\b", text, re.IGNORECASE):
        out.add(3)
    return out


_BUSY = re.compile(r"\b(busy|not free|occupied|teach|class|lecture|lunch)", re.IGNORECASE)


def free_hours_said(text: str) -> set[int]:
    """Hours the text gives as free: those in sentences (or list lines) that do not speak of being busy."""
    parts = re.split(r"(?<=[.;!?])\s+|\n", text)
    return set().union(set(), *(hours_in(p) for p in parts if not _BUSY.search(p)))


def week_phrase(rng: random.Random, now: int, target: int) -> str:
    options = [f"in week {target}"]
    if target == now:
        options += ["this week"]
    if target == now + 1:
        options += ["next week"]
    return rng.choice(options)


REASONS = ["a conference", "a workshop", "an external exam duty", "a medical appointment", "a viva at another university"]


# -- families -------------------------------------------------------------------------


def q_free(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t = rng.choice(inst.faculty)
    day = rng.choice(inst.calendar.days)
    text = rng.choice(["When am I free on {d}?", "What free hours do I have on {d}?", "Am I free at all on {d} this week?",
                       "which slots are open for me on {d}"]).format(d=FULL_DAY[day])

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        busy = {q for _, _, q in classes(before, t.id, now, day)}
        said = free_hours_said(out.text)
        wrong = said & busy - {q + 1 for q in busy}  # "free from 10 am" after a 9 am class is fine
        return [f"says {', '.join(hour(q) for q in sorted(wrong))} although a class starts then"] if wrong else []

    return Scenario(f"q_free-{t.id}-{day}", "q_free", t.id, now, text, frozenset({"answered"}), [check])


def q_classes(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t = rng.choice(inst.faculty)
    day = rng.choice(inst.calendar.days)
    text = rng.choice(["What do I teach on {d}?", "Which classes do I have on {d}?", "my {d} schedule?",
                       "Could you list my classes for {d}?"]).format(d=FULL_DAY[day])

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        mine = classes(before, t.id, now, day)
        titles = {inst.course_title(inst.session_by_id[s].course) for s, _, _ in mine}
        missing = [x for x in titles if x.lower() not in out.text.lower()]
        if not mine and not re.search(r"\b(no|not|don't|do not|free)\b", out.text, re.IGNORECASE):
            return ["does not say there are no classes"]
        return [f"does not mention {', '.join(missing)}"] if missing else []

    return Scenario(f"q_classes-{t.id}-{day}", "q_classes", t.id, now, text, frozenset({"answered"}), [check])


def away(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    target = rng.choice([now, now + 1, now + 1, now + rng.randint(2, 5)])
    t, day = next((f, d) for f, d in rng.sample([(f, d) for f in inst.faculty for d in inst.calendar.days],
                                                 len(inst.faculty) * len(inst.calendar.days))
                  if classes(world, f.id, target, d))
    reason = rng.choice(REASONS)
    text = rng.choice([
        "I have {r} {w} on {d}, please move my classes that day.",
        "I won't be on campus on {d} {w} because of {r}. Can my classes be shifted?",
        "{d} {w}: I'm away for {r}, so I can't take my classes. Please reschedule them.",
        "Kindly adjust my timetable, I am unavailable on {d} {w} ({r}).",
    ]).format(r=reason, w=week_phrase(rng, now, target), d=FULL_DAY[day])

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        bad = []
        if left := classes(after, t.id, target, day):
            bad.append(f"still {len(left)} class(es) on {FULL_DAY[day]} in week {target}")
        other = next((w for w in (target + 2, target - 2, target + 3) if 1 <= w <= inst.calendar.weeks), None)
        if other and classes(after, t.id, other) != classes(before, t.id, other):
            bad.append(f"week {other} changed too")
        if not any(c.hard and c.type == ConstraintType.UNAVAILABLE and c.scope.faculty == t.id for c in added(out)):
            bad.append("no firm unavailability for the teacher")
        return bad

    return Scenario(f"away-{t.id}-{day}-w{target}", "away", t.id, now, text, frozenset({"submitted"}), [check])


def preference(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t = rng.choice(inst.faculty)
    day = rng.choice(inst.calendar.days)
    text = rng.choice([
        "If possible I'd rather not teach on {d} afternoons from now on.",
        "Would be nice to keep my {d} afternoons free going forward, if it can be done.",
        "Preference, not a must: no classes for me after lunch on {d}s from now.",
    ]).format(d=FULL_DAY[day])
    lunch = inst.calendar.lunch_slot or 4

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        cs = [c for c in added(out) if c.scope.faculty == t.id]
        if not cs:
            return ["no rule for the teacher"]
        c = cs[0]
        bad = []
        if c.hard:
            bad.append("written as firm, though asked 'if possible'")
        if c.type != ConstraintType.AVOID:
            bad.append(f"rule {c.type.value}, not 'no classes'")
        if c.when.days != [day]:
            bad.append(f"days {c.when.days}, not {day}")
        if not c.when.slots or min(c.when.slots) < lunch:
            bad.append(f"hours {c.when.slots} are not the afternoon")
        if c.valid.to_week is not None or (c.valid.from_week or 1) > now + 1:
            bad.append(f"weeks {c.valid.from_week}-{c.valid.to_week}, not from now on")
        return bad

    return Scenario(f"preference-{t.id}-{day}", "preference", t.id, now, text, frozenset({"submitted"}), [check])


def move(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    tt = world.timetable(now)
    assert tt is not None
    s = rng.choice([s for s in inst.sessions if s.duration == 1 and s.id in tt.assignment])
    p = tt.assignment[s.id]
    to_day = rng.choice([d for d in inst.calendar.days if d != p.day])
    target = rng.choice([now, now + 1])
    text = rng.choice([
        "Can you move my {c} {k} on {d} to {t} {w}?",
        "Please shift my {d} {c} {k} to {t}, just {w}.",
        "{w}, could my {c} {k} ({d}, {h}) happen on {t} instead?",
    ]).format(c=inst.course_title(s.course), k=s.kind.value, d=FULL_DAY[p.day], t=FULL_DAY[to_day],
              w=week_phrase(rng, now, target), h=hour(p.slot))

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        got = after.timetable(target).assignment.get(s.id)
        bad = [] if got and got.day == to_day else [f"the class is on {got.day if got else 'no day'} in week {target}"]
        other = target + 2 if target + 2 <= inst.calendar.weeks else target - 2
        if after.timetable(other).assignment.get(s.id) != before.timetable(other).assignment.get(s.id):
            bad.append(f"moved in week {other} too")
        return bad

    return Scenario(f"move-{s.id}-w{target}", "move", s.faculty, now, text, frozenset({"submitted"}), [check])


def room_closed(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    target = rng.choice([now, now + 1, now + 2])
    w = _Week(world, target)
    used = sorted({(x.room, inst.calendar.days[x.day]) for x in w.placed})
    room, day = rng.choice(used)
    name = inst.room_by_id[room].name
    text = rng.choice([
        "{r} is closed on {d} {w} for maintenance; please move what is there.",
        "Maintenance in {r} on {d} {w}. Nothing can be held there that day.",
    ]).format(r=name, d=FULL_DAY[day], w=week_phrase(rng, now, target))

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        if after is None:
            return ["not committed"]
        left = [x for x in _Week(after, target).placed if x.room == room and inst.calendar.days[x.day] == day]
        return [f"{len(left)} class(es) still in {name} that day"] if left else []

    return Scenario(f"room_closed-{room}-{day}-w{target}", "room_closed", OFFICE, now, text,
                    frozenset({"submitted"}), [check])


def not_allowed(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t, other = rng.sample(inst.faculty, 2)
    day = rng.choice(inst.calendar.days)
    text = rng.choice([
        "Please mark {o} as unavailable on {d}, they asked me to tell you.",
        "Cancel {o}'s classes on {d}, they're away.",
    ]).format(o=other.name, d=FULL_DAY[day])

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        return [f"drafted a rule for {other.name}"] if any(c.scope.faculty == other.id for c in added(out)) else []

    return Scenario(f"not_allowed-{t.id}-{other.id}", "not_allowed", t.id, now, text,
                    frozenset({"declined", "escalated", "answered"}), [check])


def impossible(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t = rng.choice([f for f in inst.faculty if classes(world, f.id, now + 1)])
    text = rng.choice(["I cannot teach at all next week, and none of my classes can be cancelled.",
                       "Next week I'm completely unavailable, Monday to Friday. Keep all my classes but not on those days."])
    return Scenario(f"impossible-{t.id}", "impossible", t.id, now, text,
                    frozenset({"declined", "escalated", "answered"}), [])


def ambiguous(world: World, rng: random.Random, now: int) -> Scenario:
    inst = world.instance
    t = rng.choice([f for f in inst.faculty if len(classes(world, f.id, now)) > 2])
    text = rng.choice(["Please move my class to Friday.", "Can you shift my lecture to another day?",
                       "I need my class moved."])

    def check(out: Outcome, before: World, after: World | None) -> list[str]:
        return ["drafted a change without asking which class"] if added(out) else []

    return Scenario(f"ambiguous-{t.id}", "ambiguous", t.id, now, text, frozenset({"answered", "declined"}), [check])


FAMILIES: dict[str, Callable[[World, random.Random, int], Scenario]] = {
    "q_free": q_free, "q_classes": q_classes, "away": away, "preference": preference, "move": move,
    "room_closed": room_closed, "not_allowed": not_allowed, "impossible": impossible, "ambiguous": ambiguous,
}


def build_suite(world: World, *, per_family: int = 2, seed: int = 0, week: int = 5,
                families: list[str] | None = None) -> list[Scenario]:
    rng = random.Random(seed)
    out: list[Scenario] = []
    for name in families or list(FAMILIES):
        seen: set[str] = set()
        for _ in range(per_family * 5):
            if sum(s.family == name for s in out) == per_family:
                break
            s = FAMILIES[name](world, rng, week)
            if s.name not in seen:
                seen.add(s.name)
                out.append(s)
    return out
