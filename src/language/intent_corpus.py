"""Requests for the intents the corpus has none of (``language.intents``).

Move a class named by where it is now, swap with a colleague, an extra
class, a cancelled class, a limit on hours in a row, taking back a request,
weeks named relative to now ("next week", "from week 10") and times between
two hours. Each example is a message, the sender's timetable it refers to
(random placements: only consistency with the message matters), the current
week and the gold ``IntentOutput``.

Every example is checked by compiling its gold: it must resolve to the
classes it was written about, or (for the requests that leave something
out) ask a question. Each generator's last wording is kept for the test
split, so the test measures wordings the model has not seen; val is a
random tenth of the rest.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from core.instance import Instance, Session, SessionKind
from core.schemas import Channel, Placement, Request, Role

from .corpus import FULL_DAY, hour
from .intent_compiler import clock, random_timetable
from .intents import (
    Avoid,
    Cancel,
    ClassRef,
    ExtraClass,
    IntentOutput,
    LoadLimit,
    Move,
    Period,
    Swap,
    TimeSpec,
    Unavailable,
    compile_intents,
)

KIND_WORD = {SessionKind.LECTURE: ["lecture", "class"], SessionKind.PRACTICAL: ["lab", "practical", "lab session"],
             SessionKind.TUTORIAL: ["tutorial"]}
_OPEN = ["", "", "Hi,\n", "Hello,\n", "Dear timetable office,\n"]
_CLOSE = ["", "", " Thanks.", " Thank you!", "\n\nRegards,\n{name}"]


@dataclass
class IntentExample:
    id: str
    generator: str
    template: int
    split: str
    request: Request
    gold: IntentOutput
    timetable: dict[str, Placement]
    current_week: int
    asks: bool = False  # the message leaves something out: the gold compiles to a question
    want: dict = field(default_factory=dict)  # what the gold must resolve to


@dataclass
class _Ctx:
    instance: Instance
    rng: random.Random
    sender: str
    timetable: dict[str, Placement]
    week: int

    @property
    def name(self) -> str:
        return self.instance.faculty_by_id[self.sender].name

    def mine(self, kind: SessionKind | None = None) -> list[Session]:
        return [s for s in self.instance.sessions if s.faculty == self.sender and (kind is None or s.kind == kind)]

    def at(self, s: Session) -> tuple[str, str, str]:
        """(day, clock, hour words) where ``s`` is now."""
        p = self.timetable[s.id]
        return p.day, clock(p.slot), hour(p.slot)

    def kind_word(self, s: Session) -> str:
        return self.rng.choice(KIND_WORD[s.kind])

    def title(self, s: Session) -> str:
        return self.instance.course_title(s.course)

    def other_day(self, *not_days: str) -> str:
        return self.rng.choice([d for d in self.instance.calendar.days if d not in not_days])

    def to_time(self, s: Session, day: str) -> tuple[str, TimeSpec]:
        """Where a class should go, in words and as a ``TimeSpec``."""
        cal = self.instance.calendar
        lunch = cal.lunch_slot
        D = FULL_DAY[day]
        starts = [q for q in range(cal.slots_per_day - s.duration + 1) if q != lunch]
        q = self.rng.choice(starts)
        a, b = sorted(self.rng.sample([x for x in range(cal.slots_per_day) if x != lunch], 2))
        return self.rng.choice([
            (f"{D} morning", TimeSpec(days=[day], part="morning")),
            (f"{D} afternoon", TimeSpec(days=[day], part="afternoon")),
            (f"{D} after lunch", TimeSpec(days=[day], part="afternoon")),
            (f"{D} at {hour(q)}", TimeSpec(days=[day], at=[clock(q)])),
            (f"{hour(q)} on {D}", TimeSpec(days=[day], at=[clock(q)])),
            (f"{D} between {hour(a)} and {hour(b)}", TimeSpec(days=[day], after=clock(a), before=clock(b))),
        ])

    def period(self) -> tuple[str, Period]:
        """A week or weeks a one-off request is for, in words and as a ``Period``."""
        n = self.instance.calendar.weeks
        w = self.rng.randint(1, n - 2)
        return self.rng.choice([
            ("next week", Period(relative="next_week")) if self.week < n else (f"in week {w}", Period(weeks=[w])),
            ("this week", Period(relative="this_week")),
            (f"in week {w}", Period(weeks=[w])),
            (f"in weeks {w} and {w + 1}", Period(weeks=[w, w + 1])),
        ])


Gen = Callable[[_Ctx, int], tuple[str, IntentOutput, dict, bool] | None]


def _ref(s: Session, ctx: _Ctx, word: str | None, *, course: bool, owner: bool = False) -> ClassRef:
    """``s`` as the message names it: by day and time, its course when named, and its kind when the word
    for it says one ("class" and "slot" do not)."""
    d, c, _ = ctx.at(s)
    return ClassRef(course=s.course if course else None, kind=s.kind.value if word not in (None, "class") else None,
                    day=d, time=c, owner=s.faculty if owner else None)


def move(ctx: _Ctx, t: int):
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, h = ctx.at(s)
    to_words, to = ctx.to_time(s, ctx.other_day(d))
    D, k, title = FULL_DAY[d], ctx.kind_word(s), ctx.title(s)
    named = t in (0, 2)
    texts = [f"Could my {D} {h} {title} {k} move to {to_words}?",
             f"Please shift my {k} on {D} at {h} to {to_words}.",
             f"Is it possible to move the {title} {k} I teach on {D} at {h} to {to_words}?",
             f"My {D} {h} {k} clashes with a committee meeting; could it go to {to_words} instead?",
             f"If possible, I'd like my {h} {D} {k} moved to {to_words}."]
    intent = Move(which=_ref(s, ctx, k, course=named), to=to, hard=t != 4,
                  justification="stated" if t == 3 else "none")
    return texts[t], IntentOutput(action="compile", intents=[intent]), {"move": s.id}, False


def swap(ctx: _Ctx, t: int):
    mine = ctx.mine()
    others = [f.id for f in ctx.instance.faculty if f.id != ctx.sender
              and any(x.faculty == f.id for x in ctx.instance.sessions)]
    if not mine or not others:
        return None
    s = ctx.rng.choice(mine)
    colleague = ctx.rng.choice(others)
    o = ctx.rng.choice([x for x in ctx.instance.sessions if x.faculty == colleague])
    ctx.timetable.update(random_timetable(ctx.instance, ctx.rng, [o.id], taken=ctx.timetable))
    name = ctx.instance.faculty_by_id[colleague].name
    d, _, h = ctx.at(s)
    d2, _, h2 = ctx.at(o)
    D, D2, k, k2 = FULL_DAY[d], FULL_DAY[d2], ctx.kind_word(s), ctx.kind_word(o)
    when, period = ctx.period() if t in (2, 3) else ("", Period())
    if ctx.rng.random() < 0.12:  # no classes named: the swap flow asks which
        text = ctx.rng.choice([f"Can I swap one of my classes with {name}?",
                               f"{name} and I would like to swap a slot."])
        gold = Swap(mine=ClassRef(), theirs=ClassRef(owner=colleague))
        return text, IntentOutput(action="compile", intents=[gold]), {}, True
    texts = [f"Could I swap my {D} {h} {k} with {name}'s {D2} {h2} one?",
             f"{name} and I would like to swap slots: my {ctx.title(s)} {k} on {D} at {h} for their {D2} {h2} {k2}.",
             f"I'd like to exchange my {D} {h} class with {name}'s {D2} {h2} {k2} {when}.",
             f"Can {name} take my {D} {h} slot and I take theirs on {D2} at {h2}, {when} only?",
             f"Requesting a swap with {name}: my {k} at {h} on {D} for their {k2} at {h2} on {D2}."]
    # what each wording says about the two classes: (word for mine, course named, word for theirs)
    said = [(k, False, None), (k, True, k2), ("class", False, k2), ("slot", False, None), (k, False, k2)][t]
    gold = Swap(mine=_ref(s, ctx, said[0], course=said[1]), theirs=_ref(o, ctx, said[2], course=False, owner=True),
                period=period)
    return texts[t], IntentOutput(action="compile", intents=[gold]), {"swap": (s.id, o.id)}, False


def extra(ctx: _Ctx, t: int):
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    title, k = ctx.title(s), ctx.kind_word(s)
    w = ctx.rng.randint(2, ctx.instance.calendar.weeks - 1)
    day = ctx.rng.choice(ctx.instance.calendar.days)
    words, when = ctx.to_time(s, day)
    if t == 1 and ctx.week < ctx.instance.calendar.weeks:
        period, pw = Period(relative="next_week"), "next week"
    else:
        period, pw = Period(weeks=[w]), f"in week {w}"
    texts = [f"I'd like an extra {title} {k} {pw}.",
             f"Could you schedule an additional {title} {k} {pw}, preferably {words}?",
             f"Please add a revision {k} for {title} {pw} on {words}.",
             f"I need one more {title} {k} {pw}; {words} would be best.",
             f"We are behind in {title}: an extra {k} {pw}, {words} if it can be arranged."]
    gold = ExtraClass(which=ClassRef(course=s.course, kind=None if k == "class" else s.kind.value), period=period,
                      time=TimeSpec() if t == 0 else when)
    return texts[t], IntentOutput(action="compile", intents=[gold]), {"extra": s.course}, False


def cancel(ctx: _Ctx, t: int):
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, h = ctx.at(s)
    D, k = FULL_DAY[d], ctx.kind_word(s)
    pw, period = ctx.period()
    texts = [f"Please cancel my {D} {h} {k} {pw}.",
             f"I can't hold my {ctx.title(s)} {k} on {D} at {h} {pw}, so please cancel it.",
             f"Cancel the {h} {k} on {D} {pw}, please.",
             f"My {D} {k} at {h} will not take place {pw}.",
             f"{pw.capitalize()}, my {ctx.title(s)} {k} ({D}, {h}) is off; please cancel it."]
    gold = Cancel(which=_ref(s, ctx, k, course=t in (1, 4)), period=period)
    return texts[t], IntentOutput(action="compile", intents=[gold]), {"cancel": s.id}, False


def load_limit(ctx: _Ctx, t: int):
    n = ctx.rng.choice([2, 2, 3, 3, 4])
    texts = [f"Please don't give me more than {n} hours in a row.",
             f"I can teach at most {n} consecutive hours.",
             f"No more than {n} back-to-back classes for me, please.",
             f"Could my timetable avoid {n + 1} straight hours of teaching? {n} at a time is my limit.",
             f"I find it hard to teach beyond {n} hours without a break; please keep my runs to {n}."]
    return texts[t], IntentOutput(action="compile", intents=[LoadLimit(max_consecutive=n)]), {}, False


def withdraw(ctx: _Ctx, t: int):
    D = FULL_DAY[ctx.rng.choice(ctx.instance.calendar.days)]
    texts = [f"Please ignore my earlier request about {D}; it is sorted now.",
             "I'd like to withdraw my last request.",
             "Cancel my previous request, please. I can manage after all.",
             f"Scratch what I asked for on {D}, no change is needed.",
             "Never mind my message from yesterday; please leave my timetable as it is."]
    return texts[t], IntentOutput(action="withdraw"), {}, False


def unavailable_relative(ctx: _Ctx, t: int):
    day = ctx.rng.choice(ctx.instance.calendar.days)
    D = FULL_DAY[day]
    n = ctx.instance.calendar.weeks
    w = ctx.rng.randint(3, n - 2)
    texts = [(f"I can't teach on {D} next week; I have a conference.", Period(relative="next_week"), "stated"),
             (f"I'm unavailable on {D} this week.", Period(relative="this_week"), "none"),
             (f"From week {w} until the end of the semester I cannot teach on {D}s.", Period(from_week=w), "none"),
             (f"I will be on leave on {D}s from week {w} to week {w + 2}.", Period(from_week=w, to_week=w + 2),
              "stated"),
             (f"Next week I won't be able to take any classes on {D} because of an exam duty.",
              Period(relative="next_week"), "stated")]
    text, period, j = texts[t]
    if period.relative == "next_week" and ctx.week >= n:
        return None
    gold = Unavailable(time=TimeSpec(days=[day]), period=period, justification=j)
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def avoid_between(ctx: _Ctx, t: int):
    day = ctx.rng.choice(ctx.instance.calendar.days)
    cal = ctx.instance.calendar
    a, b = sorted(ctx.rng.sample(range(1, cal.slots_per_day), 2))
    D = FULL_DAY[day]
    texts = [(f"Please keep {D} free between {hour(a)} and {hour(b)}.", False),
             (f"I'd rather not have classes on {D} from {hour(a)} to {hour(b)}.", False),
             (f"No classes for me between {hour(a)} and {hour(b)} on {D}s, please; I have a standing meeting.", True),
             (f"Could you avoid {D} {hour(a)} to {hour(b)} for my classes?", False),
             (f"Between {hour(a)} and {hour(b)} on {D} I must not be scheduled.", True)]
    text, hard = texts[t]
    gold = Avoid(time=TimeSpec(days=[day], after=clock(a), before=clock(b)), hard=hard,
                 justification="stated" if t == 2 else "none")
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


GENERATORS: dict[str, Gen] = {"move": move, "swap": swap, "extra_class": extra, "cancel": cancel,
                              "load_limit": load_limit, "withdraw": withdraw,
                              "unavailable_relative": unavailable_relative, "avoid_between": avoid_between}
TEST_TEMPLATE = 4  # each generator's last wording is held out for test


def _resolves(ex: IntentExample, instance: Instance) -> bool:
    got = compile_intents(instance, ex.request, ex.gold, timetable=ex.timetable, current_week=ex.current_week)
    if ex.asks:
        return got.question is not None
    if got.question is not None or got.result.refusal or got.result.errors:
        return False
    w = ex.want
    if "move" in w:
        return [c.scope.session for c in got.result.constraints] == [w["move"]]
    if "swap" in w:
        return [(r.mine, r.theirs) for r in got.swaps] == [w["swap"]]
    if "extra" in w:
        return [instance.session_by_id[r.template].course for r in got.extras] == [w["extra"]]
    if "cancel" in w:
        return [r.session for r in got.cancels] == [w["cancel"]]
    return True


def intent_examples(instance: Instance, per_generator: int = 70, seed: int = 0) -> list[IntentExample]:
    rng = random.Random(seed)
    teachers = [f.id for f in instance.faculty if any(s.faculty == f.id for s in instance.sessions)]
    out: list[IntentExample] = []
    seen: set[str] = set()
    for name, gen in GENERATORS.items():
        made, tries = 0, 0
        while made < per_generator and tries < per_generator * 20:
            tries += 1
            sender = rng.choice(teachers)
            mine = [s.id for s in instance.sessions if s.faculty == sender]
            ctx = _Ctx(instance, rng, sender, random_timetable(instance, rng, mine), rng.randint(1, instance.calendar.weeks))
            t = made % 5
            got = gen(ctx, t)
            if got is None:
                continue
            text, gold, want, asks = got
            text = rng.choice(_OPEN) + text + rng.choice(_CLOSE).format(name=ctx.name)
            if text in seen:
                continue
            split = "test" if t == TEST_TEMPLATE else "val" if rng.random() < 0.1 else "train"
            request = Request(id=f"I-{name}-{made:03d}", channel=Channel.PORTAL, sender_id=sender, role=Role.FACULTY,
                              raw_text=text, received_at=datetime(2026, 9, 1))
            ex = IntentExample(request.id, name, t, split, request, gold, ctx.timetable, ctx.week, asks, want)
            if not _resolves(ex, instance):
                continue
            seen.add(text)
            out.append(ex)
            made += 1
    return out
