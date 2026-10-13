"""More request generators, registered with what they hold out, for data that later runs extend.

``language.intent_corpus`` made the first generators: five wordings each, the last one held out for the
test split. This module adds generators for what the evaluations and probes found the model does not yet
read (how firmly something is asked, which class a move is for against where it should go, weeks counted
from now, several requests in one message, the ways clubs ask for rooms, requests too vague to act on),
and registers each one:

* ``Generator``: a name, the function, how many wordings it has, which wordings are **held out** and the
  evaluation suite they go to (``data/suites/<suite>.jsonl``). Held-out wordings never reach training, so a
  suite measures wordings the model has not seen; phrase banks are split the same way (``_split``).
* ``since``: the schema version (``intents.SCHEMA_VERSION``) the generator needs. A later run adds
  generators to ``REGISTRY``; the ones already there keep their seeds and wordings, so their rows stay the
  same rows and an adapter updated with ``INIT_FROM`` reads only what is new.

Labels follow one rule for how firm a request is (``lexicon.placement_is_hard`` and the phrase banks
here): a requirement only when the message says it must be so (or, for a move, names a conflict with
where the class is now); a hedged or plain request is a wish; "hard for me" is about difficulty, not a
requirement.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from core.instance import Instance

from .corpus import FULL_DAY, hour
from .intent_compiler import clock
from .intent_corpus import CLUBS, PROGRAMMES, Gen, IntentExample, _Ctx, _examples, _ref
from .intents import Avoid, ClassRef, ClubBooking, IntentOutput, LoadLimit, Move, Only, Period, TimeSpec, Unavailable
from .lexicon import placement_is_hard


@dataclass(frozen=True)
class Generator:
    name: str
    fn: Gen
    templates: int
    held_out: tuple[int, ...]
    suite: str
    since: int = 3

    @property
    def train_templates(self) -> tuple[int, ...]:
        return tuple(t for t in range(self.templates) if t not in self.held_out)


def _split(bank: list, held: int) -> tuple[list, list]:
    """A phrase bank as (train phrases, held-out phrases): the last ``held`` are for the suites only."""
    return bank[:-held], bank[-held:]


NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def say_hour(q: int, rng: random.Random) -> str:
    """Slot ``q``'s start the ways people write it: "2 pm", "2pm", "14:00", "2 p.m."."""
    h = 9 + q
    twelve = f"{h % 12 or 12}"
    ampm = "am" if h < 12 else "pm"
    return rng.choice([f"{twelve} {ampm}", f"{twelve}{ampm}", f"{h:02d}:00", f"{twelve} {ampm[0]}.m."])


def say_range(a: int, b: int, rng: random.Random) -> str:
    """From slot ``a``'s start to slot ``b``'s: "2-4pm", "2 to 4 pm", "14:00-16:00", "between 2 and 4 pm"."""
    ha, hb = 9 + a, 9 + b
    ta, tb = f"{ha % 12 or 12}", f"{hb % 12 or 12}"
    end = "am" if hb < 12 else "pm"
    if ha < 12 <= hb:
        return rng.choice([f"{ta} am to {tb} pm", f"{ha:02d}:00-{hb:02d}:00", f"between {ta} am and {tb} pm"])
    return rng.choice([f"{ta}-{tb}{end}", f"{ta} to {tb} {end}", f"{ha:02d}:00-{hb:02d}:00",
                       f"between {ta} and {tb} {end}"])


def _when(ctx: _Ctx) -> tuple[str, TimeSpec]:
    """Times of one day a request is about, in words and as a ``TimeSpec``."""
    cal, rng = ctx.instance.calendar, ctx.rng
    day = rng.choice(cal.days)
    D = FULL_DAY[day]
    plural = rng.random() < 0.4
    Ds = D + "s" if plural else D
    a, b = sorted(rng.sample(range(1, cal.slots_per_day), 2))
    q = rng.randrange(1, cal.slots_per_day)
    return rng.choice([
        (f"before {say_hour(q, rng)} on {Ds}", TimeSpec(days=[day], before=clock(q))),
        (f"after {say_hour(q, rng)} on {Ds}", TimeSpec(days=[day], after=clock(q))),
        (f"on {D} {say_range(a, b, rng)}", TimeSpec(days=[day], after=clock(a), before=clock(b))),
        (f"on {Ds} after lunch", TimeSpec(days=[day], part="afternoon")),
        (f"on {D} afternoons", TimeSpec(days=[day], part="afternoon")),
        (f"in the afternoon on {Ds}", TimeSpec(days=[day], part="afternoon")),
        (f"on {D} mornings", TimeSpec(days=[day], part="morning")),
        (f"before lunch on {Ds}", TimeSpec(days=[day], part="morning")),
        (f"on {Ds}", TimeSpec(days=[day])),
    ])


# how firmly a time is asked to be kept free; the last two of each bank are for the suites only
HEDGES, HEDGES_HELD = _split(["if possible", "if you could", "ideally", "if it can be managed", "where possible",
                              "if it's not too much trouble", "when you get a chance", "if at all possible"], 2)
FIRM, FIRM_HELD = _split(["this is a must", "it has to stay free", "it must stay free", "this is not negotiable",
                          "it needs to stay free", "I need that time kept free"], 2)
DIFFICULT, DIFFICULT_HELD = _split(["it's hard for me", "it's difficult for me", "I find it tough",
                                    "it's a struggle for me", "it's really hard for me to manage"], 2)


def firmness(ctx: _Ctx, t: int):
    """Times kept free, said with a hedge, as a must, as a difficulty ("hard for me": a wish) or plainly."""
    words, time = _when(ctx)
    rng = ctx.rng
    held = t >= 4
    hedge = rng.choice(HEDGES_HELD if held else HEDGES)
    firm = rng.choice(FIRM_HELD if held else FIRM)
    tough = rng.choice(DIFFICULT_HELD if held else DIFFICULT)
    texts = [(f"{hedge.capitalize()}, no classes for me {words}.", False),
             (f"Please keep me free {words}; {firm}.", True),
             (f"Could you avoid scheduling me {words}? {tough.capitalize()}.", False),
             (f"I'm fine with any time except {words}.", False),
             (rng.choice([f"No teaching {words} for me, {hedge}.", f"Avoid {words} for me if you can; {tough}."]),
              False),
             (f"Nothing of mine {words} please, {firm}.", True)]
    text, hard = texts[t]
    return text, IntentOutput(action="compile", intents=[Avoid(time=time, hard=hard)]), {}, False


def which_to(ctx: _Ctx, t: int):
    """A move with the class named by course, by where it is now, or both, and where it should go; how firm
    it is follows the labelling rule."""
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, _ = ctx.at(s)
    q = ctx.timetable[s.id].slot
    D, k, title = FULL_DAY[d], ctx.kind_word(s), ctx.title(s)
    to_words, to = ctx.to_time(s, ctx.other_day(d))
    h = say_hour(q, ctx.rng)
    texts = [(f"I need my {title} {k} on {to_words}.", "course"),
             (f"Can my {title} {k} be on {to_words} instead?", "course"),
             (f"Move my {D} {h} {k} to {to_words}.", "where"),
             (f"My {k} currently on {D} at {h}: could we do {to_words}?", "where"),
             (f"{to_words[0].upper() + to_words[1:]} would suit my {title} {k} better.", "course"),
             (f"I want my {D} {k} ({h}) moved to {to_words}; it can't stay where it is.", "where")]
    text, named = texts[t]
    ref = (ClassRef(course=s.course, kind=None if k == "class" else s.kind.value) if named == "course"
           else _ref(s, ctx, k, course=False))
    gold = Move(which=ref, to=to, hard=placement_is_hard(text))
    return text, IntentOutput(action="compile", intents=[gold]), ({"move": s.id} if named == "where" else {}), False


def relative_span(ctx: _Ctx, t: int):
    """Weeks counted from now: "the next three weeks", "this week and next", "the rest of the semester"."""
    rng, n = ctx.rng, ctx.instance.calendar.weeks
    day = rng.choice(ctx.instance.calendar.days)
    D = FULL_DAY[day]
    k = rng.choice([2, 3, 4])
    kw = rng.choice([NUMBER[k], str(k)])
    texts = [(f"I can't teach on {D}s for the next {kw} weeks.", Unavailable(
                 time=TimeSpec(days=[day]), period=Period(relative="next_week", count=k))),
             (f"I'm away this week and next, so no classes on {D} please.", Unavailable(
                 time=TimeSpec(days=[day]), period=Period(relative="this_week", count=2))),
             (f"For the rest of the semester, please keep my {D} afternoons free.", Avoid(
                 time=TimeSpec(days=[day], part="afternoon"), period=Period(relative="rest_of_semester"))),
             (f"Starting next week I'm unavailable on {D}s for {kw} weeks.", Unavailable(
                 time=TimeSpec(days=[day]), period=Period(relative="next_week", count=k))),
             (f"I won't be around on {D}s over the coming {kw} weeks (project visit).", Unavailable(
                 time=TimeSpec(days=[day]), period=Period(relative="next_week", count=k), justification="stated")),
             (f"Until the end of term I can't do {D} mornings.", Unavailable(
                 time=TimeSpec(days=[day], part="morning"), period=Period(relative="rest_of_semester")))]
    text, gold = texts[t]
    last = ctx.week + (gold.period.relative == "next_week") + (gold.period.count or 1) - 1
    if gold.period.relative != "rest_of_semester" and last > n:
        return None  # past the end of term: kept for the periods the semester has
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def _part_avoid(ctx: _Ctx) -> tuple[str, Avoid]:
    words, time = _when(ctx)
    if ctx.rng.random() < 0.5:
        return f"I'd rather not teach {words}", Avoid(time=time)
    return f"I must not be scheduled {words}", Avoid(time=time, hard=True)


def _part_limit(ctx: _Ctx) -> tuple[str, LoadLimit]:
    n = ctx.rng.choice([2, 3, 4])
    return ctx.rng.choice([f"no more than {n} hours in a row for me", f"max {n} classes back to back",
                           f"at most {n} consecutive hours please"]), LoadLimit(max_consecutive=n)


def _part_off(ctx: _Ctx) -> tuple[str, Unavailable]:
    day = ctx.rng.choice(ctx.instance.calendar.days)
    w = ctx.rng.randint(2, ctx.instance.calendar.weeks - 1)
    return f"I can't teach on {FULL_DAY[day]} in week {w}", Unavailable(time=TimeSpec(days=[day]),
                                                                       period=Period(weeks=[w]))


def _part_move(ctx: _Ctx) -> tuple[str, Move, str] | None:
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, h = ctx.at(s)
    to_words, to = ctx.to_time(s, ctx.other_day(d))
    k = ctx.kind_word(s)
    return f"could my {FULL_DAY[d]} {h} {k} move to {to_words}?", Move(which=_ref(s, ctx, k, course=False), to=to), s.id


def multi(ctx: _Ctx, t: int):
    """Two requests in one message; each keeps its own firmness (a must says so in its own words)."""
    a1, i1 = _part_avoid(ctx)
    a2, i2 = _part_avoid(ctx)
    lim, il = _part_limit(ctx)
    off, io = _part_off(ctx)
    mv = _part_move(ctx)
    if mv is None or i1.time == i2.time:
        return None
    m, im, _ = mv
    texts = [(f"{a1}, and {lim}.", [i1, il]),
             (f"{off}. Also, {m}", [io, im]),
             (f"Two things: {a1}; {a2[0].lower() + a2[1:]}.", [i1, i2]),
             (f"{lim[0].upper() + lim[1:]}. Also {a1[0].lower() + a1[1:]}.", [il, i1]),
             (f"{m[0].upper() + m[1:]} Separately, {off}.", [im, io]),
             (f"{a1}; plus {lim}", [i1, il])]
    text, found = texts[t]
    if any(isinstance(i, Avoid) and not i.hard for i in found) and any(getattr(i, "hard", False) for i in found):
        return None  # a wish beside a must in one message: whether "rather" softens both is not settled
    return text, IntentOutput(action="compile", intents=found), {}, False


ROOM_KINDS, ROOM_KINDS_HELD = _split(["a seminar room", "a lecture hall", "a classroom", "a big hall",
                                      "a room with a projector", "a tutorial room", "a large classroom"], 2)
ATTEND, ATTEND_HELD = _split(["about {n} of us", "around {n} people", "{n} members", "we're roughly {n}",
                              "expecting {n}", "{n}-odd people"], 2)


def club_more(ctx: _Ctx, t: int):
    """Club bookings in more of the ways clubs write: times as "6-8pm" or "18:00 to 20:00", the number of
    people said several ways, a kind of room (``room_kind``) rather than a named one."""
    rng = ctx.rng
    club, activity = rng.choice(CLUBS)
    start_h = rng.choice([14, 15, 17, 18, 19])
    dur = rng.choice([1, 2, 2, 3])
    start, end = f"{start_h:02d}:00", f"{start_h + dur:02d}:00"
    held = t >= 4
    rk = rng.choice(ROOM_KINDS_HELD if held else ROOM_KINDS)
    n = rng.choice([12, 20, 25, 30, 40, 50, 60, 90])
    people = rng.choice(ATTEND_HELD if held else ATTEND).format(n=n)
    day = rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    D = FULL_DAY[day]
    span = rng.choice([f"{start_h - 12}-{start_h - 12 + dur}pm", f"{start} to {end}",
                       f"from {start_h - 12} till {start_h - 12 + dur} pm"])
    cohort = rng.choice(PROGRAMMES)
    texts = [(f"Hi! {club} needs {rk} on {D}, {span}, {people}, for our {activity}.", {}),
             (f"Is {rk} free this {D} {span}? {club}, {activity}, {people}.", {"period": Period(relative="this_week")}),
             (f"{club}: booking {rk} next {D} {span} for a {activity}. {people.capitalize()}.",
              {"period": Period(relative="next_week")}),
             (f"can the {club.lower()} get {rk} on {D} {span}? {people}, mostly {cohort}",
              {"cohorts": [cohort], "activity": None}),
             (f"Looking for {rk} on {D} {span} for the {club}'s {activity} - {people}.", {}),
             (f"{club} {activity}: {D}, {span}, {people}. {rk.capitalize()} would do.", {})]
    text, extra = texts[t]
    if extra.get("period") and extra["period"].relative == "next_week" and ctx.week >= ctx.instance.calendar.weeks:
        return None
    fields = {"activity": activity, **extra}
    gold = ClubBooking(club=club, day=day, start=start, end=end, attendees=n, room_kind=rk.removeprefix("a "),
                       **fields)
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def only_firmness(ctx: _Ctx, t: int):
    """The only days someone wants to teach, as a wish or a must."""
    days = sorted(ctx.rng.sample(ctx.instance.calendar.days, ctx.rng.choice([2, 3, 4])),
                  key=ctx.instance.calendar.days.index)
    names = [FULL_DAY[d] for d in days]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    texts = [(f"I'd prefer to teach only on {listed}.", False),
             (f"I can only come in on {listed}; please keep all my classes to those days.", True),
             (f"Ideally all my teaching would be on {listed}.", False),
             (f"My classes have to be on {listed}, nothing else.", True),
             (f"If you can, limit my classes to {listed}.", False),
             (f"Please restrict my teaching to {listed}; I am not on campus otherwise.", True)]
    text, hard = texts[t]
    gold = Only(time=TimeSpec(days=days), hard=hard, justification="stated" if t == 5 else "none")
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def vague(ctx: _Ctx, t: int):
    """A timetable request that leaves out what would have to be guessed: the answer is a question."""
    from .intent_compiler import _MISSING

    texts = [("Something's come up and I'll miss a couple of classes later this month.", "unavailable",
              ["days", "weeks"]),
             ("Could my timetable be a bit lighter on some days?", "avoid", ["days", "slots"]),
             ("Please move one of my classes.", "move", ["session", "slots"]),
             ("I won't be able to make it on one of the days in the coming weeks.", "unavailable", ["days", "weeks"]),
             ("Can we shift things around a little for me next month?", "avoid", ["days", "slots"]),
             ("I need a different slot for one of my lectures.", "move", ["session", "slots"])]
    text, topic, missing = texts[t]
    question = f"Could you tell me {' and '.join(_MISSING.get(m, m) for m in missing)} you mean?"
    return text, IntentOutput(action="clarify", topic=topic, missing=missing, question=question), {}, False


REGISTRY: list[Generator] = [
    Generator("firmness", firmness, 6, (4, 5), "firmness"),
    Generator("only_firmness", only_firmness, 6, (4, 5), "firmness"),
    Generator("which_to", which_to, 6, (4, 5), "moves"),
    Generator("relative_span", relative_span, 6, (4, 5), "periods"),
    Generator("multi", multi, 6, (4, 5), "multi"),
    Generator("club_more", club_more, 6, (4, 5), "clubs"),
    Generator("vague", vague, 6, (4, 5), "vague"),  # added last: the others keep their seeds and rows
]
SENDERS = {"club_more": "student_or_teacher"}


def generated(instance: Instance, per_generator: int = 60, held_per_generator: int = 30, seed: int = 200,
              names: list[str] | None = None) -> list[IntentExample]:
    """Every registered generator's examples: ``per_generator`` on its train wordings (a tenth to val) and
    ``held_per_generator`` on its held-out ones, whose split is their suite's name. Each generator has its
    own seed, so adding one leaves the others' rows as they were."""
    from . import intent_corpus

    out: list[IntentExample] = []
    for i, g in enumerate(REGISTRY):
        if names is not None and g.name not in names:
            continue
        sender = SENDERS.get(g.name)
        if sender:
            intent_corpus.SENDERS.setdefault(g.name, sender)
        train = _examples(instance, random.Random(seed + 10 * i), {g.name: (g.name, g.fn)}, per_generator, None,
                          templates=g.train_templates)
        held = _examples(instance, random.Random(seed + 10 * i + 5), {f"{g.name}@{g.suite}": (g.name, g.fn)},
                         held_per_generator, None, templates=g.held_out)
        for ex in held:
            ex.split = g.suite
        out += train + held
    return out


GENERATORS: dict[str, Callable] = {g.name: g.fn for g in REGISTRY}
