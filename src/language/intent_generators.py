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
from .intents import (
    AnyOf,
    Avoid,
    Cancel,
    ClassRef,
    ClubBooking,
    Fallback,
    IntentOutput,
    Limit,
    LoadLimit,
    Move,
    Only,
    Period,
    Ranked,
    Relate,
    TimeSpec,
    Unavailable,
    Who,
)
from .lexicon import necessary, placement_is_hard


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


def reason_firmness(ctx: _Ctx, t: int):
    """The only days someone can teach, with a reason that rules the other days out (a must, though no "must"
    is said: "those are the only days I'm on campus") beside a reason that only explains a wish."""
    days = sorted(ctx.rng.sample(ctx.instance.calendar.days, ctx.rng.choice([2, 3, 4])),
                  key=ctx.instance.calendar.days.index)
    names = [FULL_DAY[d] for d in days]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    texts = [(f"Please keep my classes to {listed}; those are the only days I'm on campus.", True),
             (f"I'm at the other campus for the rest of the week, so my teaching has to be on {listed}.", True),
             (f"Kindly schedule all my classes on {listed}. I travel in from out of town just on those days.", True),
             (f"I'd prefer my classes on {listed}, as I keep the other days for research.", False),
             (f"Only {listed} work for me; I'm away from campus the rest of the week.", True),
             (f"If you can, put my classes on {listed}. I do fieldwork on the other days.", False)]
    text, hard = texts[t]
    gold = Only(time=TimeSpec(days=days), hard=hard, justification="stated")
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def term_span(ctx: _Ctx, t: int):
    """Until the end of the term, said the ways people say it: the period is the rest of the semester."""
    day = ctx.rng.choice(ctx.instance.calendar.days)
    D = FULL_DAY[day]
    part, Part = ctx.rng.choice([("morning", "mornings"), ("afternoon", "afternoons")])
    rest = Period(relative="rest_of_semester")
    texts = [(f"For the rest of the term I can't teach on {D} {Part}.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=rest)),
             (f"Till the semester ends, please keep my {D} {Part} free.",
              Avoid(time=TimeSpec(days=[day], part=part), period=rest)),
             (f"From now until the semester is over I'm unavailable on {D}s.",
              Unavailable(time=TimeSpec(days=[day]), period=rest)),
             (f"Through to the end of the semester, I won't be able to do {D} {Part}.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=rest)),
             (f"Up to the end of this term I can't take {D} {part} classes.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=rest)),
             (f"For what's left of the semester, no {D} {Part} for me please.",
              Avoid(time=TimeSpec(days=[day], part=part), period=rest))]
    text, gold = texts[t]
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


# "hard" about how difficult a time is (a wish) beside "hard" meaning a requirement; the last two are held out
TOUGH, TOUGH_HELD = _split(["hard for me", "it's hard for me", "that's hard for me", "really hard for me",
                            "hard to manage for me", "tough for me"], 2)
STRICT, STRICT_HELD = _split(["this is a hard requirement", "hard constraint please", "it's a hard limit for me",
                              "treat this as a hard rule", "a hard requirement, not a preference"], 2)


def difficulty(ctx: _Ctx, t: int):
    """Times kept free with a short "hard for me" after them, the way people tack it on (a wish), beside
    "hard" said of the request itself (a requirement)."""
    words, time = _when(ctx)
    rng = ctx.rng
    held = t >= 4
    tough = rng.choice(TOUGH_HELD if held else TOUGH)
    strict = rng.choice(STRICT_HELD if held else STRICT)
    texts = [(f"no classes {words} please. {tough}.", False),
             (f"Please don't put me {words}, {tough}.", False),
             (f"Keep me free {words}; {strict}.", True),
             (f"Could you skip {words} for me? {tough.capitalize()}.", False),
             (f"nothing {words} pls. {tough}", False),
             (f"No teaching {words}, {strict}.", True)]
    text, hard = texts[t]
    return text, IntentOutput(action="compile", intents=[Avoid(time=time, hard=hard)]), {}, False


# why someone is away (a stated reason), as a noun phrase; the last two are held out
AWAY_FOR, AWAY_FOR_HELD = _split(["a workshop in Pune", "an examination board meeting", "a medical appointment",
                                  "a selection committee", "an invited talk", "a site visit with students",
                                  "a thesis defence elsewhere", "a funding review"], 2)
# closers that sound as if something were left open ("adjust accordingly") though nothing is: read the rest
SETTLE, SETTLE_HELD = _split(["Please plan my timetable around this.", "Kindly make the needed changes.",
                              "Please rework my schedule to suit.", "Please adjust things as needed.",
                              "Kindly sort out my classes as required.", "Please arrange my classes accordingly.",
                              "Kindly do the needful."], 2)


def dated(ctx: _Ctx, t: int):
    """A reason and when the sender cannot teach, with the week or the whole semester said outright: nothing
    is left to ask, so it is read as said (an unavailability with a stated reason, never a question). The
    wordings differ from the frozen test split's, which these are meant to generalise to, not copy."""
    cal, rng = ctx.instance.calendar, ctx.rng
    if ctx.week >= cal.weeks:
        return None
    w = rng.randint(ctx.week + 1, cal.weeks)
    why = rng.choice(AWAY_FOR_HELD if t >= 6 else AWAY_FOR)
    i = rng.randrange(len(cal.days) - 2)
    run = cal.days[i:i + rng.choice([2, 3])]
    day, D = cal.days[i], FULL_DAY[cal.days[i]]
    q = rng.randrange(1, cal.slots_per_day)
    h = say_hour(q, rng)
    part, Part = rng.choice([("morning", "mornings"), ("afternoon", "afternoons")])
    weeks = Period(weeks=[w])
    texts = [(f"Week {w}, {FULL_DAY[run[0]]} through {FULL_DAY[run[-1]]}: I'm away for {why}, so I can't teach those days.",
              TimeSpec(days=run), weeks),
             (f"All this semester I have {why} on {D} {Part}, so I'm not free then.",
              TimeSpec(days=[day], part=part), None),
             (f"Because of {why} I'm unable to take the {h} class on {D} of week {w}.",
              TimeSpec(days=[day], at=[clock(q)]), weeks),
             (f"{why[0].upper() + why[1:]} in wk {w} - not available {'/'.join(FULL_DAY[d][:3].lower() for d in run)}.",
              TimeSpec(days=run), weeks),
             (f"I'll be away for {why} in week {w} from {FULL_DAY[run[0]]} to {FULL_DAY[run[-1]]}.",
              TimeSpec(days=run), weeks),
             (f"From {FULL_DAY[run[0]]} to {FULL_DAY[run[-1]]} of week {w} I have {why}, so I can't take classes.",
              TimeSpec(days=run), weeks),
             (f"I have {why} every {D} {part} this term, so I won't be available then.",
              TimeSpec(days=[day], part=part), None),
             (f"{why}, week {w}. so i cant be there for {h} {D}.", TimeSpec(days=[day], at=[clock(q)]), weeks)]
    text, time, period = texts[t]
    if rng.random() < 0.6:
        text += " " + rng.choice(SETTLE_HELD if t >= 6 else SETTLE)
    gold = Unavailable(time=time, period=period or Period(), justification="stated")
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


def _plural(k: str) -> str:
    return "classes" if k == "class" else f"{k}s"


def exception(ctx: _Ctx, t: int):
    """A request with a day it must not use said after it ("..., but not on Friday", "any day except
    Wednesday"): the exception is a reading of its own beside the request, never dropped. A plain exception is
    a wish (the labelling rule)."""
    cal, rng = ctx.instance.calendar, ctx.rng
    if ctx.week >= cal.weeks:
        return None
    mine = ctx.mine()
    if not mine:
        return None
    w = rng.randint(ctx.week + 1, cal.weeks)
    i = rng.randrange(len(cal.days) - 1)
    away = cal.days[i:i + rng.choice([1, 2])]
    x = rng.choice([d for d in cal.days if d not in away])
    X, gone = FULL_DAY[x], " and ".join(FULL_DAY[d] for d in away)
    s = rng.choice(mine)
    d, _, h = ctx.at(s)
    k, title = ctx.kind_word(s), ctx.title(s)
    off = Unavailable(time=TimeSpec(days=away), period=Period(weeks=[w]))
    not_x = Avoid(time=TimeSpec(days=[x]), period=Period(weeks=[w]))
    course = Avoid(who=Who(kind="course", id=s.course), time=TimeSpec(days=[x]))
    elsewhere = [e for e in cal.days if e not in (d, x)]
    move = f"Please move my {FULL_DAY[d]} {h} {k} to another day, just not {X}."
    texts = [(f"I'm away on {gone} of week {w}. Please hold the make-up classes that week, but not on {X}.",
              [off, not_x]),
             (f"My {title} classes can go on any day except {X}.", [course]),
             (move, [Move(which=_ref(s, ctx, k, course=False), to=TimeSpec(days=elsewhere),
                          hard=placement_is_hard(move))]),
             (f"Week {w}: I can't come in on {gone}. Make-up classes are fine any day except {X}.", [off, not_x]),
             (f"I'll miss {gone} in week {w}; make up the classes that week, only not on {X} please.",
              [off, not_x]),
             (f"Any day apart from {X} works for my {title} classes.", [course])]
    text, found = texts[t]
    return text, IntentOutput(action="compile", intents=found), {}, False


def only_times(ctx: _Ctx, t: int):
    """The only part of the day someone can (a must) or would like to (a wish) teach, from a week on or for
    the semester: "From week 15 onwards I can only teach in the afternoons"."""
    cal, rng = ctx.instance.calendar, ctx.rng
    if ctx.week >= cal.weeks:
        return None
    w = rng.randint(ctx.week + 1, cal.weeks)
    part, parts = rng.choice([("morning", "mornings"), ("afternoon", "afternoons")])
    texts = [(f"From week {w} onwards I can only teach in the {parts}.", Period(from_week=w), True),
             (f"From week {w} on I'd prefer to teach only in the {parts}.", Period(from_week=w), False),
             (f"For the rest of the semester I can only do {parts}.", Period(relative="rest_of_semester"), True),
             (f"Ideally I'd teach just {parts} this semester.", Period(), False),
             (f"Starting week {w}, {parts} are the only time I can teach.", Period(from_week=w), True),
             (f"If it can be managed, only {parts} for me from week {w}.", Period(from_week=w), False)]
    text, period, hard = texts[t]
    gold = Only(time=TimeSpec(part=part), period=period, hard=hard)
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def compound(ctx: _Ctx, t: int):
    """Two different requests in one message, each read on its own: a cancellation beside a move or a wish,
    an absence beside a move. A class is named by what the message says of it and nothing more: a cancelled
    lecture named by its day only gets no time (the only one of that course and kind that day)."""
    cal, rng = ctx.instance.calendar, ctx.rng
    mine = ctx.mine()
    if len(mine) < 2:
        return None
    a, b = rng.sample(mine, 2)
    da, _, ha = ctx.at(a)
    db, _, hb = ctx.at(b)
    if (da, ha) == (db, hb):
        return None
    ka, kb = ctx.kind_word(a), ctx.kind_word(b)
    Da, Db, title_a = FULL_DAY[da], FULL_DAY[db], ctx.title(a)
    pw, period = ctx.period()
    to_words, to = ctx.to_time(b, ctx.other_day(db))
    words, time = _when(ctx)
    move_b = f"could my {Db} {hb} {kb} move to {to_words}?"
    moved = Move(which=_ref(b, ctx, kb, course=False), to=to, hard=placement_is_hard(move_b))
    by_time = Cancel(which=_ref(a, ctx, ka, course=False), period=period)
    by_course = Cancel(which=_ref(a, ctx, ka, course=True), period=period)
    same_day = [s for s in mine if s.course == a.course and s.kind == a.kind and ctx.timetable[s.id].day == da]
    by_day = Cancel(which=ClassRef(course=a.course, kind=None if ka == "class" else a.kind.value, day=da),
                    period=period)
    if ctx.week >= cal.weeks:
        return None
    w = rng.randint(ctx.week + 1, cal.weeks)
    x = rng.choice(cal.days)
    off = Unavailable(time=TimeSpec(days=[x]), period=Period(weeks=[w]))
    texts = [(f"Please cancel my {Da} {ha} {ka} {pw}, and {move_b}", [by_time, moved]),
             ((f"Two things: my {title_a} {ka} on {Da} at {ha} is off {pw}; also, if possible, no classes for me "
               f"{words}."), [by_course, Avoid(time=time)]),
             (f"I can't teach on {FULL_DAY[x]} in week {w}, no make-up needed. Separately, {move_b}", [off, moved]),
             (f"Cancel my {title_a} {ka} on {Da} {pw}. And I'd rather not teach {words}.",
              [by_day, Avoid(time=time)] if len(same_day) == 1 else None),
             (f"{move_b[0].upper() + move_b[1:]} And please cancel my {title_a} {ka} on {Da} at {ha} {pw}.",
              [moved, by_course]),
             ((f"{pw[0].upper() + pw[1:]} my {Da} {ha} {ka} won't happen, please cancel it. Also keep me free "
               f"{words} if you can."), [by_time, Avoid(time=time)])]
    text, found = texts[t]
    if found is None:
        return None
    return text, IntentOutput(action="compile", intents=found), {}, False


def week_patterns(ctx: _Ctx, t: int):
    """Weeks left out of a period and odd or even weeks (schema 5): "for the rest of the term, except week 12",
    "on even weeks", "every other week from week 9". The reader says which; ``intents.weeks`` counts them, the
    arithmetic the v8 reader got wrong (an exception dropped, "every other week" as the next eight)."""
    cal, rng = ctx.instance.calendar, ctx.rng
    if cal.weeks - ctx.week < 3:
        return None
    day = rng.choice(cal.days)
    D = FULL_DAY[day]
    part, Part = rng.choice([("morning", "mornings"), ("afternoon", "afternoons")])
    a = rng.randint(ctx.week + 1, cal.weeks - 2)
    b = rng.randint(a + 2, cal.weeks)
    x = rng.randint(a + 1, b - 1)  # inside every period below, so leaving it out changes something
    start = rng.randint(ctx.week + 1, cal.weeks - 2)
    every_other = "odd" if start % 2 else "even"
    even_odd = rng.choice(["even", "odd"])
    rest = Period(relative="rest_of_semester", except_weeks=[x])
    texts = [(f"I can't teach on {D}s for the rest of the term, except in week {x}.",
              Unavailable(time=TimeSpec(days=[day]), period=rest)),
             (f"From week {a} on, please keep my {D} {Part} free, apart from week {x}.",
              Avoid(time=TimeSpec(days=[day], part=part), period=Period(from_week=a, except_weeks=[x]))),
             (f"On {even_odd} weeks I'm unavailable on {D} {Part}.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=Period(parity=even_odd))),
             (f"I'm away on {D}s in weeks {a} to {b}, but not in week {x}.",
              Unavailable(time=TimeSpec(days=[day]), period=Period(from_week=a, to_week=b, except_weeks=[x]))),
             (f"Every other week, starting in week {start}, I can't do {D} {Part}.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=Period(from_week=start, parity=every_other))),
             (f"No {D} classes for me in the {even_odd} weeks from now on, please.",
              Avoid(time=TimeSpec(days=[day]), period=Period(relative="rest_of_semester", parity=even_odd))),
             (f"Apart from week {x}, I'm not available on {D} {Part} for what's left of the semester.",
              Unavailable(time=TimeSpec(days=[day], part=part), period=rest)),
             (f"Alternate weeks from week {start}: {D}s are out for me.",
              Unavailable(time=TimeSpec(days=[day]), period=Period(from_week=start, parity=every_other)))]
    text, gold = texts[t]
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def limits(ctx: _Ctx, t: int):
    """Limits on a week (schema 6): classes or hours of teaching a day, teaching days a week, free hours
    between classes; beside hours in a row (load_limit) and hours a week, which moving classes cannot change
    (out_of_scope). A must only when the message says one (the labelling rule)."""
    rng = ctx.rng
    n, h, d = rng.choice([1, 2, 3]), rng.choice([2, 3, 4]), rng.choice([2, 3, 4])
    N, H, Dw = _WORDS[n], _WORDS[h], _WORDS[d]
    texts = [(f"No more than {N} classes a day for me, please.", ("classes_per_day", n)),
             (f"I'd like to teach on at most {Dw} days a week.", ("days_per_week", d)),
             ("Please leave at least an hour between any two of my classes.", ("gap", 1)),
             (f"I can't do more than {H} hours of teaching on any one day.", ("hours_per_day", h)),
             (f"Please don't give me more than {H} hours back to back.", "in_a_row"),
             (f"Could you keep my total teaching under {rng.choice([8, 10, 12])} hours a week?", "weekly"),
             (f"Cap me at {N} classes per day.", ("classes_per_day", n)),
             (f"I need my teaching squeezed into {Dw} days a week at most.", ("days_per_week", d))]
    text, what = texts[t]
    if what == "weekly":
        return text, IntentOutput(action="out_of_scope"), {}, False
    if what == "in_a_row":
        return text, IntentOutput(action="compile", intents=[LoadLimit(max_consecutive=h)]), {}, False
    measure, k = what
    return text, IntentOutput(action="compile", intents=[Limit(measure=measure, n=k, hard=necessary(text))]), {}, False


def relations(ctx: _Ctx, t: int):
    """Classes placed against each other (schema 6): no two lectures of a course on a day, a practical right
    after one of its lectures, lectures a day apart, at the same hour or in the same room, and "not right
    after" said as a relation ruled out."""
    from core.instance import SessionKind

    lectures: dict[str, list] = {}
    for s in ctx.mine(SessionKind.LECTURE):
        lectures.setdefault(s.course, []).append(s)
    many = [c for c, ss in lectures.items() if len(ss) >= 2]
    practical = {s.course for s in ctx.mine(SessionKind.PRACTICAL)} & set(many)
    if not many:
        return None
    course = ctx.rng.choice(sorted(practical) if t in (1, 5, 7) and practical else many)
    if t in (1, 5, 7) and course not in practical:
        return None
    T = ctx.instance.course_title(course)
    lec, lab = ClassRef(course=course, kind="lecture"), ClassRef(course=course, kind="practical")
    texts = [(f"Please don't put two {T} lectures on the same day.", Relate(which=lec, relation="different_day")),
             (f"The {T} practical should come right after one of its lectures, on the same day.",
              Relate(which=lab, relation="right_after", other=lec, any=True)),
             (f"Could my {T} lectures be spread out, with at least a day between each?",
              Relate(which=lec, relation="day_gap", days=1)),
             (f"I'd like all my {T} lectures at the same hour of the day.", Relate(which=lec, relation="same_hour")),
             (f"Keep my {T} lectures in the same room, please.", Relate(which=lec, relation="same_room")),
             (f"Please don't put a {T} lecture right after the {T} practical.",
              Relate(which=lec, relation="right_after", other=lab, negate=True)),
             (f"No two of my {T} lectures should fall on one day.", Relate(which=lec, relation="different_day")),
             (f"I want the {T} practical on the same day as one of the {T} lectures.",
              Relate(which=lab, relation="same_day", other=lec, any=True))]
    text, gold = texts[t]
    gold = gold.model_copy(update={"hard": necessary(text)})
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def choices(ctx: _Ctx, t: int):
    """Choices between items (schema 6): one of two times (any_of), a first and a second choice (ranked), a
    change and what to do if it cannot be made (fallback), and a class named as another item's (``same``) to
    relate it. The two times always differ in the part of the day, so they are never one move to two days."""
    from core.instance import SessionKind

    mine = ctx.mine()
    if not mine:
        return None
    rng = ctx.rng
    s = rng.choice(mine)
    d, _, h = ctx.at(s)
    k = ctx.kind_word(s)
    d1 = ctx.other_day(d)
    d2 = ctx.other_day(d, d1)
    p1, p2 = rng.choice([("morning", "afternoon"), ("afternoon", "morning")])
    to1, to2 = TimeSpec(days=[d1], part=p1), TimeSpec(days=[d2], part=p2)
    w1, w2 = f"{FULL_DAY[d1]} {p1}", f"{FULL_DAY[d2]} {p2}"
    week = rng.randint(min(ctx.week + 1, ctx.instance.calendar.weeks), ctx.instance.calendar.weeks)
    ref = _ref(s, ctx, k, course=False)
    a = Move(id="a", which=ref, to=to1)
    b = Move(id="b", which=ClassRef(same="a"), to=to2)
    any_of, ranked = [a, b, AnyOf(of=["a", "b"])], [a, b, Ranked(of=["a", "b"])]
    on = Period(weeks=[week])
    fallback = [a.model_copy(update={"to": TimeSpec(days=[d1]), "period": on}),
                Cancel(id="b", which=ClassRef(same="a"), period=on), Fallback(of=["a", "b"])]
    my = f"my {FULL_DAY[d]} {h} {k}"
    texts = [(f"Could you move {my} to {w1} or {w2}?", any_of),
             (f"Ideally {my} would move to {w1}; if that's not possible, {w2} is fine.", ranked),
             (f"Please move {my} to {FULL_DAY[d1]} in week {week}; if that can't be done, cancel it that week.",
              fallback),
             None,
             (f"Either {w1} or {w2} would work for {my}.", any_of),
             (f"First choice for {my}: {w1}. Second choice: {w2}.", ranked),
             (f"{my[0].upper() + my[1:]} could go to {w1}, or failing that {w2}.", ranked),
             (f"Try to move {my} to {FULL_DAY[d1]} in week {week}, and if it won't fit, cancel it for that week.",
              fallback)]
    if t == 3:  # a move and a relation about the same class
        lectures = [x for x in ctx.mine(SessionKind.LECTURE) if x.course == s.course]
        if s.kind != SessionKind.LECTURE or len(lectures) < 2:
            return None
        T = ctx.title(s)
        texts[3] = (f"Move my {FULL_DAY[d]} {h} {T} lecture to {w1}, and keep it off the days of my other {T} lectures.",
                    [Move(id="a", which=_ref(s, ctx, "lecture", course=True), to=to1),
                     Relate(which=ClassRef(same="a"), relation="different_day",
                            other=ClassRef(course=s.course, kind="lecture"))])
    text, found = texts[t]
    return text, IntentOutput(action="compile", intents=found), {}, False


def move_pairs(ctx: _Ctx, t: int):
    """One move said the way ``choices`` says a choice ("could you move it to A?" beside "... to A or B?",
    "ideally it would move to A." beside "... A; if not, B"): a plain move has one destination and no labels.
    v11 read plain moves as choices with a made-up second option (45 of its 785 move rows were choices)."""
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, h = ctx.at(s)
    k = ctx.kind_word(s)
    w, to = ctx.to_time(s, ctx.other_day(d))
    my = f"my {FULL_DAY[d]} {h} {k}"
    texts = [f"Could you move {my} to {w}?",
             f"Ideally {my} would move to {w}.",
             f"{my[0].upper() + my[1:]} could go to {w}.",
             f"Please move {my} to {w}; it can't stay where it is.",
             f"{w[0].upper() + w[1:]} would work for {my}.",
             f"My choice for {my} is {w}.",
             f"Try to move {my} to {w}.",
             f"{my[0].upper() + my[1:]} has to move to {w}; where it is now clashes with a meeting."]
    text = texts[t]
    gold = Move(which=_ref(s, ctx, k, course=False), to=to, hard=placement_is_hard(text))
    return text, IntentOutput(action="compile", intents=[gold]), {"move": s.id}, False


def limit_firmness(ctx: _Ctx, t: int):
    """Each limit on a week said as a must and as a wish, so how firm it is comes from what the message says,
    not from which limit it is (v11 saw one firm limit wording in six); and hours in a row in more words
    ("keep my runs to 3", "without a break"), which v11-read-100 read as a limit a day."""
    rng = ctx.rng
    n, h, d = rng.choice([1, 2, 3]), rng.choice([2, 3, 4]), rng.choice([2, 3, 4])
    measure, k, said = rng.choice([
        ("classes_per_day", n, f"no more than {_WORDS[n]} classes on any day"),
        ("hours_per_day", h, f"at most {_WORDS[h]} hours of teaching a day"),
        ("days_per_week", d, f"my teaching on {_WORDS[d]} days a week at most"),
        ("gap", 1, "at least an hour free between any two of my classes")])
    texts = [f"I need {said}.",
             f"If possible, I'd like {said}.",
             f"Please give me {said}; this is a must.",
             f"Ideally, {said}.",
             f"Please keep my runs of teaching to {_WORDS[h]} hours.",
             f"I can't teach more than {_WORDS[h]} hours without a break.",
             f"{said[0].upper() + said[1:]} is something I have to have.",
             f"Where possible, {said}, please."]
    text = texts[t]
    gold = (LoadLimit(max_consecutive=h, hard=necessary(text)) if t in (4, 5)
            else Limit(measure=measure, n=k, hard=necessary(text)))
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def relation_pairs(ctx: _Ctx, t: int):
    """A practical placed against its course's lectures (``other``, one of them or each), and lectures
    against each other, each as a must and as a wish: v11 saw ``other`` only with right_after and never a firm
    relation, and split "the practical on the same day as one of the lectures" into two relations."""
    from core.instance import SessionKind

    lectures: dict[str, int] = {}
    for s in ctx.mine(SessionKind.LECTURE):
        lectures[s.course] = lectures.get(s.course, 0) + 1
    both = sorted({s.course for s in ctx.mine(SessionKind.PRACTICAL)} & {c for c, m in lectures.items() if m >= 2})
    if not both:
        return None
    course = ctx.rng.choice(both)
    T = ctx.instance.course_title(course)
    lec, lab = ClassRef(course=course, kind="lecture"), ClassRef(course=course, kind="practical")
    texts = [(f"The {T} practical has to be on the same day as one of its lectures.",
              Relate(which=lab, relation="same_day", other=lec, any=True)),
             (f"If possible, put the {T} practical on a day with a {T} lecture.",
              Relate(which=lab, relation="same_day", other=lec, any=True)),
             (f"Keep the {T} practical off the days of the {T} lectures.",
              Relate(which=lab, relation="different_day", other=lec)),
             (f"The {T} practical must come after the {T} lectures each week.",
              Relate(which=lab, relation="after", other=lec)),
             (f"I'd like the {T} practical straight after a {T} lecture.",
              Relate(which=lab, relation="right_after", other=lec, any=True)),
             (f"Two {T} lectures on one day won't work for me; they must be on different days.",
              Relate(which=lec, relation="different_day")),
             (f"Ideally the {T} practical would fall on the same day as a {T} lecture.",
              Relate(which=lab, relation="same_day", other=lec, any=True)),
             (f"My {T} lectures need to be on separate days.", Relate(which=lec, relation="different_day"))]
    text, gold = texts[t]
    gold = gold.model_copy(update={"hard": necessary(text)})
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def out_for_me(ctx: _Ctx, t: int):
    """Times someone cannot teach said without "can't" or "unavailable" ("Wednesdays are out for me", "count
    me out on Fridays"), beside wishes said as plainly ("I'd rather not teach on Fridays"). v11 read "out for
    me" as a wish: no training row said it."""
    cal, rng = ctx.instance.calendar, ctx.rng
    if ctx.week >= cal.weeks:
        return None
    day = rng.choice(cal.days)
    D = FULL_DAY[day]
    part, parts = rng.choice([("morning", "mornings"), ("afternoon", "afternoons")])
    w = rng.randint(ctx.week + 1, cal.weeks)
    days, half = TimeSpec(days=[day]), TimeSpec(days=[day], part=part)
    texts = [(f"{D}s are out for me.", Unavailable(time=days)),
             (f"{D} {parts} are out for me in week {w}.", Unavailable(time=half, period=Period(weeks=[w]))),
             (f"I'd rather not teach on {D}s.", Avoid(time=days)),
             (f"{D} {parts} don't work for me at all.", Unavailable(time=half)),
             (f"Count me out on {D}s from week {w}.", Unavailable(time=days, period=Period(from_week=w))),
             (f"{D}s aren't ideal for me.", Avoid(time=days)),
             (f"I'm tied up on {D} {parts} for the rest of the term.",
              Unavailable(time=half, period=Period(relative="rest_of_semester"))),
             (f"If you can, leave my {D}s free.", Avoid(time=days))]
    text, gold = texts[t]
    return text, IntentOutput(action="compile", intents=[gold]), {}, False


def _later_same_day(ctx: _Ctx, s) -> tuple[str, TimeSpec, str] | None:
    """Somewhere else on the day ``s`` is on now, in words without the day: the other part of the day, or an
    hour (not lunch, not where it is, room for its length)."""
    cal, rng = ctx.instance.calendar, ctx.rng
    d, _, _ = ctx.at(s)
    q0, lunch = ctx.timetable[s.id].slot, cal.lunch_slot
    morning = lunch is not None and q0 < lunch
    hours = [q for q in range(cal.slots_per_day - s.duration + 1)
             if q != q0 and (lunch is None or not q <= lunch < q + s.duration)]
    options = [("the afternoon", TimeSpec(days=[d], part="afternoon"), "afternoon") if morning else
               ("the morning", TimeSpec(days=[d], part="morning"), "morning")]
    if hours:
        q = rng.choice(hours)
        options.append((hour(q), TimeSpec(days=[d], at=[clock(q)]), "hour"))
    if morning:
        options.append(("after lunch", TimeSpec(days=[d], part="afternoon"), "afternoon"))
    w, to, how = rng.choice(options)
    later = how == "afternoon" or (how == "hour" and minutes_of(to.at[0]) > minutes_of(clock(q0)))
    return w, to, how, later


def minutes_of(c: str) -> int:
    h, m = c.split(":")
    return int(h) * 60 + int(m)


def same_day_part(ctx: _Ctx, t: int):
    """A move to another time of the same day, said without the day again ("move my Friday 9 am class to the
    afternoon"): the destination is that day. The v12 reader spread such moves over other days (the
    head-to-head's move-afternoon); ``language.checks._same_day`` repairs it, this teaches it."""
    mine = ctx.mine()
    if not mine:
        return None
    s = ctx.rng.choice(mine)
    d, _, h = ctx.at(s)
    got = _later_same_day(ctx, s)
    if got is None:
        return None
    w, to, how, later = got
    k, T, D = ctx.kind_word(s), ctx.title(s), FULL_DAY[d]
    at = "at " + w if how == "hour" else w if w == "after lunch" else "in " + w
    texts = [f"Move my {D} {h} {k} to {w}.",
             f"Could my {T} {k} on {D} at {h} move to {w}, same day?",
             f"Please shift my {D} {h} {k} to {w} instead.",
             f"My {D} {h} {T} {k} would suit me better {at}.",
             f"Can the {k} I have on {D} at {h} be {'later' if later else 'earlier'} in the day, {at}?",
             f"I'd like my {h} {k} on {D} moved to {w}, if possible.",
             f"Keep my {D} {h} {k} on {D} but move it to {w}.",
             f"Same day, different time: my {D} {h} {T} {k} to {w}, please."]
    text = texts[t]
    gold = Move(which=_ref(s, ctx, k, course=t in (1, 3, 7)), to=to, hard=placement_is_hard(text))
    return text, IntentOutput(action="compile", intents=[gold]), {"move": s.id}, False


ANSWER_LEAD, ANSWER_LEAD_HELD = _split(["", "", "It's ", "I mean ", "Sorry, ", "That would be ", "Oh, "], 2)


def _lead(ctx: _Ctx, held: bool) -> str:
    return ctx.rng.choice(ANSWER_LEAD_HELD if held else ANSWER_LEAD)


def answered(ctx: _Ctx, t: int):
    """A request too vague to act on, then its sender's answer to the question it was sent back with, as one
    message (``web.world.joined``: the request, a space, the answer). Read whole: the answer fills what the
    request left out. The v12 reader asked again or dropped the answer's details (the head-to-head's
    move-no-class-named passed in one run of three); no training row had an answer after a question."""
    cal, rng = ctx.instance.calendar, ctx.rng
    mine = ctx.mine()
    if not mine or ctx.week >= cal.weeks:
        return None
    s = rng.choice(mine)
    d, c, h = ctx.at(s)
    k, T, D = ctx.kind_word(s), ctx.title(s), FULL_DAY[d]
    w = rng.randint(ctx.week + 1, cal.weeks)
    to_day = ctx.other_day(d)
    to_words, to = ctx.to_time(s, to_day)
    q = rng.choice([x for x in range(cal.slots_per_day - s.duration + 1) if x != cal.lunch_slot])
    x = rng.choice(cal.days)
    lead = _lead(ctx, t >= 6)
    cap = (lambda u: u[0].upper() + u[1:]) if not lead else (lambda u: u)
    named = ClassRef(course=s.course, kind=None if k == "class" else s.kind.value, day=d, time=c)
    texts = [  # (request, answer, gold)
        ("Move my class to " + FULL_DAY[to_day] + ".",
         cap(f"the {T} {k} on {D} at {h}, to {FULL_DAY[to_day]} {hour(q)} in week {w}."),
         Move(which=named, to=TimeSpec(days=[to_day], at=[clock(q)]), period=Period(weeks=[w]))),
        ("Please move one of my classes.", cap(f"my {T} {k} on {D} at {h}; {to_words} would be fine."),
         Move(which=named, to=to)),
        (f"I'll be away one of the days in week {w}.", cap(f"{FULL_DAY[x]}, the whole day."),
         Unavailable(time=TimeSpec(days=[x]), period=Period(weeks=[w]))),
        (f"Please cancel my {k}." if k != "class" else "Please cancel one of my classes.",
         cap(f"the {T} {k} on {D} at {h}, in week {w}."),
         Cancel(which=named, period=Period(weeks=[w]))),
        (f"Could my {T} {k} move?", cap(f"the one on {D} at {h}, to {to_words}, in week {w} only."),
         Move(which=named, to=to, period=Period(weeks=[w]))),
        ("Something's come up and I can't teach one morning soon.", cap(f"{FULL_DAY[x]} morning of week {w}."),
         Unavailable(time=TimeSpec(days=[x], part="morning"), period=Period(weeks=[w]))),
        ("I'd like a different slot for one of my classes.",
         cap(f"the {T} {k} on {D} at {h}. {to_words[0].upper() + to_words[1:]} would work."),
         Move(which=named, to=to)),
        (f"I won't be able to make it on one of the days in week {w}.", cap(f"{FULL_DAY[x]}."),
         Unavailable(time=TimeSpec(days=[x]), period=Period(weeks=[w]))),
    ]
    request, answer, gold = texts[t]
    if isinstance(gold, Move) and len([o for o in mine if o.course == s.course and o.kind == s.kind
                                       and ctx.timetable[o.id].day == d and ctx.timetable[o.id].slot ==
                                       ctx.timetable[s.id].slot]) != 1:
        return None
    text = f"{request} {lead}{answer}"
    if isinstance(gold, Move | Avoid):
        gold = gold.model_copy(update={"hard": placement_is_hard(text)})
    return text, IntentOutput(action="compile", intents=[gold]), {"move": s.id} if isinstance(gold, Move) else {}, False


REGISTRY: list[Generator] = [
    Generator("firmness", firmness, 6, (4, 5), "firmness"),
    Generator("only_firmness", only_firmness, 6, (4, 5), "firmness"),
    Generator("which_to", which_to, 6, (4, 5), "moves"),
    Generator("relative_span", relative_span, 6, (4, 5), "periods"),
    Generator("multi", multi, 6, (4, 5), "multi"),
    Generator("club_more", club_more, 6, (4, 5), "clubs"),
    Generator("vague", vague, 6, (4, 5), "vague"),  # added last: the others keep their seeds and rows
    # intents-v4: what the v3 model missed on the suites ("I am not on campus otherwise", "until the end of term")
    Generator("reason_firmness", reason_firmness, 6, (4, 5), "reasons"),
    Generator("term_span", term_span, 6, (4, 5), "terms"),
    # intents-v5: what v4 lost on the test splits ("... please. hard for me." read as a must; a week or
    # "this semester" said outright, yet asked for)
    Generator("difficulty", difficulty, 6, (4, 5), "difficulty"),
    Generator("dated", dated, 8, (6, 7), "dated"),
    # intents-v7: what the full pipeline's reading missed (the request battery of 2026-10-26): an exception
    # said after a request ("but not on Friday") dropped, "I can only teach afternoons" read as a wish, and a
    # message of two different requests misread (a time made up for a class named by its day)
    Generator("exception", exception, 6, (4, 5), "exceptions"),
    Generator("only_times", only_times, 6, (4, 5), "only_times"),
    Generator("compound", compound, 6, (4, 5), "compound"),
    # schema 5: weeks left out and odd or even weeks, said by the reader and counted here
    Generator("week_patterns", week_patterns, 8, (6, 7), "week_patterns", since=5),
    # schema 6: limits on a week, classes placed against others, and choices between items
    Generator("limits", limits, 8, (6, 7), "limits", since=6),
    Generator("relations", relations, 8, (6, 7), "relations", since=6),
    Generator("choices", choices, 8, (6, 7), "choices", since=6),
    # intents-v12: what v11 got wrong beside them: plain moves read as choices, limits and relations never
    # read as musts, a practical against its lectures split in two, and "X are out for me" read as a wish
    Generator("move_pairs", move_pairs, 8, (6, 7), "move_pairs"),
    Generator("limit_firmness", limit_firmness, 8, (6, 7), "limit_firmness", since=6),
    Generator("relation_pairs", relation_pairs, 8, (6, 7), "relation_pairs", since=6),
    Generator("out_for_me", out_for_me, 8, (6, 7), "out_for_me"),
    # intents-v13: what the head-to-head found the v12 reader missing (runs/head_to_head/v1, v3): a move to
    # another time of the same day, and a request completed by the answer to its clarifying question
    Generator("same_day_part", same_day_part, 8, (6, 7), "same_day_part"),
    Generator("answered", answered, 8, (6, 7), "answered"),
]
SENDERS = {"club_more": "student_or_teacher"}


def generated(instance: Instance, per_generator: int = 60, held_per_generator: int = 30, seed: int = 200,
              names: list[str] | None = None, boost: dict[str, int] | None = None) -> list[IntentExample]:
    """Every registered generator's examples: ``per_generator`` on its train wordings (a tenth to val) and
    ``held_per_generator`` on its held-out ones, whose split is their suite's name. Each generator has its
    own seed, so adding one leaves the others' rows as they were. ``boost``: a generator's own number of train
    rows instead of ``per_generator`` (the newest and weakest kinds, upsampled instead of another epoch); its
    first ``per_generator`` rows are the ones it makes without it."""
    from . import intent_corpus

    out: list[IntentExample] = []
    for i, g in enumerate(REGISTRY):
        if names is not None and g.name not in names:
            continue
        sender = SENDERS.get(g.name)
        if sender:
            intent_corpus.SENDERS.setdefault(g.name, sender)
        train = _examples(instance, random.Random(seed + 10 * i), {g.name: (g.name, g.fn)}, (boost or {}).get(g.name, per_generator), None,
                          templates=g.train_templates)
        held = _examples(instance, random.Random(seed + 10 * i + 5), {f"{g.name}@{g.suite}": (g.name, g.fn)},
                         held_per_generator, None, templates=g.held_out)
        for ex in held:
            ex.split = g.suite
        out += train + held
    return out


GENERATORS: dict[str, Callable] = {g.name: g.fn for g in REGISTRY}
