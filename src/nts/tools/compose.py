"""Compose tools (from coordinator-v6): what a request says that one rule or one move cannot.

    move_to_window  one class moved into a span of hours on one or more days ("Tuesday or Friday afternoon")
    limit_load      at most so many classes a day or teaching days a week, or free hours between classes
    relate_classes  classes placed against others: the same day, a different day, right after, not on the
                    same day as, a day apart
    choose          earlier changes made alternatives: one of them ("Tuesday morning or Friday afternoon"),
                    or in order ("ideally Friday, else Monday")
    set_weeks       an earlier change held only in even or odd weeks, or not in some weeks ("except week 16")

The kernel kinds are core-rework's (load, relate, any_of, ranked: ``nts.kernel.semantics``), so the solver,
the validator and ``check_draft`` treat them as any other rule. Like the draft tools, each is one numbered
change; choose and set_weeks rewrite earlier changes, which keep their number.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nts.kernel.instance import Session
from nts.kernel.schemas import Constraint, ConstraintType, Validity, When
from nts.kernel.schemas import Scope as Who
from nts.state import ConstraintAdded, ConstraintRemoved

from . import words
from .draft import (
    CANCEL,
    OFFICE,
    _changes,
    _draft,
    _may,
    _pick,
    _range_missed,
    _record,
    _slots,
    _standing_missed,
    _tier,
    _validity,
    _weeks,
)
from .names import Directory
from .registry import Context, Refusal, Scope, tool


class MoveToWindow(BaseModel):
    course: str
    kind: Literal["lecture", "tutorial", "practical"] | None = None
    section: str | None = None
    on_day: str | None = Field(None, description="Where it is now (to pick one class)")
    at_hour: str | None = None
    to_days: list[str] = Field(min_length=1, description="Any of these days")
    from_hour: str | None = Field(None, description="The span it must sit in: morning 9 am to 1 pm, afternoon 2 pm on")
    to_hour: str | None = Field(None, description="Until ('1 pm'); none: to the end of the day")
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = Field(None, description="= from_week for one week")
    firm: bool = True


@tool(Scope.COMPOSE)
def move_to_window(ctx: Context, args: MoveToWindow) -> dict:
    """Move one class into a span of hours on any of some days ("Friday morning", "Tuesday or Thursday
    afternoon"): the whole class sits inside the span."""
    d = Directory(ctx.state.instance)
    valid, tt = _weeks(ctx, args.from_week, args.to_week)
    s = _pick(ctx, d, tt, args.course, args.kind, args.section, None, args.on_day, args.at_hour)
    _may(ctx, Who(session=s.id))
    slots = _slots(d, args.from_hour, args.to_hour)
    if slots is not None and len(slots) < s.duration:
        raise Refusal("bad_arguments", f"a {s.duration}-hour class does not fit in {words.hours(slots)}")
    tier, why = _tier(ctx, args.firm, None)
    c = Constraint(id="new", type=ConstraintType.PREFER, hard=args.firm, tier=tier, owner=s.faculty,
                   scope=Who(session=s.id), when=When(days=[d.day(x) for x in args.to_days], slots=slots),
                   justification=why, valid=valid)
    result = _record(ctx, [c], None)
    if note := _range_missed(ctx.request, valid) or _standing_missed(ctx.request, valid):
        result["note"] = note
    return result


class LimitLoad(BaseModel):
    measure: Literal["classes_per_day", "hours_per_day", "days_per_week", "gap"] = Field(
        description="gap: free hours between two classes on a day (at least n); the others: at most n")
    n: int = Field(ge=1)
    teacher: str | None = Field(None, description="Name, or 'me'")
    section: str | None = None
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = None
    firm: bool = True
    reason: str | None = None


@tool(Scope.COMPOSE)
def limit_load(ctx: Context, args: LimitLoad) -> dict:
    """Limit how a teacher's or a section's week is spread: classes or hours a day, teaching days a week, or
    free hours between two classes."""
    d = Directory(ctx.state.instance)
    if bool(args.teacher) == bool(args.section):
        raise Refusal("bad_arguments", "name a teacher or a section")
    scope = Who(faculty=d.teacher(args.teacher, ctx.actor).id) if args.teacher else Who(group=d.section(args.section).id)
    _may(ctx, scope)
    tier, why = _tier(ctx, args.firm, args.reason)
    c = Constraint(id="new", type=ConstraintType.LOAD, hard=args.firm, tier=tier,
                   owner=scope.faculty or (ctx.actor if ctx.actor != OFFICE else None), scope=scope,
                   measure=args.measure, limit=args.n, justification=why,
                   valid=_validity(ctx, args.from_week, args.to_week))
    return _record(ctx, [c], args.reason)


class Classes(BaseModel):
    """Every class of the course that matches (all its lectures, say); on_day and at_hour narrow it to one."""

    course: str
    kind: Literal["lecture", "tutorial", "practical"] | None = None
    section: str | None = None
    on_day: str | None = Field(None, description="Only the one there now")
    at_hour: str | None = None


class RelateClasses(BaseModel):
    classes: Classes = Field(description="The classes placed")
    relation: Literal["same_day", "different_day", "right_after", "after", "day_gap"] = Field(
        description="right_after: starts as the other ends; after: later the same day; day_gap: more than a day apart")
    others: Classes | None = Field(None, description="Placed against these; none: the classes against each other")
    one_of_them: bool = Field(False, description="true: against one of the others ('right after one of its "
                                                  "lectures'); false: against each of them")
    negate: bool = Field(False, description="The relation must not hold ('not on the same day as')")
    from_week: int | None = Field(None, description="No weeks: the whole term")
    to_week: int | None = None
    firm: bool = True


def _select(ctx: Context, d: Directory, tt, sel: Classes) -> list[Session]:
    inst = d.instance
    courses = d.courses(sel.course)
    who = ctx.actor if ctx.actor != OFFICE else None
    sec = d.section(sel.section).id if sel.section else None
    day = d.day(sel.on_day) if sel.on_day else None
    at = d.slot(sel.at_hour) if sel.at_hour else None
    found = []
    for s in inst.sessions:
        p = tt.assignment.get(s.id)
        if (s.course in courses and (sel.kind is None or s.kind.value == sel.kind) and (who is None or s.faculty == who)
                and (sec is None or sec in s.groups) and (day is None or (p is not None and p.day == day))
                and (at is None or (p is not None and p.slot <= at < p.slot + s.duration))):
            found.append(s)
    if not found:
        raise Refusal("no_such_class", f"no {sel.course} class matches"
                      + (" among your classes" if ctx.actor != OFFICE else ""))
    return found


@tool(Scope.COMPOSE)
def relate_classes(ctx: Context, args: RelateClasses) -> dict:
    """Place classes against others: on the same day, a different day, right after, later that day, or more
    than a day apart (or, with negate, not so)."""
    d = Directory(ctx.state.instance)
    valid, tt = _weeks(ctx, args.from_week, args.to_week)
    mine = _select(ctx, d, tt, args.classes)
    others = _select(ctx, d, tt, args.others) if args.others else None
    if others is None and len(mine) < 2:
        raise Refusal("bad_arguments", "one class against itself: name the others it is placed against")
    for s in mine + (others or []):
        _may(ctx, Who(session=s.id))
    tier, why = _tier(ctx, args.firm, None)
    c = Constraint(id="new", type=ConstraintType.RELATE, hard=args.firm, tier=tier, owner=mine[0].faculty,
                   scope=Who(faculty=mine[0].faculty), sessions=[s.id for s in mine],
                   others=None if others is None else [s.id for s in others if s.id not in {m.id for m in mine}],
                   relation=args.relation, any_other=args.one_of_them, negate=args.negate, justification=why,
                   valid=valid)
    result = _record(ctx, [c], None)
    if note := _range_missed(ctx.request, valid) or _standing_missed(ctx.request, valid):
        result["note"] = note
    return result


class Choose(BaseModel):
    changes: list[int] = Field(min_length=2, description="Change numbers, the first choice first")
    ranked: bool = Field(False, description="true: in order of preference; false: any one of them")


@tool(Scope.COMPOSE)
def choose(ctx: Context, args: Choose) -> dict:
    """Make earlier changes alternatives: any one of them must hold, or (ranked) the first if it can, else the
    next. Draft each choice first (move_class, constrain), then choose with their numbers."""
    draft, changes = _draft(ctx), _changes(ctx)
    if missing := [n for n in args.changes if n not in changes]:
        raise Refusal("unknown_change", f"there is no change {missing[0]}", changes=sorted(changes))
    members = []
    for n in args.changes:
        ids = changes[n]
        if len(ids) != 1 or ids[0].startswith(CANCEL):
            raise Refusal("bad_arguments", f"change {n} is not one rule or move, so it cannot be a choice")
        members.append(draft.world.constraints[ids[0]])
    for n in args.changes:
        draft.add(ConstraintRemoved(id=changes.pop(n)[0]))
    first = members[0]
    hard = all(m.hard for m in members)
    tier, why = _tier(ctx, hard, None)

    def made(kind: ConstraintType, hard: bool, tag: str) -> Constraint:
        return Constraint(id="new", type=kind, hard=hard, tier=tier if hard else _tier(ctx, False, None)[0],
                          owner=first.owner, scope=Who(faculty=first.owner) if first.owner else Who(),
                          justification=why if hard else first.justification,
                          members=[m.model_copy(update={"id": f"{m.id}{tag}"}) for m in members])

    out = [made(ConstraintType.ANY_OF, hard, ".any")]
    if args.ranked:  # one of them must hold; the order is a wish on top
        out.append(made(ConstraintType.RANKED, False, ".rank").model_copy(update={"weight": 5.0}))
    return _record(ctx, out, None)


class SetWeeks(BaseModel):
    change: int
    every: Literal["week", "even", "odd"] = "week"
    except_weeks: list[int] = Field(default_factory=list, description="Weeks it does not hold in")


@tool(Scope.COMPOSE)
def set_weeks(ctx: Context, args: SetWeeks) -> dict:
    """Make an earlier change hold only in even or odd weeks, or not in some weeks ("except week 16")."""
    draft, changes = _draft(ctx), _changes(ctx)
    if args.change not in changes:
        raise Refusal("unknown_change", f"there is no change {args.change}", changes=sorted(changes))
    n = ctx.state.instance.calendar.weeks
    if bad := [w for w in args.except_weeks if not 1 <= w <= n]:
        raise Refusal("unknown_week", f"week {bad[0]} is not in term", weeks=f"1-{n}")
    if args.every == "week" and not args.except_weeks:
        raise Refusal("bad_arguments", "say even or odd weeks, or the weeks it does not hold in")
    out = []
    for cid in changes[args.change]:
        if cid.startswith(CANCEL):
            raise Refusal("bad_arguments", "a cancellation is for one week already")
        c = draft.world.constraints[cid]
        lo, hi = c.valid.from_week or 1, c.valid.to_week or n
        weeks = None if args.every == "week" else [w for w in range(lo, hi + 1) if w % 2 == (args.every == "odd")]
        new = c.model_copy(update={
            "when": When(days=c.when.days, slots=c.when.slots, weeks=weeks if weeks is not None else c.when.weeks),
            "valid": Validity(from_week=c.valid.from_week, to_week=c.valid.to_week,
                              except_weeks=sorted({*c.valid.except_weeks, *args.except_weeks}))})
        draft.add(ConstraintRemoved(id=cid))
        try:
            draft.add(ConstraintAdded(constraint=new))
        except ValueError as e:
            raise Refusal("bad_change", str(e)) from None
        out.append(words.constraint(draft.world.instance, new))
    return {"change": args.change, "now": out}


COMPOSE_TOOLS = [move_to_window, limit_load, relate_classes, choose, set_weeks]
