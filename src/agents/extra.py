"""Extra classes: "On Thursday in week 11 I want an extra class."

The timetable is a fixed set of weekly sessions and every other request limits where they go; an
extra class adds one, for one week. It is handled outside the parser and the solver, like a swap:

1. ``ExtraReader`` finds which class it is (a copy of one of the sender's weekly sessions: same
   course, section, length and kind of room), the week, and any day or time asked for. When the
   sender teaches several classes and the message does not say which, or no week is given, it
   returns a question. A clarification answer is appended to the text, so the last week named wins.
2. ``free_placements`` lists the times in that week when the teacher, the section and a suitable
   room are all free, every hard rule in force that week is kept (the lunch break, closures,
   absences, consecutive-hour limits), and nothing else has to move.
3. ``block_constraints`` turns the chosen placement into hard Tier 2 constraints for that week (the
   teacher, each section and the room are taken at that hour), so later solves of the week keep
   clear of it without the solver having to know the extra session.

Deterministic throughout: the fine-tuned model was never trained on extra classes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from core.instance import ExtraClass, Instance, Session, SessionKind
from core.schemas import Constraint, ConstraintType, Placement, Role, Scope, Source, Tier, When
from core.semantics import PLACEMENT_TYPES, consecutive_entities, occupied, room_compatible, violates
from language.corpus import FULL_DAY, hour
from language.rule_parser import EXTRA, RuleParser

TEACHERS = (Role.FACULTY, Role.HOD, Role.GUEST_FACULTY)
EXTRA_TEXT = re.compile(rf"{EXTRA.pattern}|\b(one more|another)\s+(class|lecture|session)\b", re.IGNORECASE)
_WEEK = re.compile(r"\bweek\s+(\d{1,2})\b", re.IGNORECASE)
RULE = "extra class"  # Source.rule of the block constraints


def looks_like_extra(text: str) -> bool:
    return bool(EXTRA_TEXT.search(text))


@dataclass
class ExtraReading:
    template: Session | None = None  # the weekly session the extra class copies
    week: int | None = None
    days: list[str] | None = None
    slots: list[int] | None = None  # start slots asked for
    question: str | None = None  # set when the message does not say enough
    options: list[str] = field(default_factory=list)


class ExtraReader:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.rules = RuleParser(instance)

    def label(self, s: Session) -> str:
        inst = self.instance
        groups = ", ".join(inst.group_by_id[g].name for g in s.groups if g in inst.group_by_id)
        kind = "" if s.kind == SessionKind.LECTURE else f" {s.kind.value}"
        return f"{inst.course_title(s.course)}{kind} ({groups})" if groups else f"{inst.course_title(s.course)}{kind}"

    def read(self, text: str, sender: str) -> ExtraReading:
        inst = self.instance
        own = [s for s in inst.sessions if s.faculty == sender]
        if not own:
            return ExtraReading(question="You have no classes in the timetable to add an extra one to.")
        low = text.lower()
        named = [inst.session_by_id[i] for i in self.rules.sessions_named(text, sender)] or own
        groups = [g.id for g in inst.groups if re.search(rf"\b{re.escape(g.name.lower())}\b", low)]
        if groups:
            named = [s for s in named if set(groups) & set(s.groups)] or named
        if not re.search(r"\b(practical|lab session|tutorial)\b", low):
            # an extra class is a lecture unless it says otherwise
            named = [s for s in named if s.kind == SessionKind.LECTURE] or named
        classes: dict[tuple, Session] = {}
        for s in named:
            classes.setdefault((s.course, s.kind, tuple(s.groups)), s)
        weeks = [int(w) for w in _WEEK.findall(text)]  # the last one named: an answer comes after the request
        week = weeks[-1] if weeks else None
        if week is not None and not 1 <= week <= inst.calendar.weeks:
            week = None
        reading = ExtraReading(week=week, days=self.rules.days(text), slots=self.rules.slots(text))
        if len(classes) > 1:
            reading.options = [self.label(s) for s in classes.values()]
            reading.question = f"Which class is the extra one for: {_or(reading.options)}?"
            return reading
        reading.template = next(iter(classes.values()))
        if week is None:
            reading.question = (f"Which week should the extra {self.label(reading.template)} class be in? "
                                f"Teaching weeks are 1 to {inst.calendar.weeks}.")
        return reading


def extra_session(template: Session, case_id: str) -> Session:
    return template.model_copy(update={"id": f"X-{case_id}"})


def free_placements(instance: Instance, session: Session, week: int, assignment: Mapping[str, Placement],
                    constraints: Iterable[Constraint], extras: Iterable[ExtraClass] = (),
                    days: list[str] | None = None, slots: list[int] | None = None,
                    skip: Iterable[str] = ()) -> list[Placement]:
    """Where ``session`` can be held in ``week`` without moving anything: best first (the teacher's own
    soft wishes kept, then the session's usual room, then the earliest time)."""
    inst, cal = instance, instance.calendar
    skip = set(skip)
    busy: dict[tuple[str, str], set[tuple[int, int]]] = {}  # (kind, id) -> (day, slot)

    def take(s: Session, p: Placement) -> None:
        d = inst.day_index(p.day)
        for key in [("faculty", s.faculty), ("room", p.room), *(("group", g) for g in s.groups)]:
            busy.setdefault(key, set()).update(occupied(s, d, p.slot))

    for sid, p in assignment.items():
        if sid in inst.session_by_id and sid not in skip:
            take(inst.session_by_id[sid], p)
    for x in extras:
        if x.session.id != session.id:
            take(x.session, x.placement)

    active = [c for c in constraints if c.active_in(week)]
    hard = [c for c in active if c.hard and _covers(c.scope, session)]
    wishes = [c for c in active if not c.hard and c.owner == session.faculty and _covers(c.scope, session)]
    mine = [("faculty", session.faculty), *(("group", g) for g in session.groups)]
    usual = assignment.get(next((s.id for s in inst.sessions if s.course == session.course and s.kind == session.kind
                                 and s.groups == session.groups and s.faculty == session.faculty), ""))
    rooms = sorted((r for r in inst.rooms if room_compatible(inst, session, r)),
                   key=lambda r: (usual is None or r.id != usual.room, r.capacity))
    out: list[tuple[tuple, Placement]] = []
    for day in [d for d in cal.days if days is None or d in days]:
        d = inst.day_index(day)
        for slot in [q for q in range(cal.slots_per_day) if slots is None or q in slots]:
            if slot + session.duration > cal.slots_per_day:
                continue
            cells = set(occupied(session, d, slot))
            if any(cells & busy.get(key, set()) for key in mine):
                continue
            if not _within_limits(inst, hard, session, d, cells, busy):
                continue
            for room in rooms:
                if cells & busy.get(("room", room.id), set()):
                    continue
                if any(c.type in (ConstraintType.UNAVAILABLE, ConstraintType.AVOID, ConstraintType.PREFER,
                                  ConstraintType.REQUIRE_ROOM) and violates(inst, c, session, d, slot, room)
                       for c in hard):
                    continue
                kept = sum(violates(inst, c, session, d, slot, room) for c in wishes
                           if c.type in PLACEMENT_TYPES)  # a wish about a placement, not a limit or a relation
                out.append(((kept, usual is None or room.id != usual.room, d, slot), Placement(day=day, slot=slot, room=room.id)))
                break  # the best room for this time
    return [p for _, p in sorted(out, key=lambda x: x[0])]


def _covers(scope: Scope, s: Session) -> bool:
    if scope.faculty is not None:
        return scope.faculty == s.faculty
    if scope.group is not None:
        return scope.group in s.groups
    if scope.session is not None:
        return scope.session == s.id
    if scope.course is not None:
        return scope.course == s.course
    return True  # everyone, or a room (``violates`` checks the room)


def _within_limits(inst: Instance, hard: list[Constraint], s: Session, day: int, cells: set[tuple[int, int]],
                   busy: dict[tuple[str, str], set[tuple[int, int]]]) -> bool:
    """No consecutive-hours limit in force is broken for the teacher or a section."""
    for c in hard:
        if c.type != ConstraintType.MAX_CONSECUTIVE or not c.limit:
            continue
        ents = set(consecutive_entities(inst, c.scope)) & {("faculty", s.faculty), *(("group", g) for g in s.groups)}
        for ent in ents:
            taken = {q for dd, q in busy.get(ent, set()) | cells if dd == day}
            run = best = 0
            for q in range(inst.calendar.slots_per_day):
                run = run + 1 if q in taken else 0
                best = max(best, run)
            if best > c.limit:
                return False
    return True


def block_constraints(x: ExtraClass, owner: str) -> list[Constraint]:
    """The teacher, each section and the room are taken at the extra class's hour in its week."""
    s, p = x.session, x.placement
    when = When(days=[p.day], slots=list(range(p.slot, p.slot + s.duration)), weeks=[x.week])
    scopes = [("F", Scope(faculty=s.faculty)), *((g, Scope(group=g)) for g in s.groups), ("R", Scope(room=p.room))]
    return [Constraint(id=f"{s.id}-{tag}", type=ConstraintType.UNAVAILABLE, hard=True, tier=Tier.COMMITMENT,
                       owner=owner, scope=scope, when=when, source=Source(request=x.request, rule=RULE))
            for tag, scope in scopes]


def when_text(instance: Instance, p: Placement, week: int) -> str:
    room = instance.room_by_id[p.room].name if p.room in instance.room_by_id else p.room
    return f"{FULL_DAY[p.day]} at {hour(p.slot)} in {room}, week {week}"


def _or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]
