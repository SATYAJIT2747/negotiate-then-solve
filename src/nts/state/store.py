"""The event log, the drafts agents work in, and the one gate every change goes through.

    store = bootstrap(instance, policy_constraints(instance), path="runs/agentic/dept.jsonl")
    draft = store.draft(actor="F-102", episode="ep-7")
    draft.add(ConstraintAdded(constraint=c))
    draft.add(TimetableSet(week=7, assignment=solved.assignment))
    store.commit(draft)      # checked by the kernel, or InvalidChange; StaleDraft if the store moved on

A draft is a private copy: nothing an agent does reaches the store until ``commit``, and ``commit``
re-checks the result with the kernel's validator, whatever the agent said about it.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from nts.kernel.instance import Instance
from nts.kernel.schemas import Constraint
from nts.kernel.solver import TimetableSolver
from nts.kernel.validator import validate_constraint, verify_timetable

from .events import (
    GENESIS,
    Change,
    ClassCancelled,
    ConstraintAdded,
    Created,
    Event,
    TimetableSet,
)
from .world import World


class StaleDraft(RuntimeError):
    """The store changed after the draft was taken; take a new draft and redo the work on it."""


class InvalidChange(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class CorruptLog(RuntimeError):
    pass


class Draft:
    def __init__(self, base: World, actor: str, episode: str | None) -> None:
        self.base = base
        self.actor = actor
        self.episode = episode
        self.changes: list[Change] = []
        self.world = base

    def add(self, change: Change) -> World:
        """Apply ``change`` to the draft; ValueError (and no change) if it cannot apply."""
        self.world = self.world.apply(change, f"{self.base.version}+{len(self.changes) + 1}")
        self.changes.append(change)
        return self.world


def touched(world: World, c: Constraint) -> set[int | None]:
    """The timetables a constraint applies to: a permanent one the semester timetable and every week that
    has its own; one with weeks, each of those weeks (most of them run the semester timetable)."""
    if c.active_in(None):
        return {None} | {w for w in world.timetables if w is not None}
    return {w for w in range(1, world.instance.calendar.weeks + 1) if c.active_in(w)}


def check(after: World, changes: Iterable[Change]) -> list[str]:
    """What the kernel finds wrong with ``changes``: added constraints that name unknown things or
    impossible times, and any timetable they touch or write that breaks a physical rule or a hard
    constraint in force (so a new constraint cannot slip in under a timetable that ignores it)."""
    inst = after.instance
    problems: list[str] = []
    weeks: set[int | None] = set()
    for ch in changes:
        if isinstance(ch, ConstraintAdded):
            problems += [f"{ch.constraint.id}: {p}" for p in validate_constraint(ch.constraint, inst)]
            weeks |= touched(after, ch.constraint)
        elif isinstance(ch, TimetableSet | ClassCancelled):
            weeks.add(ch.week)
    for w in sorted(weeks, key=lambda w: 0 if w is None else w):
        tt = after.timetable(w)
        if tt is None:
            continue
        found = verify_timetable(inst, tt.assignment, after.active(w), w)
        skipped = {f"{s} not scheduled" for s in tt.cancelled}
        label = "semester timetable" if w is None else f"week {w}"
        problems += [f"{label}: {p}" for p in found if p not in skipped]
    return problems


class Store:
    def __init__(self, events: list[Event], path: Path | None = None) -> None:
        self._events = events
        self.path = path
        self.world = World.replay((e.change, e.hash) for e in events)

    @classmethod
    def create(cls, instance: Instance, changes: Iterable[Change] = (), *, actor: str = "system",
               path: Path | str | None = None) -> Store:
        path = Path(path) if path else None
        if path is not None:
            if path.exists() and path.stat().st_size:
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        first = _event(0, GENESIS, Created(instance=instance), actor, None)
        store = cls([first], path)
        store._write([first])
        store._append(list(changes), actor, None)
        return store

    @classmethod
    def open(cls, path: Path | str) -> Store:
        path = Path(path)
        events = [Event.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        parent = GENESIS
        for i, e in enumerate(events):
            if e.seq != i or e.parent != parent or e.hash != e.digest():
                raise CorruptLog(f"{path}: event {i} does not follow the one before it")
            parent = e.hash
        return cls(events, path)

    @property
    def version(self) -> str:
        return self.world.version

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def at(self, version: str) -> World:
        """The world as it was at ``version`` (any earlier hash)."""
        for i, e in enumerate(self._events):
            if e.hash == version:
                return World.replay((x.change, x.hash) for x in self._events[: i + 1])
        raise KeyError(version)

    def fork(self, version: str | None = None) -> Store:
        """An in-memory copy of the log up to ``version`` (default: now), for replays and what-if runs."""
        upto = len(self._events) if version is None else next(
            (i + 1 for i, e in enumerate(self._events) if e.hash == version), None)
        if upto is None:
            raise KeyError(version)
        return Store(list(self._events[:upto]))

    def draft(self, actor: str, episode: str | None = None) -> Draft:
        return Draft(self.world, actor, episode)

    def commit(self, draft: Draft) -> World:
        if draft.base.version != self.version:
            raise StaleDraft(f"draft taken at {draft.base.version}, store is at {self.version}")
        if problems := check(draft.world, draft.changes):
            raise InvalidChange(problems)
        self._append(draft.changes, draft.actor, draft.episode)
        return self.world

    def _append(self, changes: list[Change], actor: str, episode: str | None) -> None:
        new: list[Event] = []
        world = self.world
        for ch in changes:
            e = _event(len(self._events) + len(new), (new[-1] if new else self._events[-1]).hash, ch, actor, episode)
            world = world.apply(ch, e.hash)  # raises before anything is written
            new.append(e)
        self._write(new)
        self._events += new
        self.world = world

    def _write(self, events: list[Event]) -> None:
        if self.path is not None and events:
            with self.path.open("a", encoding="utf-8") as f:
                f.writelines(e.model_dump_json() + "\n" for e in events)


def _event(seq: int, parent: str, change: Change, actor: str, episode: str | None) -> Event:
    e = Event(seq=seq, at=datetime.now(), actor=actor, episode=episode, parent=parent, change=change)
    e.hash = e.digest()
    return e


def bootstrap(instance: Instance, constraints: Iterable[Constraint] = (), *, path: Path | str | None = None,
              time_limit: float = 30.0) -> Store:
    """A new store for ``instance``: its constraints and a solved semester timetable."""
    constraints = list(constraints)
    result = TimetableSolver(instance, constraints, time_limit=time_limit).solve()
    if not result.ok:
        raise RuntimeError(f"no semester timetable: {result.status}")
    changes: list[Change] = [ConstraintAdded(constraint=c) for c in constraints]
    changes.append(TimetableSet(assignment=result.assignment))
    return Store.create(instance, changes, path=path)
