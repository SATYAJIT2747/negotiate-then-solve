"""The department's state at one version: the instance, the constraints in force and the timetables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from nts.kernel.instance import Instance
from nts.kernel.schemas import Constraint, Placement

from .events import Change, ConstraintAdded, ConstraintRemoved, TimetableSet


@dataclass(frozen=True)
class Timetable:
    assignment: Mapping[str, Placement]
    cancelled: frozenset[str] = frozenset()


@dataclass(frozen=True)
class World:
    """Immutable: ``apply`` returns a new World, so a draft never touches the version it started from."""

    instance: Instance
    version: str
    constraints: Mapping[str, Constraint] = field(default_factory=dict)  # by id, in the order added
    timetables: Mapping[int | None, Timetable] = field(default_factory=dict)  # None: the semester timetable

    @classmethod
    def replay(cls, changes: Iterable[tuple[Change, str]]) -> World:
        """The world after ``(change, version)`` pairs; the first change must be ``Created``."""
        world: World | None = None
        for change, version in changes:
            if world is None:
                if change.kind != "created":
                    raise ValueError("a log starts with a 'created' event")
                world = cls(instance=change.instance, version=version)
            else:
                world = world.apply(change, version)
        if world is None:
            raise ValueError("empty log")
        return world

    def apply(self, change: Change, version: str) -> World:
        constraints, timetables = dict(self.constraints), dict(self.timetables)
        if change.kind == "created":
            raise ValueError("the department already exists")
        if isinstance(change, ConstraintAdded):
            if change.constraint.id in constraints:
                raise ValueError(f"constraint {change.constraint.id} already exists")
            constraints[change.constraint.id] = change.constraint
        elif isinstance(change, ConstraintRemoved):
            if constraints.pop(change.id, None) is None:
                raise ValueError(f"no constraint {change.id}")
        elif isinstance(change, TimetableSet):
            timetables[change.week] = Timetable(MappingProxyType(dict(change.assignment)), frozenset(change.cancelled))
        return World(self.instance, version, MappingProxyType(constraints), MappingProxyType(timetables))

    def timetable(self, week: int | None = None) -> Timetable | None:
        """The timetable that runs in ``week``: its own one if it was repaired, else the semester one."""
        if week is not None and week in self.timetables:
            return self.timetables[week]
        return self.timetables.get(None)

    def active(self, week: int | None = None) -> list[Constraint]:
        return [c for c in self.constraints.values() if c.active_in(week)]
