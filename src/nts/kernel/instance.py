"""A timetabling instance: calendar, rooms, people and the sessions to place."""

from __future__ import annotations

from enum import Enum
from functools import cached_property

from pydantic import BaseModel, Field

from .schemas import (
    Constraint,
    ConstraintType,
    Placement,
    Role,
    Scope,
    Source,
    Tier,
    When,
)


class RoomType(str, Enum):
    LECTURE = "lecture"
    LAB = "lab"


class SessionKind(str, Enum):
    LECTURE = "lecture"
    TUTORIAL = "tutorial"
    PRACTICAL = "practical"


class Room(BaseModel):
    id: str
    name: str
    capacity: int
    type: RoomType
    equipment: list[str] = Field(default_factory=list)


class Faculty(BaseModel):
    id: str
    name: str
    department: str
    role: Role = Role.FACULTY
    reports_to: str | None = None


class Group(BaseModel):
    """A student section that attends its sessions together."""

    id: str
    name: str
    size: int


class Session(BaseModel):
    """One weekly meeting of a course. Multi-slot sessions stay on one day."""

    id: str
    course: str
    kind: SessionKind
    faculty: str
    groups: list[str]
    duration: int = 1
    room_type: RoomType = RoomType.LECTURE
    equipment: list[str] = Field(default_factory=list)
    size: int | None = None  # overrides the sum of group sizes (e.g. ITC-2007 enrolments)


class ExtraClass(BaseModel):
    """A class held in one week only (an extra class asked for by its teacher): a copy of one of the
    teacher's weekly sessions, at a placement that was free in that week. Kept in the store, not in
    the instance, so solves of other weeks never see it."""

    session: Session
    week: int
    placement: Placement
    request: str | None = None


class Calendar(BaseModel):
    days: list[str] = Field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])
    slots_per_day: int = 8
    lunch_slot: int | None = 4
    weeks: int = 16


class Policy(BaseModel):
    """Institute rules that become Tier 1 constraints."""

    max_consecutive: int | None = 3
    lunch_break: bool = True


class Instance(BaseModel):
    name: str
    calendar: Calendar = Field(default_factory=Calendar)
    rooms: list[Room]
    faculty: list[Faculty]
    groups: list[Group]
    sessions: list[Session]
    policy: Policy = Field(default_factory=Policy)
    course_titles: dict[str, str] = Field(default_factory=dict)

    @cached_property
    def room_by_id(self) -> dict[str, Room]:
        return {r.id: r for r in self.rooms}

    @cached_property
    def faculty_by_id(self) -> dict[str, Faculty]:
        return {f.id: f for f in self.faculty}

    @cached_property
    def group_by_id(self) -> dict[str, Group]:
        return {g.id: g for g in self.groups}

    @cached_property
    def session_by_id(self) -> dict[str, Session]:
        return {s.id: s for s in self.sessions}

    def session_size(self, session: Session) -> int:
        if session.size is not None:
            return session.size
        return sum(self.group_by_id[g].size for g in session.groups)

    def course_title(self, course: str) -> str:
        return self.course_titles.get(course, course)

    def day_index(self, day: str) -> int:
        return self.calendar.days.index(day)


def policy_constraints(instance: Instance) -> list[Constraint]:
    """Institute policy as Tier 1 constraints, each citing its rule."""
    out: list[Constraint] = []
    cal, pol = instance.calendar, instance.policy
    if pol.lunch_break and cal.lunch_slot is not None:
        out.append(
            Constraint(
                id="P-LUNCH",
                type=ConstraintType.UNAVAILABLE,
                hard=True,
                tier=Tier.POLICY,
                scope=Scope(),
                when=When(slots=[cal.lunch_slot]),
                source=Source(rule="common lunch break"),
            )
        )
    if pol.max_consecutive:
        out.append(
            Constraint(
                id="P-MAXCONSEC",
                type=ConstraintType.MAX_CONSECUTIVE,
                hard=True,
                tier=Tier.POLICY,
                scope=Scope(),
                limit=pol.max_consecutive,
                source=Source(rule=f"at most {pol.max_consecutive} consecutive teaching hours"),
            )
        )
    return out
