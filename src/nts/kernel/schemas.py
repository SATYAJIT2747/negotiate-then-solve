"""Core data objects shared by every layer (proposal Section 7.4)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class Tier(IntEnum):
    """Priority tiers (proposal Table 5). Lower tiers always win."""

    PHYSICAL = 0  # nobody may relax
    POLICY = 1  # dean / academic council
    COMMITMENT = 2  # academic office
    VERIFIED_UNAVAILABILITY = 3  # owner, with an alternative
    OPERATIONAL = 4  # owner or HoD
    PREFERENCE = 5  # owner, freely


class ConstraintType(str, Enum):
    UNAVAILABLE = "unavailable"
    PREFER = "prefer"
    AVOID = "avoid"
    REQUIRE_ROOM = "require_room"
    MAX_CONSECUTIVE = "max_consecutive"


# Types that can be soft; unavailability and hour limits are always hard.
SOFT_TYPES = frozenset({ConstraintType.PREFER, ConstraintType.AVOID, ConstraintType.REQUIRE_ROOM})


class Justification(str, Enum):
    NONE = "none"
    STATED = "stated"
    VERIFIED = "verified"


class Scope(BaseModel):
    """Which sessions a constraint applies to. All fields empty means everyone."""

    faculty: str | None = None
    group: str | None = None
    session: str | None = None
    course: str | None = None
    room: str | None = None

    @model_validator(mode="after")
    def _at_most_one(self) -> Scope:
        set_fields = [k for k, v in self.model_dump().items() if v is not None]
        if len(set_fields) > 1:
            raise ValueError(f"scope must name at most one entity, got {set_fields}")
        return self

    @property
    def is_global(self) -> bool:
        return all(v is None for v in self.model_dump().values())


class When(BaseModel):
    """Times a constraint talks about. ``None`` means "all" on that axis."""

    days: list[str] | None = None
    slots: list[int] | None = None
    weeks: list[int] | None = None

    @model_validator(mode="after")
    def _known_days(self) -> When:
        for d in self.days or []:
            if d not in DAYS:
                raise ValueError(f"unknown day {d!r}")
        return self


class RoomRequirement(BaseModel):
    rooms: list[str] | None = None
    equipment: list[str] = Field(default_factory=list)
    room_type: str | None = None


class Source(BaseModel):
    request: str | None = None
    rule: str | None = None


class Validity(BaseModel):
    from_week: int | None = None
    to_week: int | None = None

    @property
    def is_permanent(self) -> bool:
        return self.from_week is None and self.to_week is None

    def covers(self, week: int) -> bool:
        lo = self.from_week if self.from_week is not None else 1
        hi = self.to_week if self.to_week is not None else 10**6
        return lo <= week <= hi


class Constraint(BaseModel):
    """A typed constraint record, as produced by the parsing layer (L2)."""

    id: str
    type: ConstraintType
    hard: bool
    tier: Tier
    owner: str | None = None
    scope: Scope = Field(default_factory=Scope)
    when: When = Field(default_factory=When)
    room: RoomRequirement | None = None
    limit: int | None = None
    justification: Justification = Justification.NONE
    source: Source = Field(default_factory=Source)
    valid: Validity = Field(default_factory=Validity)
    weight: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Constraint:
        if self.tier == Tier.PHYSICAL and not self.hard:
            raise ValueError("Tier 0 constraints must be hard")
        if self.tier == Tier.PREFERENCE and self.hard:
            raise ValueError("Tier 5 preferences must be soft")
        if not self.hard and self.type not in SOFT_TYPES:
            raise ValueError(f"{self.type.value} constraints must be hard")
        if self.type == ConstraintType.REQUIRE_ROOM and self.room is None:
            raise ValueError("require_room needs a room requirement")
        if self.type == ConstraintType.MAX_CONSECUTIVE and not self.limit:
            raise ValueError("max_consecutive needs a positive limit")
        return self

    def active_in(self, week: int | None) -> bool:
        """Whether this constraint applies when solving ``week``.

        ``week=None`` is the semester-wide base timetable, which only carries
        permanent constraints; a specific week also carries the constraints
        whose validity window covers it.
        """
        if week is None:
            return self.valid.is_permanent and self.when.weeks is None
        if not self.valid.covers(week):
            return False
        return self.when.weeks is None or week in self.when.weeks


# ---------------------------------------------------------------------------
# Requests and their lifecycle (proposal Figure 2)
# ---------------------------------------------------------------------------


class Channel(str, Enum):
    PORTAL = "portal"
    EMAIL = "email"
    MESSAGING = "messaging"
    SYSTEM = "system"


class Role(str, Enum):
    COORDINATOR = "coordinator"
    DEAN = "dean"
    HOD = "hod"
    FACULTY = "faculty"
    GUEST_FACULTY = "guest_faculty"
    LAB_INCHARGE = "lab_incharge"
    EXAM_CELL = "exam_cell"
    STUDENT = "student"


class RequestStatus(str, Enum):
    RECEIVED = "received"
    CLASSIFIED = "classified"
    REFUSED = "refused"
    POLICY_CHECKED = "policy_checked"
    DENIED = "denied"
    CLARIFICATION = "clarification_requested"
    COMPILED = "compiled"
    SOLVED = "solved"
    NEGOTIATING = "negotiating"
    FAIRNESS_AUDITED = "fairness_audited"
    AWAITING_APPROVAL = "awaiting_approval"
    PUBLISHED = "published"
    ESCALATED = "escalated"
    ANSWERED = "answered"  # policy question answered from the handbook
    FORWARDED = "forwarded"  # clash report or out-of-scope message passed to a person
    WITHDRAWN = "withdrawn"  # taken back by the sender before it changed the published timetable


_S = RequestStatus
TRANSITIONS: dict[RequestStatus, frozenset[RequestStatus]] = {
    _S.RECEIVED: frozenset({_S.CLASSIFIED}),
    _S.CLASSIFIED: frozenset({_S.POLICY_CHECKED, _S.REFUSED, _S.ANSWERED, _S.FORWARDED}),
    _S.POLICY_CHECKED: frozenset({_S.COMPILED, _S.DENIED, _S.ESCALATED, _S.CLARIFICATION}),
    _S.COMPILED: frozenset({_S.SOLVED, _S.CLARIFICATION}),
    _S.CLARIFICATION: frozenset({_S.RECEIVED, _S.WITHDRAWN}),
    _S.SOLVED: frozenset({_S.FAIRNESS_AUDITED, _S.NEGOTIATING}),
    _S.NEGOTIATING: frozenset({_S.FAIRNESS_AUDITED, _S.ESCALATED}),
    _S.FAIRNESS_AUDITED: frozenset({_S.AWAITING_APPROVAL, _S.PUBLISHED}),  # nothing moved: nothing to approve
    _S.AWAITING_APPROVAL: frozenset({_S.PUBLISHED, _S.NEGOTIATING, _S.WITHDRAWN}),
    _S.REFUSED: frozenset(),
    _S.DENIED: frozenset(),
    _S.PUBLISHED: frozenset({_S.WITHDRAWN}),  # only one that moved nothing (Orchestrator.withdrawable)
    _S.ESCALATED: frozenset({_S.COMPILED, _S.DENIED, _S.WITHDRAWN}),  # a person with the authority grants or declines it
    _S.ANSWERED: frozenset(),
    _S.FORWARDED: frozenset({_S.WITHDRAWN}),
    _S.WITHDRAWN: frozenset(),
}


class Request(BaseModel):
    """One normalised message from any channel (after L1)."""

    id: str
    channel: Channel
    sender_id: str
    role: Role
    raw_text: str
    thread_id: str | None = None
    received_at: datetime
    status: RequestStatus = RequestStatus.RECEIVED

    def advance(self, new: RequestStatus) -> None:
        if new not in TRANSITIONS[self.status]:
            raise ValueError(f"illegal transition {self.status.value} -> {new.value}")
        self.status = new


# ---------------------------------------------------------------------------
# Timetables and the concession ledger
# ---------------------------------------------------------------------------


class Placement(BaseModel):
    """Where one session starts: day, first slot, room."""

    model_config = ConfigDict(frozen=True)

    day: str
    slot: int
    room: str


class TimetableVersion(BaseModel):
    version: int
    assignment: dict[str, Placement]
    parent: int | None = None
    approved_by: str | None = None
    week: int | None = None  # None: the semester timetable; n: the repaired timetable for week n
    cancelled: list[str] = []  # sessions not held in this week: every room they can use is closed

    def diff(self, other: TimetableVersion) -> dict[str, tuple[Placement | None, Placement | None]]:
        """Sessions whose placement differs, as ``id -> (self, other)``."""
        ids = self.assignment.keys() | other.assignment.keys()
        return {
            s: (self.assignment.get(s), other.assignment.get(s))
            for s in sorted(ids)
            if self.assignment.get(s) != other.assignment.get(s)
        }


class ConcessionEntry(BaseModel):
    stakeholder: str
    constraint_id: str
    conceded_to: str | None = None
    semester: str
    week: int | None = None
    credit: float = 1.0
