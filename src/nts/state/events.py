"""Everything that can happen to the department's state, as entries of an append-only log.

The state at any point is the changes up to it applied in order (``World.apply``). Each event carries
the hash of the one before it, so a version is one hash, and editing the log in the middle changes
every later hash (``Store.open`` checks the chain).
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from nts.kernel.instance import Instance
from nts.kernel.schemas import ConcessionEntry, Constraint, Placement

GENESIS = "0" * 16


class Created(BaseModel):
    kind: Literal["created"] = "created"
    instance: Instance


class ConstraintAdded(BaseModel):
    kind: Literal["constraint_added"] = "constraint_added"
    constraint: Constraint


class ConstraintRemoved(BaseModel):
    kind: Literal["constraint_removed"] = "constraint_removed"
    id: str


class TimetableSet(BaseModel):
    kind: Literal["timetable_set"] = "timetable_set"
    week: int | None = None  # None: the semester timetable; n: the timetable of week n
    assignment: dict[str, Placement]
    cancelled: list[str] = Field(default_factory=list)  # sessions not held that week


class Conceded(BaseModel):
    """Someone gave way: they agreed to waive a rule of theirs for another's request (the fairness ledger)."""

    kind: Literal["conceded"] = "conceded"
    entry: ConcessionEntry


Change = Annotated[Created | ConstraintAdded | ConstraintRemoved | TimetableSet | Conceded, Field(discriminator="kind")]


class Event(BaseModel):
    seq: int
    at: datetime
    actor: str  # who caused it: "system", a faculty id, "coordinator"
    episode: str | None = None  # the agent episode that committed it, if any
    parent: str  # hash of the previous event (GENESIS for the first)
    change: Change
    hash: str = ""

    def digest(self) -> str:
        body = self.model_dump_json(exclude={"hash"})
        return hashlib.sha256(body.encode()).hexdigest()[:16]
