"""Priority model (proposal Section 8.4): tiers first, then scores.

Within a tier, each constraint k gets a published score

    pi_k = w1 auth + w2 impact + w3 just + w4 lead + w5 credit - w6 disrupt

and a correction option M (a set of constraints to relax) costs

    cost(M) = sum_{k in M} pi_k + l1 disruption(M) + l2 dGini(M).

A higher pi_k means a greater cost of relaxing constraint k, and dGini is
computed over weighted concession burden (``ledger.burden``). The same
cost, with hidden flexibility known, is the oracle's objective.

The negotiator asks the owner of the cheapest option first. ``flat=True``
is ablation A2: every constraint scores the same and the ledger is ignored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from core.instance import Instance
from core.schemas import Constraint, Justification, Placement, Role
from core.semantics import PLACEMENT_TYPES, sessions_in_scope, violates

from .ledger import ConcessionLedger, burden

ROLE_AUTHORITY = {
    Role.DEAN: 1.0, Role.COORDINATOR: 0.9, Role.HOD: 0.8, Role.EXAM_CELL: 0.7,
    Role.LAB_INCHARGE: 0.6, Role.FACULTY: 0.5, Role.GUEST_FACULTY: 0.4, Role.STUDENT: 0.1,
}
JUSTIFICATION = {Justification.NONE: 0.0, Justification.STATED: 0.5, Justification.VERIFIED: 1.0}


@dataclass(frozen=True)
class Weights:
    """Published by the academic office. Defaults weigh the five positive
    terms equally and disruption the same; ``lead`` counts half."""

    auth: float = 1.0
    impact: float = 1.0
    just: float = 1.0
    lead: float = 0.5
    credit: float = 1.0
    disrupt: float = 1.0
    moved: float = 0.1  # l1: per session an option moves
    fairness: float = 2.0  # l2: per unit of Gini increase


class PriorityModel:
    def __init__(
        self,
        instance: Instance,
        *,
        weights: Weights = Weights(),
        ledger: ConcessionLedger | None = None,
        semester: str = "current",
        baseline: Mapping[str, Placement] | None = None,
        notice_days: Mapping[str, float] | None = None,
        flat: bool = False,
    ) -> None:
        self.instance = instance
        self.w = weights
        self.ledger = ledger or ConcessionLedger()
        self.semester = semester
        self.baseline = dict(baseline or {})
        self.notice_days = dict(notice_days or {})
        self.flat = flat
        self._students = sum(g.size for g in instance.groups) or 1
        self._group_size = {g.id: g.size for g in instance.groups}

    # -- terms, each in [0, 1] -------------------------------------------------

    def role(self, owner: str | None) -> Role:
        if owner is None:
            return Role.DEAN  # institutional rules
        fac = self.instance.faculty_by_id.get(owner)
        if fac is not None:
            return fac.role
        return Role.LAB_INCHARGE if owner.startswith("S-LAB") else Role.FACULTY

    def impact(self, c: Constraint) -> float:
        groups = {g for s in sessions_in_scope(self.instance, c.scope) for g in s.groups}
        return min(1.0, sum(self._group_size.get(g, 0) for g in groups) / self._students)

    def lead(self, c: Constraint) -> float:
        return min(1.0, self.notice_days.get(c.id, 7.0) / 14.0)

    def credit(self, c: Constraint) -> float:
        """Fairness credit 1 - 0.5^c: 0.5 after one unit of weighted burden, 0.75
        after two, 0.875 after three, so each further concession still counts.
        (The earlier ``min(1, c)`` was flat after about one concession; in the
        fairness stress test this lowers the Gini from 0.18 to 0.13.)"""
        return 1.0 - 0.5 ** self.ledger.credit(c.owner, self.semester) if c.owner else 0.0

    def disrupt(self, c: Constraint) -> float:
        """Share of the constraint's sessions it forces off their current place."""
        sessions = sessions_in_scope(self.instance, c.scope)
        if not self.baseline or not sessions or c.type not in PLACEMENT_TYPES:
            return 0.0
        days = self.instance.calendar.days
        moved = 0
        for s in sessions:
            p = self.baseline.get(s.id)
            if p and violates(self.instance, c, s, days.index(p.day), p.slot, self.instance.room_by_id[p.room]):
                moved += 1
        return moved / len(sessions)

    # -- scores ----------------------------------------------------------------

    def score(self, c: Constraint) -> float:
        if self.flat:
            return 1.0
        w = self.w
        return (w.auth * ROLE_AUTHORITY[self.role(c.owner)] + w.impact * self.impact(c)
                + w.just * JUSTIFICATION[c.justification] + w.lead * self.lead(c)
                + w.credit * self.credit(c) - w.disrupt * self.disrupt(c))

    def int_scores(self, constraints: Iterable[Constraint]) -> dict[str, int]:
        """Positive integer scores for CP-SAT (lexicographic solve, MCS costs)."""
        return {c.id: 1 + max(0, round(100 * self.score(c))) for c in constraints}

    def option_cost(self, drop: Iterable[Constraint], moved: int, stakeholders: Iterable[str]) -> float:
        """cost(M): a higher score means a constraint is dearer to relax; the
        fairness term is the rise in burden dispersion if M's owners concede."""
        drop = list(drop)
        base = sum(self.score(c) for c in drop) + self.w.moved * moved
        if self.flat:
            return base
        return base + self.w.fairness * self.ledger.delta_gini(stakeholders, burden(drop), self.semester)
