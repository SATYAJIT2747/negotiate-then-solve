"""Concession ledger and fairness measures (proposal Sections 9.6-9.7 and 13).

Every accepted concession adds weighted credit to the stakeholder who
conceded: ``importance(k) x magnitude(k)``, so giving up a verified
unavailability weighs more than giving up a preference, and a partial
concession (a counter-offer) weighs less than dropping the constraint.
Credit lowers the chance of being asked to concede again, and it decays each
semester so history matters without dominating.

The same weighted credits are each stakeholder's *concession burden*; how
unevenly burden falls is summarised by the Gini coefficient, the maximum
individual burden and the coefficient of variation (``dispersion``).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping

from .schemas import ConcessionEntry, Constraint, Justification, Tier

# How much a concession on a constraint of each tier costs its owner. Tiers
# 0-2 are never conceded by an owner (a higher authority decides), so they
# weigh the most if they ever are.
TIER_IMPORTANCE = {
    Tier.PHYSICAL: 1.0, Tier.POLICY: 1.0, Tier.COMMITMENT: 1.0,
    Tier.VERIFIED_UNAVAILABILITY: 1.0, Tier.OPERATIONAL: 0.75, Tier.PREFERENCE: 0.5,
}
JUSTIFICATION_FACTOR = {Justification.NONE: 0.75, Justification.STATED: 0.875, Justification.VERIFIED: 1.0}

FULL = 1.0  # the constraint is dropped (the owner accepts an offered alternative)
PARTIAL = 0.5  # the owner narrows it themselves (a counter-offer)


def importance(c: Constraint) -> float:
    """Weight of conceding ``c``, in (0, 1]: its tier, scaled by how well the
    owner justified it (a verified Tier 3 unavailability is 1.0, an
    unjustified Tier 5 preference 0.375)."""
    return TIER_IMPORTANCE[c.tier] * JUSTIFICATION_FACTOR[c.justification]


def burden(constraints: Iterable[Constraint], magnitude: float = FULL) -> dict[str, float]:
    """Weighted burden per owner of conceding ``constraints``."""
    out: dict[str, float] = defaultdict(float)
    for c in constraints:
        if c.owner:
            out[c.owner] += importance(c) * magnitude
    return dict(out)


def gini(values: Iterable[float]) -> float:
    """Gini coefficient of non-negative values: 0 is perfectly even, values
    near 1 mean one stakeholder carries everything. All zeros give 0."""
    xs = sorted(max(0.0, float(v)) for v in values)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * weighted) / (n * total) - (n + 1) / n


def dispersion(values: Iterable[float]) -> dict[str, float]:
    """Burden dispersion D (proposal Section 9.7): the Gini coefficient as the
    main summary, plus the maximum individual burden and the coefficient of
    variation, so a conclusion does not rest on one index."""
    xs = [max(0.0, float(v)) for v in values]
    total = sum(xs)
    mu = total / len(xs) if xs else 0.0
    cv = statistics.pstdev(xs) / mu if mu else 0.0
    return {"gini": gini(xs), "max": max(xs, default=0.0), "cv": cv, "total": total}


class ConcessionLedger:
    def __init__(self, entries: Iterable[ConcessionEntry] = (), *, decay: float = 0.5) -> None:
        self.entries: list[ConcessionEntry] = list(entries)
        self.decay = decay

    def record(self, entry: ConcessionEntry) -> None:
        self.entries.append(entry)

    def semesters(self) -> list[str]:
        """Semester labels sort chronologically ("2025-2" < "2026-1")."""
        return sorted({e.semester for e in self.entries})

    def credit(self, stakeholder: str, semester: str) -> float:
        """Decayed credit as of ``semester``: this semester counts fully, the
        previous one ``decay`` times, and so on. Later semesters are ignored."""
        order = sorted({e.semester for e in self.entries} | {semester})
        now = order.index(semester)
        total = 0.0
        for e in self.entries:
            if e.stakeholder != stakeholder:
                continue
            age = now - order.index(e.semester)
            if age >= 0:
                total += e.credit * self.decay**age
        return total

    def credits(self, stakeholders: Iterable[str], semester: str) -> dict[str, float]:
        return {s: self.credit(s, semester) for s in stakeholders}

    def counts(self, semester: str | None = None) -> dict[str, float]:
        """Undecayed weighted burden per stakeholder, optionally in one semester."""
        out: dict[str, float] = defaultdict(float)
        for e in self.entries:
            if semester is None or e.semester == semester:
                out[e.stakeholder] += e.credit
        return dict(out)

    def times(self, semester: str | None = None) -> dict[str, int]:
        """How many constraints each stakeholder has conceded (for explanations,
        which state a count, not a weighted burden)."""
        out: dict[str, int] = defaultdict(int)
        for e in self.entries:
            if semester is None or e.semester == semester:
                out[e.stakeholder] += 1
        return dict(out)

    def gini(self, stakeholders: Iterable[str], semester: str | None = None) -> float:
        counts = self.counts(semester)
        return gini(counts.get(s, 0.0) for s in stakeholders)

    def dispersion(self, stakeholders: Iterable[str], semester: str | None = None) -> dict[str, float]:
        counts = self.counts(semester)
        return dispersion(counts.get(s, 0.0) for s in stakeholders)

    def delta_gini(self, stakeholders: Iterable[str], conceders: Mapping[str, float],
                   semester: str | None = None) -> float:
        """How much the Gini would change if ``conceders`` took on the given
        weighted burden now (Delta D in the option cost)."""
        people = list(stakeholders)
        counts = self.counts(semester)
        before = gini(counts.get(s, 0.0) for s in people)
        after = gini(counts.get(s, 0.0) + conceders.get(s, 0.0) for s in people)
        return after - before
