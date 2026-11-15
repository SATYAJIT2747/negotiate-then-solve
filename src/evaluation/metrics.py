"""Metrics shared by the evaluation scripts (proposal Table 11)."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from core.instance import Instance
from core.schemas import Constraint, ConstraintType
from core.semantics import room_satisfies


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, ok = np.asarray(confidences, float), np.asarray(correct, float)
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - ok[mask].mean())
    return float(total)


def atoms(c: Constraint, instance: Instance) -> Counter[tuple]:
    """A constraint as a bag of facts, with "all days/slots" spelled out so
    ``days=None`` and ``days=[Mon..Fri]`` compare equal."""
    cal = instance.calendar
    scope = next(((k, v) for k, v in c.scope.model_dump().items() if v is not None), ("global", None))
    out: Counter[tuple] = Counter({
        ("type", c.type.value): 1,
        ("hard", c.hard): 1,
        ("tier", int(c.tier)): 1,
        ("scope", *scope): 1,
    })
    out.update(("day", d) for d in (c.when.days or cal.days))
    out.update(("slot", s) for s in (c.when.slots or range(cal.slots_per_day)))
    out.update(("week", w) for w in (c.when.weeks or []))
    if c.room is not None:
        out.update(("equipment", e) for e in c.room.equipment)
        out.update(("room", r) for r in (c.room.rooms or []))
    return out


def meaning(c: Constraint, instance: Instance) -> tuple:
    """What a constraint rules out, independent of phrasing: "prefer Mon-Thu"
    and "avoid Fri" on a five-day week mean the same thing."""
    cal = instance.calendar
    cells = {(d, s) for d in cal.days for s in range(cal.slots_per_day)}
    inside = {(d, s) for d, s in cells
              if (c.when.days is None or d in c.when.days) and (c.when.slots is None or s in c.when.slots)}
    if c.type in (ConstraintType.UNAVAILABLE, ConstraintType.AVOID):
        excluded = frozenset(inside)
    elif c.type == ConstraintType.PREFER:
        excluded = frozenset(cells - inside)
    elif c.type == ConstraintType.REQUIRE_ROOM:
        assert c.room is not None
        excluded = frozenset(r.id for r in instance.rooms if not room_satisfies(c.room, r))
    elif c.type == ConstraintType.LOAD:
        excluded = frozenset({("load", c.measure, c.limit)})
    elif c.type == ConstraintType.RELATE:
        excluded = frozenset({("relate", c.relation, tuple(sorted(c.sessions or ())),
                               tuple(sorted(c.others)) if c.others is not None else None, c.amount, c.any_other,
                               c.negate)})
    elif c.type == ConstraintType.ANY_OF:  # one of them, in any order
        excluded = frozenset({("any_of", frozenset(meaning(m, instance) for m in c.members or ()))})
    elif c.type == ConstraintType.RANKED:  # the order is the meaning
        excluded = frozenset({("ranked", tuple(meaning(m, instance) for m in c.members or ()))})
    else:
        excluded = frozenset({("limit", c.limit)})
    scope = next(((k, v) for k, v in c.scope.model_dump().items() if v is not None), ("global", None))
    return (c.hard, int(c.tier), scope, excluded, tuple(c.when.weeks or ()))


def semantic_match(pred: Iterable[Constraint], gold: Iterable[Constraint], instance: Instance) -> bool:
    return Counter(meaning(c, instance) for c in pred) == Counter(meaning(c, instance) for c in gold)


def canonical(c: Constraint, instance: Instance) -> frozenset:
    return frozenset(atoms(c, instance).items())


def exact_match(pred: Iterable[Constraint], gold: Iterable[Constraint], instance: Instance) -> bool:
    return Counter(canonical(c, instance) for c in pred) == Counter(canonical(c, instance) for c in gold)


def atom_counts(pred: Iterable[Constraint], gold: Iterable[Constraint], instance: Instance) -> tuple[int, int, int]:
    """(true positives, predicted, gold) atom counts for micro P/R/F1."""
    p, g = Counter(), Counter()
    for c in pred:
        p += atoms(c, instance)
    for c in gold:
        g += atoms(c, instance)
    return sum((p & g).values()), sum(p.values()), sum(g.values())


def prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 1.0
    recall = tp / n_gold if n_gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0
