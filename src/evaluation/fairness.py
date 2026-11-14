"""Fairness stress test (H4): does the concession ledger spread burden over a
semester without costing resolutions?

The 60 benchmark scenarios are mostly independent one-off conflicts, so the
ledger rarely meets the same person twice and its effect is small there. This
test builds the situation the ledger exists for: one department, the same
four teachers, a semester of lab clashes in which two of them want the one
equipped lab on the same afternoon. F-301 and F-304 teach the smallest
section, so moving their practical always disturbs the fewest students: on
paper they are the cheaper ones to ask, every time.

Three configurations meet identical sequences (same conflicts, same hidden
flexibility, scripted replies, no API):

* ``ours``       tiers, weights, ledger credit and the fairness (dGini) term
* ``no-ledger``  the same weights with the credit and dGini terms off: isolates the ledger
* ``flat``       ablation A1: every constraint scores the same, no ledger
* ``ours-smooth`` experimental: ours with credit 1 - 0.5^credit instead of min(1, credit),
               so the credit term keeps growing after the first concession (the live
               priority model is unchanged)

Reported per configuration, over run seeds: final burden per teacher, Gini,
maximum burden and CV of weighted burden, resolution rate, rounds per
resolved conflict, students in the conceding teacher's section (the cost of
ignoring impact),
and the Gini after each conflict; paired Wilcoxon on Gini
and max burden, and the paired difference in resolutions.

    uv run python -m evaluation.fairness                    # 10 seeds x 24 conflicts
    uv run python -m evaluation.fairness --seeds 3 --conflicts 12
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from agents.explainer import Explainer
from agents.ledger import ConcessionLedger
from agents.negotiation import Negotiator
from agents.priority import PriorityModel, Weights
from agents.simulators import Profile, Window
from core.instance import policy_constraints
from core.schemas import ConstraintType, Justification, RoomRequirement, Scope, Tier, When
from core.solver import TimetableSolver

from .benchmark import (AFTERNOON, DAYS, FULL_DAY, MORNING, SEMESTER, Scenario, _c, _department, _lab, _names,
                        _practical, _title)
from .negotiation import STAKEHOLDERS, WORKERS, _row
from .stats import bootstrap_ci, mean, wilcoxon

CONFIGS = ["ours", "no-ledger", "flat", "ours-smooth"]
SECTION_SIZES = {"G-A": 35, "G-B": 55, "G-C": 60}  # F-301 and F-304 teach G-A


def conflict(dept_seed: int, rng: random.Random, sid: str, a: int, b: int, flex_p: float) -> Scenario:
    """Teachers ``a`` and ``b`` both want the one GPU-and-routers lab on the
    same afternoon. Each would, with chance ``flex_p``, take another window
    (at least one would, so every conflict is resolvable by agreement)."""
    labs = [_lab("L-1", 1, ["gpu", "routers"]), _lab("L-2", 2, []), _lab("L-3", 3, ["electronics"])]
    inst = _department(random.Random(dept_seed), sid, labs)  # the same department every time
    for g in inst.groups:
        g.size = SECTION_SIZES[g.id]
    pa, pb = _practical(inst, a), _practical(inst, b)
    fa, fb = f"F-30{a}", f"F-30{b}"
    day = rng.choice(DAYS)
    other = [d for d in DAYS if d != day]
    planted = []
    for tag, owner, session, need in (("A", fa, pa, "gpu"), ("B", fb, pb, "routers")):
        planted += [
            _c(f"C-{tag}-ROOM", owner, Tier.OPERATIONAL, ConstraintType.REQUIRE_ROOM, scope=Scope(session=session),
               room=RoomRequirement(equipment=[need]), request=f"R-{tag}", just=Justification.STATED),
            _c(f"C-{tag}-TIME", owner, Tier.OPERATIONAL, ConstraintType.PREFER, scope=Scope(session=session),
               when=When(days=[day], slots=AFTERNOON), request=f"R-{tag}", just=Justification.STATED),
        ]
    flexible = [rng.random() < flex_p, rng.random() < flex_p]
    if not any(flexible):
        flexible[rng.randrange(2)] = True
    profiles = [Profile(owner=o, needs=[n], reveal=True,
                        windows=[Window(days=rng.sample(other, 2), slots=rng.choice([AFTERNOON, MORNING]))]
                        if f else [])
                for o, n, f in ((fa, "gpu", flexible[0]), (fb, "routers", flexible[1]))]
    names, full = _names(inst), FULL_DAY[day]
    raw = {"R-A": f"My {_title(inst, pa)} practical needs the GPU machines on {full} afternoon. {names[fa]}",
           "R-B": f"The {_title(inst, pb)} lab needs routers; {full} afternoon please. {names[fb]}"}
    base = policy_constraints(inst)
    baseline = TimetableSolver(inst, base, time_limit=20, workers=WORKERS).solve()
    if not baseline.ok:
        raise RuntimeError(f"{sid}: base instance infeasible")
    return Scenario(id=sid, kind="contention", seed=dept_seed, instance=inst, base=base, planted=planted,
                    baseline=baseline.assignment, profiles=profiles, raw_requests=raw)


def semester(seed: int, n: int, flex_p: float = 0.85) -> list[Scenario]:
    """``n`` conflicts in one department; each pair of teachers is drawn uniformly."""
    rng = random.Random(seed)
    dept_seed = rng.randrange(10**9)
    out = []
    for i in range(n):
        a, b = rng.sample([1, 2, 3, 4], 2)
        out.append(conflict(dept_seed, rng, f"F{seed}-{i + 1:02d}", a, b, flex_p))
    return out


class SmoothCredit(PriorityModel):
    """Credit 1 - 0.5^c: 0.5 after one unit of weighted burden, 0.75 after two,
    so a third concession still costs more than a second (``min(1, c)`` is
    flat after the first)."""

    def credit(self, c) -> float:
        return 1.0 - 0.5 ** self.ledger.credit(c.owner, self.semester) if c.owner else 0.0


def negotiator(sc: Scenario, config: str, ledger: ConcessionLedger) -> Negotiator:
    flat = config == "flat"
    weights = Weights(credit=0.0, fairness=0.0) if config == "no-ledger" else Weights()
    model = SmoothCredit if config == "ours-smooth" else PriorityModel
    pm = model(sc.instance, weights=weights, ledger=ledger, semester=SEMESTER, baseline=sc.baseline, flat=flat)
    return Negotiator(sc.instance, priority=pm, explainer=Explainer(sc.instance), ledger=ledger, semester=SEMESTER,
                      use_ledger=config in ("ours", "ours-smooth"), time_limit=20, workers=WORKERS)


def disturbed(sc: Scenario, conceders: list[str]) -> int:
    """Students in the sections of the teachers who gave way: their practical
    moved off the time it was asked for (the impact term of the priority model)."""
    size = {g.id: g.size for g in sc.instance.groups}
    return sum(size[g] for s in sc.instance.sessions if s.faculty in conceders and s.id.endswith("-P")
               for g in s.groups)


def run(scenarios: list[Scenario], config: str) -> dict:
    from .benchmark import simulators

    ledger = ConcessionLedger()
    rows, trace = [], []
    for sc in scenarios:
        neg = negotiator(sc, config, ledger)
        out = neg.resolve(sc.constraints, simulators(sc), baseline=sc.baseline, week=sc.week,
                          private=sc.private, raw_requests=sc.raw_requests)
        if out.status != "escalated":
            ledger.entries.extend(out.concessions)
        assignment = out.result.assignment if out.result and out.result.ok else None
        row = _row(sc, out, assignment)
        row["asked_first"] = out.messages[0].to if out.messages else None
        row["students"] = disturbed(sc, row["conceders"]) if out.status == "agreed" else None
        rows.append({k: row[k] for k in ("status", "rounds", "conceders", "valid", "asked_first", "moved", "students")}
                    | {"scenario": sc.id})
        trace.append(ledger.dispersion(STAKEHOLDERS, SEMESTER)["gini"])
    burden = ledger.counts(SEMESTER)
    d = ledger.dispersion(STAKEHOLDERS, SEMESTER)
    return {
        "burden": {s: burden.get(s, 0.0) for s in STAKEHOLDERS},
        # conflicts in which each teacher gave way (the ledger counts constraints, two per lab request)
        "times": {s: sum(s in r["conceders"] for r in rows) for s in STAKEHOLDERS},
        "gini": d["gini"], "max": d["max"], "cv": d["cv"], "total": d["total"],
        "resolved": sum(r["status"] == "agreed" for r in rows),
        "valid": all(r["valid"] is not False for r in rows),
        "rounds_resolved": mean([r["rounds"] for r in rows if r["status"] == "agreed"]),
        "students_per_conflict": mean([r["students"] for r in rows if r["students"] is not None]),
        "moved_per_conflict": mean([r["moved"] for r in rows if r["status"] == "agreed"]),
        "gini_trace": trace,
        "rows": rows,
    }


def summarise(runs: list[dict], n: int) -> dict:
    return {
        "gini": mean([r["gini"] for r in runs]), "gini_ci": bootstrap_ci([r["gini"] for r in runs]),
        "max": mean([r["max"] for r in runs]), "max_ci": bootstrap_ci([r["max"] for r in runs]),
        "cv": mean([r["cv"] for r in runs]),
        "total": mean([r["total"] for r in runs]),
        "resolution_rate": mean([r["resolved"] / n for r in runs]),
        "rounds_resolved": mean([r["rounds_resolved"] for r in runs]),
        "students_per_conflict": mean([r["students_per_conflict"] for r in runs]),
        "students_ci": bootstrap_ci([r["students_per_conflict"] for r in runs]),
        "moved_per_conflict": mean([r["moved_per_conflict"] for r in runs]),
        "all_valid": all(r["valid"] for r in runs),
        "burden": {s: mean([r["burden"][s] for r in runs]) for s in STAKEHOLDERS},
        "times": {s: mean([r["times"][s] for r in runs]) for s in STAKEHOLDERS},
        "gini_trace": [mean(step) for step in zip(*(r["gini_trace"] for r in runs), strict=True)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="independent semesters")
    ap.add_argument("--conflicts", type=int, default=24, help="conflicts per semester")
    ap.add_argument("--flex", type=float, default=0.85, help="chance each teacher would take another window")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    per: dict[str, list[dict]] = {c: [] for c in CONFIGS}
    for seed in range(args.seeds):
        scenarios = semester(seed, args.conflicts, args.flex)
        for c in CONFIGS:
            r = run(scenarios, c)
            per[c].append(r)
            print(f"[seed {seed}] {c:9s} gini={r['gini']:.3f} max={r['max']:.2f} "
                  f"resolved={r['resolved']}/{args.conflicts} students={r['students_per_conflict']:.1f} times={r['times']}", flush=True)
    summary = {c: summarise(per[c], args.conflicts) for c in CONFIGS}
    paired = {}
    for ref, c in (("ours", "no-ledger"), ("ours", "flat"), ("ours-smooth", "ours"), ("ours-smooth", "no-ledger")):
        paired[f"{ref} vs {c}"] = {
            m: wilcoxon([r[m] for r in per[ref]], [r[m] for r in per[c]])
            for m in ("gini", "max", "students_per_conflict")
        } | {"resolved_diff": sum(a["resolved"] - b["resolved"] for a, b in zip(per[ref], per[c], strict=True))}
    report = {"seeds": args.seeds, "conflicts": args.conflicts, "flex": args.flex, "section_sizes": SECTION_SIZES,
              "summary": summary, "paired": paired, "runs": per, "wall_s": time.perf_counter() - t0}
    out = args.out or Path("runs") / f"fairness-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for c, s in summary.items():
        print(f"\n== {c}: gini {s['gini']:.3f} {s['gini_ci']}, max {s['max']:.2f}, cv {s['cv']:.2f}, "
              f"resolution {s['resolution_rate']:.1%}, rounds {s['rounds_resolved']:.2f}\n   times {s['times']}")
    print("\npaired:", json.dumps(paired, indent=1))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
