"""Paired analysis of the ablations A1-A3 against ours, from the runs already
made (no new negotiations).

The ablations ran once (seed 0) on a stratified 30 of the 60 scenarios, while
ours ran on all 60 x 3 seeds, so their pooled summaries are not like-for-like.
Here every ablation row is paired with ours on the same scenario and seed:
McNemar on correct outcome and resolution, Wilcoxon on rounds, objective,
faithfulness and first-offer acceptance, bootstrap CIs on paired differences,
and burden dispersion from the concessions recorded on those 30 scenarios only.
With ``--judge``, the LLM judge's clarity and acceptability (``evaluation.judge``)
are paired too.

    uv run python -m evaluation.ablations
    uv run python -m evaluation.ablations --judge runs/judge-ratings.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.ledger import ConcessionLedger, burden
from core.schemas import ConcessionEntry

from .benchmark import SEMESTER, load
from .negotiation import STAKEHOLDERS, compare, correct_outcome
from .stats import bootstrap_ci, mcnemar, mean, wilcoxon
from .study import judge_id

PAIRS = [  # (report, reference config, ablation config, what it isolates)
    ("runs/negotiation-main-ours-4b.json", "ours-llm", "A1", "flat weights, no tiers or ledger (H4)"),
    ("runs/negotiation-main-ours-4b.json", "ours-llm", "A2", "free-form explanations (H3)"),
    ("runs/negotiation-main-ours-4b.json", "ours-llm", "A3", "keyword/regex reply handler (tool calling)"),
    ("runs/negotiation-main-offline.json", "ours", "A1-offline", "flat weights, scripted replies, all 60"),
]


def _key(r: dict) -> tuple[str, int]:
    return r["scenario"], r.get("seed", 0)


def _spread(rows: list[dict], scenarios) -> dict:
    """Burden dispersion of the concessions these rows made. Rows from before
    ``ledger_added`` was logged are rebuilt from what they relaxed, each
    weighed by ``ledger.importance`` as a full concession (an escalated
    conflict changes nothing, so it adds no burden)."""
    entries = [ConcessionEntry(**e) for r in rows for e in r.get("ledger_added") or []]
    if not any("ledger_added" in r for r in rows):
        for r in rows:
            if r["status"] != "agreed":
                continue
            by_id = {c.id: c for c in scenarios[r["scenario"]].constraints}
            for owner, credit in burden(by_id[i] for i in r["relaxed"] if i in by_id).items():
                entries.append(ConcessionEntry(stakeholder=owner, constraint_id="-", semester=SEMESTER, credit=credit))
    return ConcessionLedger(entries).dispersion(STAKEHOLDERS, SEMESTER)


def _diff(a: list, b: list) -> dict:
    d = [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return {"n": len(d), "mean_diff": mean(d), "ci": bootstrap_ci(d)}


def paired(ref: list[dict], abl: list[dict], scenarios, judge: dict | None, ref_name: str, abl_name: str) -> dict:
    ka = {_key(r): r for r in ref}
    ids = [_key(r) for r in abl if _key(r) in ka]
    a, b = [ka[k] for k in ids], [{_key(r): r for r in abl}[k] for k in ids]
    out = compare(a, b, scenarios)
    agree = [i for i, k in enumerate(ids) if scenarios[k[0]].expected == "agree"]
    out["resolution_rate"] = {ref_name: mean([a[i]["status"] == "agreed" for i in agree]),
                              abl_name: mean([b[i]["status"] == "agreed" for i in agree])}
    out["correct_outcome"] = {ref_name: mean([correct_outcome(r, scenarios[r["scenario"]]) for r in a]),
                              abl_name: mean([correct_outcome(r, scenarios[r["scenario"]]) for r in b])}
    out["identical_status"] = sum(x["status"] == y["status"] for x, y in zip(a, b, strict=True))
    out["identical_replies"] = sum(x.get("replies") == y.get("replies") for x, y in zip(a, b, strict=True))
    for metric in ("faithfulness", "leaks", "invented_times", "omitted_options", "moved"):
        xa, xb = [r.get(metric) for r in a], [r.get(metric) for r in b]
        out[metric] = {ref_name: mean(xa), abl_name: mean(xb), "wilcoxon": wilcoxon(xa, xb), "diff": _diff(xa, xb)}
    fa = [(x.get("first_accepted"), y.get("first_accepted")) for x, y in zip(a, b, strict=True)
          if x.get("rounds") and y.get("rounds")]
    out["first_proposal_acceptance"] = {ref_name: mean([x for x, _ in fa]), abl_name: mean([y for _, y in fa]),
                                        "mcnemar": mcnemar([bool(x) for x, _ in fa], [bool(y) for _, y in fa])}
    out["burden"] = {ref_name: _spread(a, scenarios), abl_name: _spread(b, scenarios)}
    # does the ablation change who gives way? (both resolved, different people conceded)
    both = [(x, y) for x, y in zip(a, b, strict=True) if x["status"] == y["status"] == "agreed"]
    out["different_conceders"] = f"{sum(x['conceders'] != y['conceders'] for x, y in both)}/{len(both)}"
    if judge:
        for metric in ("clarity", "acceptability"):
            xa, xb = [], []
            for x, y in zip(a, b, strict=True):
                # the first message of each negotiation: present in both whenever both negotiated
                ja = judge.get(judge_id(ref_name, x["scenario"], 0, x.get("seed", 0)))
                jb = judge.get(judge_id(abl_name, y["scenario"], 0, y.get("seed", 0)))
                if ja and jb:
                    xa.append(ja[metric])
                    xb.append(jb[metric])
            out[f"judge_{metric}"] = {ref_name: mean(xa), abl_name: mean(xb), "wilcoxon": wilcoxon(xa, xb),
                                      "diff": _diff(xa, xb)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios.jsonl"))
    ap.add_argument("--judge", type=Path, default=None, help="LLM judge ratings (evaluation.judge)")
    ap.add_argument("--out", type=Path, default=Path("runs/negotiation-ablations-paired.json"))
    args = ap.parse_args()
    scenarios = {s.id: s for s in load(args.scenarios)}
    judge = json.loads(args.judge.read_text(encoding="utf-8")) if args.judge and args.judge.exists() else None
    report, cache = {}, {}
    for path, ref, abl, what in PAIRS:
        if path not in cache:
            cache[path] = json.loads(Path(path).read_text(encoding="utf-8"))["configs"]
        cfg = cache[path]
        if ref not in cfg or abl not in cfg:
            continue
        report[f"{ref} vs {abl}"] = {"isolates": what, **paired(cfg[ref]["rows"], cfg[abl]["rows"], scenarios,
                                                                 judge, ref, abl)}
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
