"""Merge the Kaggle ablation rows (runs/kaggle-ablations-v2/<run>/results/<config>-s<seed>.rows.jsonl) and pair each
ablation with ours-llm on (scenario, seed). Writes runs/ablations-v2-paired.json and prints the Table 8 rows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evaluation.ablations import paired  # noqa: E402
from evaluation.benchmark import load  # noqa: E402

rows: dict[str, list] = {}
for f in sorted(Path("runs/kaggle-ablations-v2").glob("*/results/*.rows.jsonl")):
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.setdefault(r["config"], []).append(r)
sc = {s.id: s for s in load(Path("data/scenarios.jsonl"))}
print({c: len(v) for c, v in rows.items()})
ref = rows["ours-llm"]
out = {}
for c in ("A1", "A2", "A3"):
    if c not in rows:
        continue
    o = paired(ref, rows[c], sc, None, "ours", c)
    out[f"ours vs {c}"] = o
json.dump(out, open("runs/ablations-v2-paired.json", "w"), indent=1, default=str)


def line(name, o, side):
    agree = [r for r in (ref if side == "ours" else rows[name]) if sc[r["scenario"]].expected == "agree"]
    allr = ref if side == "ours" else rows[name]
    res = sum(r["status"] == "agreed" for r in agree)
    fp = o["first_proposal_acceptance"]
    return (f"{side:5s} resolved {res}/{len(agree)} correct {round(o['correct_outcome'][side] * len(allr))}/{len(allr)} "
            f"conceder {o['different_conceders'] if side != 'ours' else '--'} moved {o['moved'][side]:.2f} "
            f"omitted {round(o['omitted_options'][side] * len(allr))} leaks {round(o['leaks'][side] * len(allr))} "
            f"1st {fp[side]:.0%} gini {o['burden'][side]['gini']:.3f}")


for k, o in out.items():
    c = k.split()[-1]
    print("==", k)
    print(line(c, o, "ours"))
    print(line(c, o, c))
    print("  p: leaks", round(o["leaks"]["wilcoxon"]["p"], 4), "omitted", round(o["omitted_options"]["wilcoxon"]["p"], 4),
          "1st", o["first_proposal_acceptance"]["mcnemar"], "correct", o["correct_outcome_mcnemar"],
          "faith", {x: round(y, 3) for x, y in o["faithfulness"].items() if isinstance(y, float)})
