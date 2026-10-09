"""Scoring parsers that read intents (``language.intents``) on ``data/compiler-intents``.

    uv run python -m evaluation.intents --parser gold                 # sanity check: must score 100%
    uv run python -m evaluation.intents --parser rule                 # the rule parser, the baseline
    uv run python -m evaluation.intents --parser local --local-model qwen3.5-4b-nts-read-100.Q8_0
    uv run python -m evaluation.intents --parser gemini --limit 60

A prediction is scored by what it does, not how it is written: the gold and
the predicted intents are both compiled (``compile_intents``) against the
same request, timetable and current week, and the outcomes compared:

* **action**: compile, clarify (a question), withdraw, investigate, answer,
  out of scope, refuse;
* **outcome match**: the same action, and the same constraints (exact, as in
  ``evaluation.parsing``), swaps, extra classes and cancellations; when both
  ask a question, any question counts;
* **constraint atoms**: micro P/R/F1 over the constraints' facts, for partial
  credit;
* **literal match**: the same JSON as the gold, a stricter reading.

Results are broken down by kind (corpus request type or new intent) and by
source (``corpus``: converted from the request corpus; ``generated``: the new
intents). The model sees the row's user message exactly as in training; the
rule parser reads the request text. Model answers are cached
(``runs/llm_cache``), so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.instance import Instance
from core.schemas import Placement, Request
from evaluation.metrics import atom_counts, exact_match, prf
from language.corpus import ExpectedAction
from language.intent_compiler import INTENT_PROMPT
from language.intents import Compiled, IntentOutput, compile_intents
from language.llm import LLMError


def outcome(instance: Instance, c: Compiled) -> dict:
    """What a compiled reading does: its action, whether it asks, and what it resolved to."""
    r = c.result
    asks = c.question is not None or r.action == ExpectedAction.CLARIFY
    return {
        "action": "clarify" if asks and r.action != ExpectedAction.REFUSE else r.action.value,
        "constraints": r.constraints,
        "swaps": sorted((x.mine, x.theirs, tuple(x.weeks or ())) for x in c.swaps),
        "extras": sorted((_class(instance, x.template), tuple(x.weeks or ()), tuple(x.days or ()), tuple(x.slots or ()))
                         for x in c.extras),
        "cancels": sorted((x.session, tuple(x.weeks or ())) for x in c.cancels),
    }


def _class(instance: Instance, sid: str) -> tuple:
    """An extra class copies a class, not one meeting of it: any meeting is the same answer."""
    s = instance.session_by_id[sid]
    return s.course, s.kind.value, tuple(s.groups)


def same(instance: Instance, pred: dict, gold: dict) -> bool:
    if pred["action"] != gold["action"]:
        return False
    if gold["action"] == "clarify":
        return True
    return (exact_match(pred["constraints"], gold["constraints"], instance)
            and all(pred[k] == gold[k] for k in ("swaps", "extras", "cancels")))


def _context(row: dict) -> tuple[Request, dict[str, Placement], int | None]:
    request = Request.model_validate(row["request"])
    timetable = {sid: Placement.model_validate(p) for sid, p in row["timetable"].items()}
    return request, timetable, row.get("current_week")


class Reader:
    """Reads one row: the gold itself, the rule parser, or a model on the row's training prompt."""

    def __init__(self, kind: str, instance: Instance, client=None) -> None:
        self.kind, self.instance, self.client = kind, instance, client
        if kind == "rule":
            from language.rule_parser import RuleParser

            self.rules = RuleParser(instance)

    def read(self, row: dict) -> IntentOutput:
        if self.kind == "gold":
            return IntentOutput.model_validate_json(row["messages"][2]["content"])
        if self.kind == "rule":
            return self.rules.read_intents(_context(row)[0])
        return self.client.generate(INTENT_PROMPT, row["messages"][1]["content"], IntentOutput)


def score(rows: list[dict], instance: Instance, reader: Reader, workers: int = 1) -> dict:
    lock = threading.Lock()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    atoms = [0, 0, 0]
    confusion: Counter = Counter()
    out_rows = []

    def one(row: dict) -> None:
        request, timetable, week = _context(row)
        gold_out = IntentOutput.model_validate_json(row["messages"][2]["content"])
        gold = outcome(instance, compile_intents(instance, request, gold_out, timetable=timetable, current_week=week))
        error = None
        try:
            pred_out = reader.read(row)
            pred = outcome(instance, compile_intents(instance, request, pred_out, timetable=timetable,
                                                     current_week=week))
        except (LLMError, ValueError) as e:  # an invalid or cut-off answer is a wrong answer
            pred_out, pred, error = None, {"action": "invalid", "constraints": [], "swaps": [], "extras": [],
                                           "cancels": []}, f"{type(e).__name__}: {str(e)[:120]}"
        hit = same(instance, pred, gold)
        literal = pred_out is not None and json.loads(pred_out.model_dump_json(exclude_defaults=True)) == \
            json.loads(gold_out.model_dump_json(exclude_defaults=True))
        tp, n_pred, n_gold = atom_counts(pred["constraints"], gold["constraints"], instance)
        with lock:
            for c in (by_kind[row["kind"]], by_source[row["source"]]):
                c["n"] += 1
                c["action"] += pred["action"] == gold["action"]
                c["match"] += hit
                c["literal"] += literal
                c["invalid"] += error is not None
            atoms[0] += tp
            atoms[1] += n_pred
            atoms[2] += n_gold
            if pred["action"] != gold["action"]:
                confusion[f"{gold['action']}->{pred['action']}"] += 1
            out_rows.append({"id": row["id"], "kind": row["kind"], "source": row["source"], "match": hit,
                             "gold_action": gold["action"], "pred_action": pred["action"], "error": error,
                             "text": row["request"]["raw_text"][:200],
                             "pred": pred_out.model_dump(mode="json", exclude_defaults=True) if pred_out else None,
                             "gold": gold_out.model_dump(mode="json", exclude_defaults=True)})

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, rows))
    else:
        for r in rows:
            one(r)

    def rates(c: Counter) -> dict:
        n = c["n"]
        return {"n": n, "action_accuracy": round(c["action"] / n, 3), "outcome_match": round(c["match"] / n, 3),
                "literal_match": round(c["literal"] / n, 3), "invalid": c["invalid"]}

    total = sum(by_source.values(), Counter())
    p, r, f = prf(*atoms)
    return {
        "summary": {**rates(total), "constraint_atom_precision": round(p, 3), "constraint_atom_recall": round(r, 3),
                    "constraint_atom_f1": round(f, 3)},
        "by_source": {k: rates(v) for k, v in sorted(by_source.items())},
        "by_kind": {k: rates(v) for k, v in sorted(by_kind.items())},
        "confusion": dict(confusion.most_common()),
        "rows": sorted(out_rows, key=lambda x: x["id"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", choices=["gold", "rule", "local", "gemini"], default="rule")
    ap.add_argument("--data", type=Path, default=Path("data/compiler-intents"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--source", choices=["all", "corpus", "generated"], default="all")
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--local", metavar="URL", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None, help="model on a router-mode server (default NTS_LOCAL_MODEL)")
    ap.add_argument("--model", default=None, help="Gemini model id (default: the pinned agent model)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.data / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.source != "all":
        rows = [r for r in rows if r["source"] == args.source]
    rows = rows[: args.limit]
    client = None
    if args.parser == "local":
        from language.local import LocalClient

        client = LocalClient(base_url=args.local, server_model=args.local_model)
    elif args.parser == "gemini":
        from language.llm import GeminiClient

        client = GeminiClient(args.model)
    report = score(rows, instance, Reader(args.parser, instance, client), args.workers)
    name = args.parser if client is None else f"{args.parser}-{client.model.split(':')[-1].removesuffix('.gguf')}"
    report = {"parser": name, "split": args.split, "source": args.source, **report}

    s = report["summary"]
    print(f"{name} on {args.split} ({s['n']} requests): outcome match {s['outcome_match']:.1%}, action "
          f"{s['action_accuracy']:.1%}, literal {s['literal_match']:.1%}, constraint atom F1 "
          f"{s['constraint_atom_f1']:.3f}, invalid {s['invalid']}")
    print(f"  {'kind':22} {'n':>4} {'match':>7} {'action':>7} {'literal':>8}")
    for group in ("by_source", "by_kind"):
        for k, v in report[group].items():
            print(f"  {k:22} {v['n']:>4} {v['outcome_match']:>7.1%} {v['action_accuracy']:>7.1%} {v['literal_match']:>8.1%}")
        print()
    if report["confusion"]:
        print("  action confusion (gold->predicted):", report["confusion"])
    out = args.out or Path("runs") / f"intents-{name}-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
