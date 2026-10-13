"""Retrieval evaluation: BM25, dense and hybrid (proposal: RAG row of Table 11).

    uv run python -m evaluation.retrieval                  # no API calls
    uv run python -m evaluation.retrieval --no-documents   # handbook only

The rules are the handbook plus the policy documents in ``data/policies``
(OCR cached), as the web app loads them. Two query sets:

* ``corpus``: requests from the request corpus (templated and paraphrased)
  whose labels name the rules a correct policy check cites. Scored on the raw
  message and on the message plus the words the parse implies
  (``query_hints``, from the offline rule parser), which is what the policy
  agent searches with.
* ``paraphrase``: ``data/retrieval-queries.jsonl``, questions written in
  other words than the rules (and covering the policy documents), where
  lexical matching is expected to struggle.

Reported per retriever: recall@1/3/5 (share of gold rules in the top k) and
mean reciprocal rank of the first gold rule.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agents.policy import RETRIEVERS, Rule, load_handbook, make_retriever, query_hints
from core.instance import Instance
from language.corpus import load_jsonl
from language.rule_parser import RuleParser


def score(retriever, queries: list[tuple[str, list[str]]], ks=(1, 3, 5)) -> dict:
    hits = {k: 0 for k in ks}
    total = rr = 0.0
    started = time.perf_counter()
    misses = []
    for text, gold in queries:
        ranked = [r.id for r in retriever.search(text, max(ks))]
        for k in ks:
            hits[k] += sum(g in ranked[:k] for g in gold)
        total += len(gold)
        first = next((i for i, rid in enumerate(ranked, 1) if rid in gold), None)
        rr += 1 / first if first else 0.0
        if first is None or first > 3:
            misses.append({"query": text, "gold": gold, "top3": ranked[:3]})
    n = len(queries)
    return {"n": n, **{f"recall@{k}": round(hits[k] / total, 4) for k in ks},
            "mrr": round(rr / n, 4), "ms_per_query": round(1000 * (time.perf_counter() - started) / n, 2),
            "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("data/requests.jsonl"), Path("data/requests-para.jsonl")])
    ap.add_argument("--queries", type=Path, default=Path("data/retrieval-queries.jsonl"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--handbook", type=Path, default=Path("data/handbook.md"))
    ap.add_argument("--policies", type=Path, default=Path("data/policies"))
    ap.add_argument("--no-documents", action="store_true", help="handbook rules only")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.no_documents:
        rules: list[Rule] = load_handbook(args.handbook)
    else:
        from agents.documents import load_corpus, ocr_backend

        rules, _ = load_corpus(args.handbook, args.policies, ocr_backend())
    known = {r.id for r in rules}
    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    parser = RuleParser(instance, robust=False)  # as in the recorded runs

    corpus_raw, corpus_hinted = [], []
    for path in args.corpus:
        for ex in load_jsonl(path, instance):
            gold = [r for r in ex.rules if r in known]
            if not gold or (args.split != "all" and ex.split != args.split):
                continue
            text = ex.request.raw_text
            corpus_raw.append((text, gold))
            corpus_hinted.append((f"{text} {query_hints(instance, parser.parse(ex.request))}", gold))
    para = []
    for line in args.queries.read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            if gold := [r for r in q["rules"] if r in known]:
                para.append((q["query"], gold))

    report = {"rules": len(rules), "documents": not args.no_documents, "split": args.split, "sets": {}}
    for kind in RETRIEVERS:
        retriever = make_retriever(rules, kind)
        for name, queries in (("corpus", corpus_raw), ("corpus+hints", corpus_hinted), ("paraphrase", para)):
            report["sets"].setdefault(name, {})[kind] = score(retriever, queries)

    print(f"{len(rules)} rules; split={args.split}")
    print(f"{'set':<14}{'retriever':<10}{'n':>5}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'ms/q':>8}")
    for name, by_kind in report["sets"].items():
        for kind, s in by_kind.items():
            print(f"{name:<14}{kind:<10}{s['n']:>5}{s['recall@1']:>8.3f}{s['recall@3']:>8.3f}"
                  f"{s['recall@5']:>8.3f}{s['mrr']:>8.3f}{s['ms_per_query']:>8.1f}")
    out = args.out or Path("runs") / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
