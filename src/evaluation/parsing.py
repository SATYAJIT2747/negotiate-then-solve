"""Parsing evaluation (RQ3 and the fine-tuning/tool metrics of Table 11).

    uv run python -m evaluation.parsing --limit 40
    uv run python -m evaluation.parsing --local --corpus data/requests-para.jsonl --limit 151 --workers 1

* System One: trained on the train split, tau tuned on val, scored on test
  (type/action accuracy, ECE, fast-path share, latency). No API calls.
* System Two: the Gemini parser on the first ``--limit`` test requests
  (action accuracy, constraint-compilation exact match and atom F1,
  authority refusals, injections, latency, tokens). Responses are cached
  under runs/llm_cache, so rerunning costs nothing.

Rule-breaking requests (expected ``deny``) are skipped for System Two action
accuracy: denial is the policy agent's job (L3), after parsing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from core.instance import Instance
from language.corpus import SLICES, CorpusExample, ExpectedAction, load_jsonl
from language.llm import DailyQuotaReached, GeminiClient, LLMError
from language.parsing import SystemTwoParser
from language.system_one import SystemOne, tune_tau

from .metrics import atom_counts, ece, exact_match, percentile, prf, semantic_match


def eval_system_one(corpus: list[CorpusExample], train_corpus: list[CorpusExample] | None = None) -> dict:
    """Train/val come from ``train_corpus`` (default: ``corpus``), test from ``corpus``."""
    source = train_corpus or corpus
    split = {s: [ex for ex in source if ex.split == s] for s in ("train", "val")}
    split["test"] = [ex for ex in corpus if ex.split == "test"]
    model = SystemOne().fit(split["train"])
    tau = tune_tau([model.decide(ex.request) for ex in split["val"]], split["val"])
    decisions = [model.decide(ex.request) for ex in split["test"]]
    test = split["test"]
    type_ok = [d.request_type == ex.request_type for d, ex in zip(decisions, test, strict=True)]
    action_ok = [d.action == ex.expected_action for d, ex in zip(decisions, test, strict=True)]
    fast = [d.fast_path(tau) for d in decisions]
    fast_ok = [t and a for t, a, f in zip(type_ok, action_ok, fast, strict=True) if f]
    latencies = [d.latency_ms for d in decisions]
    return {
        "n_test": len(test),
        "tau": tau,
        "type_accuracy": sum(type_ok) / len(test),
        "action_accuracy": sum(action_ok) / len(test),
        "type_ece": ece([d.type_p for d in decisions], type_ok),
        "action_ece": ece([d.action_p for d in decisions], action_ok),
        "fast_path_share": sum(fast) / len(test),
        "fast_path_precision": sum(fast_ok) / len(fast_ok) if fast_ok else None,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def by_slice(rows: list[dict]) -> dict:
    """Action accuracy and compilation exact match per test slice (unseen
    wording, unseen combination, ambiguous, adversarial, multi-constraint)."""
    out = {}
    for name in SLICES:
        got = [r for r in rows if name in r.get("slices", ()) and "error" not in r]
        compiled = [r for r in got if "exact" in r]
        out[name] = {
            "n": len(got),
            "action_accuracy": sum(r["expected"] == r["predicted"] for r in got) / len(got) if got else None,
            "compile_exact_match": sum(r["exact"] for r in compiled) / len(compiled) if compiled else None,
        }
    return out


def eval_system_two(
    corpus: list[CorpusExample],
    instance: Instance,
    client,
    limit: int,
    split: str = "test",
    workers: int = 4,
    parser=None,
) -> dict:
    """Requests run concurrently (``workers`` in flight); a Gemini client
    still starts at most RPM requests per minute, so this only hides latency.
    ``parser`` defaults to the Gemini System Two parser over ``client``."""
    parser = parser or SystemTwoParser(instance, client)
    examples = [ex for ex in corpus if ex.split == split][:limit]
    rows = []
    tp = n_pred = n_gold = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(parser.parse, ex.request) for ex in examples]
    for i, (ex, fut) in enumerate(zip(examples, futures, strict=True)):
        try:
            result = fut.result()
        except DailyQuotaReached as e:
            print(f"stopping early: {e}")
            pool.shutdown(cancel_futures=True)
            examples = examples[:i]
            break
        except LLMError as e:
            rows.append({"id": ex.id, "error": str(e)})
            continue
        u = client.usage
        print(f"[{i + 1}/{len(examples)}] {ex.id} calls={u.calls} cached={u.cache_hits} "
              f"retries={u.retries} {u.last_retry_reason[:80]}", flush=True)
        row = {
            "id": ex.id,
            "variant": ex.variant.value,
            "expected": ex.expected_action.value,
            "predicted": result.action.value,
            "type_ok": result.request_type == ex.request_type,
            "errors": result.errors,
            "refusal": result.refusal,
            "slices": ex.slices,
        }
        if ex.expected_action == ExpectedAction.COMPILE:
            row["exact"] = exact_match(result.constraints, ex.targets, instance)
            row["semantic"] = semantic_match(result.constraints, ex.targets, instance)
            counts = atom_counts(result.constraints, ex.targets, instance)
            tp, n_pred, n_gold = tp + counts[0], n_pred + counts[1], n_gold + counts[2]
            if not row["exact"]:
                row["pred"] = [c.model_dump(mode="json", exclude_none=True) for c in result.constraints]
                row["gold"] = [c.model_dump(mode="json", exclude_none=True) for c in ex.targets]
        rows.append(row)
    pool.shutdown()

    scored =[r for r in rows if "error" not in r and r["expected"] != ExpectedAction.DENY.value]
    compiled = [r for r in rows if "exact" in r]
    injected = [r for r, ex in zip(rows, examples, strict=True) if ex.injection and "error" not in r]
    p, r, f1 = prf(tp, n_pred, n_gold)
    return {
        "model": client.model,
        "split": split,
        "n": len(examples),
        "api_errors": sum("error" in r for r in rows),
        "action_accuracy": sum(r["expected"] == r["predicted"] for r in scored) / len(scored) if scored else None,
        "type_accuracy": sum(r["type_ok"] for r in rows if "error" not in r) / max(1, len(rows)),
        "compile_exact_match": sum(r["exact"] for r in compiled) / len(compiled) if compiled else None,
        "compile_semantic_match": sum(r["semantic"] for r in compiled) / len(compiled) if compiled else None,
        "compile_atom_precision": p,
        "compile_atom_recall": r,
        "compile_atom_f1": f1,
        "injections_handled": sum(r["expected"] == r["predicted"] for r in injected),
        "injections_total": len(injected),
        "confusion": dict(Counter(f"{r['expected']}->{r['predicted']}" for r in scored if r["expected"] != r["predicted"])),
        "by_slice": by_slice(rows),
        "workers": workers,
        "latency_s_p50": percentile(client.latencies, 50),
        "latency_s_p95": percentile(client.latencies, 95),
        "usage": vars(client.usage),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/requests.jsonl"))
    ap.add_argument("--s1-train", type=Path, default=None,
                    help="train System One on this corpus's train/val splits (default: --corpus)")
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--limit", type=int, default=40, help="requests sent to System Two")
    ap.add_argument("--workers", type=int, default=4, help="requests in flight at once (RPM still enforced)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="split for System Two; iterate prompts on val, report test once")
    ap.add_argument("--model", default=None, help="Gemini model id (default: the pinned agent model, or GEMINI_MODEL)")
    ap.add_argument("--local", metavar="URL", nargs="?", const="http://localhost:8080/v1", default=None,
                    help="use the fine-tuned compiler on a local OpenAI-compatible server instead of Gemini")
    ap.add_argument("--local-model", default=None, help="model to run on a router-mode server (default NTS_LOCAL_MODEL)")
    ap.add_argument("--prompted", action="store_true",
                    help="with --local: give the local model the full prompt the Gemini parser gets (the prompted "
                         "base-model baseline) instead of the fine-tuned compiler's short prompt")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus, instance)
    s1_train = load_jsonl(args.s1_train, instance) if args.s1_train else None
    report: dict = {"corpus": str(args.corpus), "s1_train": str(args.s1_train or args.corpus),
                    "system_one": eval_system_one(corpus, s1_train)}
    print("System One:", json.dumps(report["system_one"], indent=1))
    if not args.skip_llm:
        parser = None
        if args.local:
            from language.compiler import CompilerParser
            from language.local import LocalClient
            client = LocalClient(args.model, base_url=args.local, server_model=args.local_model)
            parser = None if args.prompted else CompilerParser(instance, client)
        else:
            client = GeminiClient(args.model)
        report["system_two"] = eval_system_two(corpus, instance, client, args.limit, args.split, args.workers,
                                               parser=parser)
        print("System Two:", json.dumps({k: v for k, v in report["system_two"].items() if k != "rows"}, indent=1))

    out = args.out or Path("runs") / f"parsing-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
