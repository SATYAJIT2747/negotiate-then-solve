"""Policy agent evaluation (RQ-RAG row of Table 11).

    uv run python -m evaluation.policy --corpus data/requests-para.jsonl --split val
    uv run python -m evaluation.policy --split val --local --workers 1   # parser and agent on llama-server

Each request is parsed (System Two, cached) and, if the parser compiles,
asks for clarification or recognises a policy question, reviewed by the
policy agent. Scored against the corpus labels:

* allow/deny accuracy: rule-breaking requests must come back ``forbidden``,
  everything else must not;
* citation: a denial must cite the rule broken; an answer the rule asked about;
* obligations: a one-off absence must carry the make-up rule (P-MAKEUP);
* retrieval recall@k and hallucinated citations.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from agents.policy import RETRIEVERS, PolicyAgent, load_handbook
from core.instance import Instance
from language.adapters import add_args
from language.corpus import CorpusExample, ExpectedAction, Variant, load_jsonl
from language.llm import DailyQuotaReached, GeminiClient, LLMError, default_model
from language.parsing import SystemTwoParser

SCORED = {ExpectedAction.COMPILE, ExpectedAction.DENY, ExpectedAction.ANSWER}


def _ratio(hits: int, n: int) -> float | None:
    return hits / n if n else None


def eval_policy(
    corpus: list[CorpusExample],
    instance: Instance,
    parser: SystemTwoParser,
    agent: PolicyAgent,
    *,
    split: str = "val",
    limit: int | None = None,
    workers: int = 4,
) -> dict:
    examples = [ex for ex in corpus if ex.split == split and ex.expected_action in SCORED
                and ex.variant != Variant.UNAUTHORISED][:limit]

    def run(ex: CorpusExample) -> dict:
        parse = parser.parse(ex.request)
        row = {"id": ex.id, "expected": ex.expected_action.value, "gold_rules": ex.rules,
               "parse_action": parse.action.value}
        if parse.action.value not in ("compile", "clarify", "answer"):
            row["skipped"] = True  # the orchestrator would not ask the policy agent
            return row
        d = agent.review(ex.request.raw_text, parse)
        row.update(verdict=d.verdict.value, cited=d.cited, obligations=d.obligations, retrieved=d.retrieved,
                   hallucinated=d.hallucinated, uncited=d.uncited_escalation, explanation=d.explanation,
                   alternative=d.alternative, answer=d.answer)
        return row

    rows: list[dict] = []
    stopped = None
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(run, ex) for ex in examples]
    for i, (ex, fut) in enumerate(zip(examples, futures, strict=True), 1):
        try:
            rows.append(fut.result())
        except DailyQuotaReached as e:
            stopped = str(e)
            pool.shutdown(cancel_futures=True)
            break
        except LLMError as e:
            rows.append({"id": ex.id, "error": str(e)})
        print(f"[{i}/{len(examples)}] {ex.id} policy calls={agent.client.usage.calls} "
              f"cached={agent.client.usage.cache_hits}", flush=True)
    pool.shutdown()

    reviewed = [r for r in rows if "verdict" in r]
    deny = [r for r in reviewed if r["expected"] == "deny"]
    allow = [r for r in reviewed if r["expected"] == "compile"]
    answers = [r for r in reviewed if r["expected"] == "answer"]
    makeup = [r for r in allow if "P-MAKEUP" in r["gold_rules"]]
    no_makeup = [r for r in allow if "P-MAKEUP" not in r["gold_rules"]]
    gold_total = sum(len(r["gold_rules"]) for r in reviewed)
    return {
        "model": agent.client.model,
        "retriever": agent.retriever_kind,
        "split": split,
        "n": len(rows),
        "reviewed": len(reviewed),
        "skipped_by_parser": [r["id"] for r in rows if r.get("skipped")],
        "errors": sum("error" in r for r in rows),
        "stopped": stopped,
        "allow_deny_accuracy": _ratio(sum((r["verdict"] == "forbidden") == (r["expected"] == "deny")
                                          for r in deny + allow), len(deny + allow)),
        "deny_recall": _ratio(sum(r["verdict"] == "forbidden" for r in deny), len(deny)),
        "deny_citation_correct": _ratio(sum(r["verdict"] == "forbidden" and set(r["gold_rules"]) <= set(r["cited"])
                                            for r in deny), len(deny)),
        "false_deny_rate": _ratio(sum(r["verdict"] == "forbidden" for r in allow), len(allow)),
        "needs_approval_rate": _ratio(sum(r["verdict"] == "needs_approval" for r in reviewed), len(reviewed)),
        "answer_citation_recall": _ratio(sum(set(r["gold_rules"]) <= set(r["cited"]) for r in answers), len(answers)),
        "makeup_obligation_recall": _ratio(sum("P-MAKEUP" in r["obligations"] for r in makeup), len(makeup)),
        "makeup_false_positive_rate": _ratio(sum("P-MAKEUP" in r["obligations"] for r in no_makeup), len(no_makeup)),
        "retrieval_recall": _ratio(sum(g in r["retrieved"] for r in reviewed for g in r["gold_rules"]), gold_total),
        "hallucinated_citations": sum(len(r["hallucinated"]) for r in reviewed),
        "uncited_escalations": sum(r["uncited"] for r in reviewed),
        "confusion": dict(Counter(f"{r['expected']}->{r['verdict']}" for r in reviewed)),
        "usage": vars(agent.client.usage),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/requests-para.jsonl"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--handbook", type=Path, default=Path("data/handbook.md"))
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parser-model", default=None, help="default: the agent model")
    ap.add_argument("--policy-model", default=None, help="default: the agent model")
    ap.add_argument("--k", type=int, default=5, help="rules retrieved per request")
    ap.add_argument("--retriever", default="bm25", choices=RETRIEVERS,
                    help="bm25 (the recorded runs), dense, or hybrid (BM25 + embeddings, reciprocal rank fusion)")
    ap.add_argument("--local", action="store_true",
                    help="the fine-tuned local model parses (compiler prompt) and reviews (policy prompt)")
    ap.add_argument("--local-url", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None, help="model to run on a router-mode server (default NTS_LOCAL_MODEL)")
    add_args(ap)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus, instance)
    if args.local:
        from language.adapters import local_client, registry
        from language.compiler import CompilerParser
        from language.intent_compiler import IntentCompilerParser

        reg = registry(args.adapters)
        policy_client = local_client("policy", base_url=args.local_url, timeout=600, server_model=args.local_model,
                                     adapter=args.adapter, reg=reg)
        reader = local_client("compile", base_url=args.local_url, timeout=600, server_model=args.local_model,
                              reg=reg) if reg else policy_client
        # NTS_INTENTS=1: the model reads intents (language.intent_compiler), as in the web app
        parser = (IntentCompilerParser if os.environ.get("NTS_INTENTS") == "1" else CompilerParser)(instance, reader)
    else:
        parser = SystemTwoParser(instance, GeminiClient(args.parser_model or default_model()))
        policy_client = GeminiClient(args.policy_model or default_model())
    agent = PolicyAgent(load_handbook(args.handbook), policy_client, instance, k=args.k, retriever=args.retriever)
    report = eval_policy(corpus, instance, parser, agent, split=args.split, limit=args.limit, workers=args.workers)
    print("Policy:", json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    out = args.out or Path("runs") / f"policy-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
