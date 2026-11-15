"""Policy question answering with and without retrieval, and abstention
(brief Benchmark C: "40 + 20 queries"; the RAG row of the capability table).

Questions, each sent to the policy agent as a policy question (the parse the
pipeline gives one):

* 40 answerable questions worded unlike the rules (``data/retrieval-queries.jsonl``,
  gold rule IDs);
* 20 questions no rule covers (``data/abstention-queries.jsonl``): leave,
  grading, fees, parking. The right answer is "no rule about that".

Conditions:

* ``rag``      the agent as deployed: top-k retrieval (hybrid), the model answers from
               the rules shown, and an answer citing no rule shown becomes "no rule"
* ``no-rag``   the same model and prompt with no rules shown ("(none found)"), its
               answer taken as written: what the model says from its own knowledge

A judge (Gemini, batched) labels every answer ``abstain`` (says it does not know
or no rule covers it) or ``answer``, and for answerable questions whether the
answer agrees with the gold rule text. Metrics: answer accuracy (answered and
agrees), citation precision and recall (rag only), abstention on unanswerable
questions, false abstention on answerable ones, and hallucination (a substantive
answer to an unanswerable question).

    uv run python -m evaluation.rag --local http://10.1.25.191:8080/v1 --adapter nts-read100
    uv run python -m evaluation.rag                       # the agent on Gemini
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agents.policy import NO_RULE, SYSTEM_PROMPT, PolicyAgent, PolicyOutput
from core.instance import Instance
from core.schemas import Channel, Request, Role
from language.llm import GeminiClient, default_model
from language.parsing import ParseOutput, ParseResult

from .stats import bootstrap_ci, mean

JUDGE_PROMPT = """You check answers given by a university's policy assistant.
For each item you get the question, the answer, and (when one exists) the
official rule text. Return:
- abstains: true if the answer declines, says it does not know, or says no
  rule or regulation covers the question; false if it states a substantive
  answer (a number, a procedure, a yes/no, a person who decides).
- agrees: only when a rule is given and the answer does not abstain: true if
  what the answer states is consistent with the rule and adds no claim the
  rule contradicts; otherwise false. Use null when no rule is given or the
  answer abstains."""


class _Verdict(BaseModel):
    id: str
    abstains: bool
    agrees: bool | None = None


class _Verdicts(BaseModel):
    items: list[_Verdict]


def _parse(instance: Instance, text: str, i: int) -> ParseResult:
    from datetime import UTC

    req = Request(id=f"Q-{i:03d}", channel=Channel.PORTAL, sender_id=instance.faculty[0].id, role=Role.FACULTY,
                  raw_text=text, received_at=datetime.now(UTC))
    return ParseResult(request=req, output=ParseOutput(request_type="policy_question", action="answer"))


def ask(agent: PolicyAgent, instance: Instance, q: dict, i: int, condition: str) -> dict:
    parse = _parse(instance, q["query"], i)
    if condition == "rag":
        d = agent.review(q["query"], parse)
        return {"answer": d.answer or d.explanation, "cited": d.cited, "retrieved": d.retrieved,
                "no_rule": d.answer == NO_RULE, "hallucinated_citations": d.hallucinated}
    out = agent.client.generate(SYSTEM_PROMPT, agent.build_prompt(q["query"], parse, []), PolicyOutput)
    return {"answer": out.answer or out.explanation, "cited": [], "retrieved": [], "no_rule": False,
            "hallucinated_citations": list(out.cited_rules)}


def judge(rows: list[dict], rules: dict, client, batch: int = 10) -> None:
    for k in range(0, len(rows), batch):
        chunk = rows[k: k + batch]
        items = []
        for r in chunk:
            gold = "\n".join(f"[{g}] {rules[g].text}" for g in r["gold"] if g in rules) or "(no rule)"
            items.append(f'<item id="{r["key"]}">\nQuestion: {r["query"]}\nAnswer: {r["answer"]}\n'
                         f'Rule: {gold}\n</item>')
        got = client.generate(JUDGE_PROMPT, "\n\n".join(items), _Verdicts)
        by = {v.id: v for v in got.items}
        for r in chunk:
            v = by.get(r["key"])
            r["abstains"] = v.abstains if v else None
            r["agrees"] = v.agrees if v else None


def score(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["gold"]]
    una = [r for r in rows if not r["gold"]]

    def abstained(r):
        return r["no_rule"] or bool(r.get("abstains"))

    correct = [(not abstained(r)) and bool(r.get("agrees")) for r in ans]
    cited = [r for r in ans if r["cited"]]
    return {
        "answerable": len(ans), "unanswerable": len(una),
        "answer_accuracy": mean(correct), "answer_accuracy_ci": bootstrap_ci(correct),
        "false_abstention": mean([abstained(r) for r in ans]),
        "citation_recall": mean([bool(set(r["cited"]) & set(r["gold"])) for r in ans]),
        "citation_precision": (sum(len(set(r["cited"]) & set(r["gold"])) for r in cited)
                               / sum(len(r["cited"]) for r in cited)) if cited else None,
        "retrieval_recall": mean([bool(set(r["retrieved"]) & set(r["gold"])) for r in ans]) if any(
            r["retrieved"] for r in ans) else None,
        "abstention_on_unanswerable": mean([abstained(r) for r in una]),
        "hallucination_on_unanswerable": mean([not abstained(r) for r in una]),
        "hallucinated_citations": sum(len(r["hallucinated_citations"]) for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, default=Path("data/retrieval-queries.jsonl"))
    ap.add_argument("--unanswerable", type=Path, default=Path("data/abstention-queries.jsonl"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--handbook", type=Path, default=Path("data/handbook.md"))
    ap.add_argument("--policies", type=Path, default=Path("data/policies"))
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--local", metavar="URL", nargs="?", const="http://localhost:8080/v1", default=None)
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--adapter", default=None, help="LoRA adapter for the policy agent (e.g. nts-read100)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from agents.documents import load_corpus, ocr_backend

    rules, _ = load_corpus(args.handbook, args.policies, ocr_backend())
    by_id = {r.id: r for r in rules}
    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    if args.local:
        from language.local import LocalClient

        client = LocalClient(base_url=args.local, server_model=args.local_model, adapter=args.adapter,
                             max_tokens=1024, timeout=600)
    else:
        client = GeminiClient(default_model("agent"))
    agent = PolicyAgent(rules, client, instance, k=args.k, retriever=args.retriever)
    queries = [json.loads(x) for p in (args.queries, args.unanswerable)
               for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    for q in queries:
        q["rules"] = [g for g in q["rules"] if g in by_id]

    report: dict = {"model": getattr(client, "model", None), "retriever": args.retriever, "k": args.k,
                    "conditions": {}}
    jclient = GeminiClient(default_model("simulator"))
    for condition in ("rag", "no-rag"):
        with ThreadPoolExecutor(args.workers) as pool:
            outs = list(pool.map(lambda iq: ask(agent, instance, iq[1], iq[0], condition), enumerate(queries)))
        rows = [{"key": f"{condition}/{q['id']}", "id": q["id"], "query": q["query"], "gold": q["rules"], **o}
                for q, o in zip(queries, outs, strict=True)]
        judge(rows, by_id, jclient)
        report["conditions"][condition] = {"summary": score(rows), "rows": rows}
        print(f"== {condition}\n" + json.dumps(report["conditions"][condition]["summary"], indent=1, default=str),
              flush=True)
    out = args.out or Path("runs") / f"rag-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
