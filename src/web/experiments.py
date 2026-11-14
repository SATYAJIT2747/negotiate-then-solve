"""Sections of the Experiments page beyond the main negotiation run and the model table (``web.app``).

Each section is a table the page draws the same way::

    {"id", "kind": "table", "icon", "title", "subtitle", "at",
     "columns": [{"key", "label", "hint", "fmt": pct|num|int|text|p, "better": high|low|None, "bar": bool}],
     "rows": [{"label", "sub", "highlight", "cells": {key: value}}],
     "notes": [str]}

Every number is read from a result file in ``runs/`` (or, for the agent adapters, ``docs/regression-matrix.md``);
a section whose files are missing is left out, so the same module serves both branches.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _col(key, label, fmt="num", better=None, hint="", bar=False, digits=None):
    return {"key": key, "label": label, "fmt": fmt, "better": better, "hint": hint, "bar": bar, "digits": digits}


def _row(label, cells, sub="", highlight=False):
    return {"label": label, "sub": sub, "highlight": highlight, "cells": cells}


def _table(id_, icon, title, subtitle, at, columns, rows, notes=()):
    return {"id": id_, "kind": "table", "icon": icon, "title": title, "subtitle": subtitle, "at": at,
            "columns": columns, "rows": rows, "notes": list(notes)}


# -- same model for both: ours vs B4 on Gemini ----------------------------------------------------------------


def bias_check(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-bias-gemini.json")
    main, _ = _load(runs / "negotiation-main-ours-4b.json")
    if not rep:
        return []
    ids = {r["scenario"] for r in rep["configs"]["ours-llm"]["rows"]}
    rows = []

    want = {"feasible": {"feasible"}, "agree": {"agreed"}, "escalate": {"escalated", "failed"}}  # correct_outcome

    def cells(rs: list[dict], s: dict | None = None) -> dict:
        """From the run's summary where there is one, else from the rows (the 4B run's subset)."""
        s = s or {}
        negotiated = [r for r in rs if r.get("rounds")]
        return {"resolved": sum(r["status"] == "agreed" for r in rs),
                "correct": s.get("correct_outcome", sum(r["status"] in want.get(r.get("expected"), ()) for r in rs)
                                 / len(rs) if rs else None),
                "gap": s.get("objective_gap_abs_mean"),
                "calls": s.get("option_calls_per_scenario",
                               sum(r.get("option_calls", 0) for r in negotiated) / len(negotiated) if negotiated else None),
                "first": s.get("first_proposal_acceptance",
                               sum(bool(r.get("first_accepted")) for r in negotiated) / len(negotiated)
                               if negotiated else None),
                "n": len(rs)}

    if main:
        sub = [r for r in main["configs"]["ours-llm"]["rows"] if r["scenario"] in ids and r.get("seed", 0) == 0]
        rows.append(_row("Our system", cells(sub), "Qwen3.5-4B + LoRA (the main run, same 20 scenarios)", True))
    for k, label in (("ours-llm", "Our system"), ("B4", "LLM proposes, solver filters (B4)")):
        rows.append(_row(label, cells(rep["configs"][k]["rows"], rep["configs"][k]["summary"]), "Gemini Flash-Lite",
                         k == "ours-llm"))
    p = (rep.get("paired") or {}).get("ours-llm vs B4", {}).get("resolved_mcnemar", {})
    return [_table(
        "bias", "scale", "Same model for both: is the gap the options or the model?",
        "The main run had our system on the local 4B and the LLM baselines on Gemini. Here both run on Gemini, "
        "on a stratified 20 scenarios, seed 0.", at,
        [_col("resolved", "Resolved", "int", "high", "Conflicts settled by agreement"),
         _col("correct", "Right outcome", "pct", "high", "Ended as the oracle did"),
         _col("gap", "Gap to oracle", "num", "low", "Distance of the final timetable's cost from the oracle's"),
         _col("calls", "LLM calls per scenario", "num", "low", "Calls to invent options (B4 regenerates)"),
         _col("first", "First offer taken", "pct", "high")],
        rows,
        [f"Our system resolves the same conflicts on Gemini as on the 4B; the gap to B4 comes from where the options "
         f"come from. With 20 scenarios the McNemar test is {p.get('only_first', '?')} vs {p.get('only_second', '?')} "
         f"discordant (p = {p.get('p', float('nan')):.2f}); the 180-pair main run is significant."])]


# -- ablations, paired ------------------------------------------------------------------------------------------

ABLATION_NAMES = {"A1": ("Without tiers or ledger", "A1"), "A2": ("Without grounded explanations", "A2"),
                  "A3": ("Without LLM reply reading", "A3"),
                  "A1-offline": ("Without tiers or ledger, scripted (all 60)", "A1")}


def ablations(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-ablations-paired.json")
    if not rep:
        return []
    rows = []
    for key, v in rep.items():
        ref, abl = key.split(" vs ")
        name, _ = ABLATION_NAMES.get(abl, (abl, abl))
        if not rows or ref == "ours":
            rows.append(_row("Our system" + (", scripted" if ref == "ours" else ""), {
                "resolution": v["resolution_rate"][ref], "correct": v["correct_outcome"][ref],
                "gini": v["burden"][ref]["gini"], "leaks": v["leaks"][ref] * v["pairs"],
                "first": v["first_proposal_acceptance"][ref], "changed": None,
                "clarity": (v.get("judge_clarity") or {}).get(ref),
                "accept": (v.get("judge_acceptability") or {}).get(ref)},
                f"{v['pairs']} scenarios, seed 0", True))
        rows.append(_row(name, {
            "resolution": v["resolution_rate"][abl], "correct": v["correct_outcome"][abl],
            "gini": v["burden"][abl]["gini"], "leaks": v["leaks"][abl] * v["pairs"],
            "first": v["first_proposal_acceptance"][abl], "changed": v["different_conceders"],
            "clarity": (v.get("judge_clarity") or {}).get(abl),
            "accept": (v.get("judge_acceptability") or {}).get(abl)}, v["isolates"]))
    return [_table(
        "ablations", "flask", "Ablations: our system minus one part, on the same scenarios",
        "Each ablation paired with our system on the same scenario and seed (stratified 30, seed 0; the scripted "
        "pair on all 60).", at,
        [_col("resolution", "Resolved", "pct", "high"), _col("correct", "Right outcome", "pct", "high"),
         _col("changed", "Changes who gives way", "text", None, "Resolved conflicts where a different person conceded"),
         _col("gini", "Burden Gini", "num", "low", "Unevenness of concessions on these scenarios; 0 = even"),
         _col("leaks", "Privacy leaks", "int", "low", "Private reasons that reached a colleague"),
         _col("first", "First offer taken", "pct", "high"),
         _col("clarity", "Clarity (judge)", "num", "high", "LLM judge, 1-5; not yet validated against people"),
         _col("accept", "Acceptability (judge)", "num", "high", "LLM judge, 1-5; not yet validated against people")],
        rows,
        ["A1: the benchmark's conflicts rarely meet the same person twice, so flat weights almost never change who "
         "is asked; fairness is tested by the stress test below.",
         "A2: free explanations leaked a private reason in every scenario that had one to leak (4 of 4); grounded "
         "ones in none.",
         "A3: the replies here are almost all plain accepts and refusals; on the 100-reply benchmark the keyword "
         "handler gets 66% of actions right against 100% for the fine-tuned 4B (model table)."])]


# -- fairness stress test ---------------------------------------------------------------------------------------

FAIRNESS_NAMES = {
    "ours": ("Our system (ledger)", "Tiers, weights, ledger credit and the fairness term", True),
    "no-ledger": ("Same policy, no ledger", "Credit and fairness terms off: isolates the ledger", False),
    "flat": ("Flat weights (A1)", "Every option costs the same; no ledger", False),
    "ours-smooth": ("Smooth credit (experimental)", "Credit 1 - 0.5^c instead of min(1, c); not in the live system",
                    False),
}


def fairness(runs: Path) -> list[dict]:
    rep, at = _load(runs / "fairness-final.json")
    if not rep:
        return []
    rows = []
    for k, s in rep["summary"].items():
        name, sub, hi = FAIRNESS_NAMES.get(k, (k, "", False))
        times = " / ".join(f"{s['times'][t]:.1f}" for t in sorted(s["times"]))
        rows.append(_row(name, {"gini": s["gini"], "max": s["max"], "times": times,
                                "resolved": s["resolution_rate"], "students": s.get("students_per_conflict")},
                         sub, hi))
    pv = rep.get("paired", {})

    def p(key, m):
        return pv.get(key, {}).get(m, {}).get("p")

    return [_table(
        "fairness", "scale", "Fairness: who gives way over a semester",
        f"One department, the same four teachers, {rep['conflicts']} lab clashes per semester, {rep['seeds']} "
        "semesters. F-301 and F-304 teach the smallest section, so moving their practical is always cheapest on "
        "paper. Every configuration meets the same conflicts; scripted replies, no LLM.", at,
        [_col("gini", "Burden Gini", "num", "low", "0 = everyone gives way equally", bar=True),
         _col("max", "Most burdened", "num", "low", "Largest weighted burden on one teacher"),
         _col("times", "Times gave way (F-301 / 302 / 303 / 304)", "text"),
         _col("resolved", "Resolved", "pct", "high"),
         _col("students", "Students affected per conflict", "num", "low",
              "Students in the section of the teacher who gave way")],
        rows,
        [f"Ledger vs no ledger: Gini {rep['summary']['ours']['gini']:.2f} vs {rep['summary']['no-ledger']['gini']:.2f} "
         f"(paired Wilcoxon p = {p('ours vs no-ledger', 'gini'):.3f}), no resolutions lost: the ledger spreads the "
         "burden.",
         f"Ledger vs flat: Gini not different (p = {p('ours vs flat', 'gini'):.2f}), but flat affects more students "
         f"(p = {p('ours vs flat', 'students_per_conflict'):.3f}): it is even because it ignores who is affected.",
         f"Smooth credit vs the live ledger: Gini p = {p('ours-smooth vs ours', 'gini'):.3f}. The live credit stops "
         "growing after about one concession."])]


# -- RAG --------------------------------------------------------------------------------------------------------


def rag(runs: Path) -> list[dict]:
    files = [("rag-4b-read100.json", "k = 5"), ("rag-4b-read100-k8.json", "k = 8")]
    rows, at = [], None
    for name, k in files:
        rep, a = _load(runs / name)
        if not rep:
            continue
        at = max(x for x in (at, a) if x)
        for cond, block in rep["conditions"].items():
            if cond == "no-rag" and any(r["label"].startswith("No retrieval") for r in rows):
                continue
            s = block["summary"]
            label = f"Retrieval ({rep.get('retriever', 'hybrid')}, {k})" if cond == "rag" else "No retrieval"
            rows.append(_row(label, {"accuracy": s["answer_accuracy"], "cite_recall": s["citation_recall"],
                                     "cite_precision": s["citation_precision"], "retrieval": s["retrieval_recall"],
                                     "abstain": s["abstention_on_unanswerable"],
                                     "hallucinate": s["hallucination_on_unanswerable"],
                                     "false_abstain": s["false_abstention"]},
                             "the model alone, no rules shown" if cond == "no-rag"
                             else "as deployed" if k == "k = 8" else "the default until 2026-11-14",
                             cond == "rag" and k == "k = 8"))
    out = []
    if rows:
        out.append(_table(
            "rag", "book", "Policy questions: with and without retrieval",
            "40 questions worded unlike the rules, and 20 that no rule covers (leave, attendance, fees). Policy agent "
            "on Qwen3.5-4B + LoRA; answers labelled by a Gemini judge.", at,
            [_col("accuracy", "Answer accuracy", "pct", "high", "Answered and consistent with the gold rule", bar=True),
             _col("retrieval", "Rule retrieved", "pct", "high", "The gold rule among those shown"),
             _col("cite_recall", "Cites the rule", "pct", "high"),
             _col("cite_precision", "Citation precision", "pct", "high"),
             _col("abstain", "Says 'no rule' when none", "pct", "high", "On the 20 questions no rule covers"),
             _col("hallucinate", "Invents policy", "pct", "low", "A substantive answer when no rule exists"),
             _col("false_abstain", "Wrongly says 'no rule'", "pct", "low")],
            rows,
            ["Without retrieval the model invents institute policy (\"10 days of casual leave\", \"75% attendance\").",
             "At k = 5 most misses are retrieval misses (the rule was not among the five shown). At k = 8 the rule is "
             "shown for 38 of 40 questions, but most remaining misses cite a neighbouring rule that was shown beside "
             "it: the bottleneck moves from retrieval to choosing between similar rules."]))
    rk, at = _load(runs / "retrieval-k.json")
    if rk:
        rec = rk["recall"]
        ks = sorted({int(x.split("@")[1]) for x in rec})
        rows = [_row(kind.upper() if kind == "bm25" else kind.capitalize(),
                     {f"k{k}": rec.get(f"{kind}@{k}") for k in ks},
                     {"bm25": "words", "dense": "sentence embeddings", "hybrid": "both, rank-fused (deployed)"}[kind],
                     kind == "hybrid")
                for kind in ("bm25", "dense", "hybrid")]
        out.append(_table(
            "retrieval", "search", "Retrieval: is the right rule among those shown?",
            f"Recall on the {rk['queries']} paraphrased questions, by retriever and number of rules shown (k). "
            "No model calls.", at,
            [_col(f"k{k}", f"k = {k}", "pct", "high", bar=k == 8) for k in ks], rows,
            ["Hybrid at k = 8 finds the rule for 38 of 40 questions, against 34 at k = 5."]))
    return out


# -- safety -----------------------------------------------------------------------------------------------------

SAFETY_RUNS = [
    ("safety-compromised-noguards.json", "Checks off, compromised model", "Baseline: no deterministic checks, no "
     "human approval, and a model that obeys every instruction", False),
    ("safety-compromised.json", "Checks on, compromised model", "The same model; only the code checks stand", False),
    ("safety-4b-noguards.json", "Checks off, Qwen3.5-4B + LoRA", "What the model resists on its own", False),
    ("safety-4b.json", "Checks on, Qwen3.5-4B + LoRA", "The system as deployed", True),
    ("safety-gemini.json", "Checks on, Gemini Flash-Lite", "The earlier run with the prompted parser", False),
]


def safety(runs: Path) -> list[dict]:
    rows, at = [], None
    for name, label, sub, hi in SAFETY_RUNS:
        rep, a = _load(runs / name)
        if not rep:
            continue
        at = max(x for x in (at, a) if x)
        rows.append(_row(label, {"pass": rep["pass_rate"], "unapproved": rep["unapproved_publishes"],
                                 "authority": rep["authority_violations"], "injection": rep["injection_successes"],
                                 "leaks": rep["private_leaks"]}, sub, hi))
    if not rows:
        return []
    return [_table(
        "safety", "shield", "Safety: 50 attacks, with and without the deterministic checks",
        "Students and colleagues changing someone else's classes, claimed authority, \"I approve\" emails, prompt "
        "injections, questions about a colleague's private reason, spoofed senders.", at,
        [_col("pass", "Cases handled safely", "pct", "high", bar=True),
         _col("unapproved", "Published without approval", "int", "low"),
         _col("authority", "Authority violations", "int", "low"),
         _col("injection", "Injections obeyed", "int", "low"),
         _col("leaks", "Private reasons leaked", "int", "low")],
        rows,
        ["The checks (sender verification, injection pre-filter, authority check, human approval) are code, not "
         "prompts: with them on, even a model that obeys every injection publishes nothing unapproved.",
         "Privacy holds in every row because replies are built from constraints, never from the reasons given."])]


# -- the agentic branch: pipeline vs agent, and the agent adapters ------------------------------------------------


def head_to_head(runs: Path) -> list[dict]:
    base = runs / "head_to_head"
    full = base / "v3" / "results.jsonl"
    if not full.exists():
        return []
    results = [json.loads(x) for x in full.read_text(encoding="utf-8").splitlines() if x.strip()]
    rerun = base / "v4-aside" / "results.jsonl"
    fixed = {r["id"]: r for r in (json.loads(x) for x in rerun.read_text(encoding="utf-8").splitlines() if x.strip())
             } if rerun.exists() else {}
    at = datetime.fromtimestamp(max(p.stat().st_mtime for p in (full, rerun) if p.exists()), UTC).isoformat()
    systems = {"intents": "Pipeline (intents adapter)", "agent": "Agent loop (agent adapter)"}
    rows = []
    for sys_, label in systems.items():
        rs = [r for r in results if r["system"] == sys_]
        if sys_ == "agent":
            rs = [fixed.get(r["id"], r) for r in rs]
        if not rs:
            continue
        times = sorted(r.get("seconds") or r.get("elapsed") or 0 for r in rs)
        changes = [r for r in rs if r.get("checks")]
        rows.append(_row(label, {
            "pass": sum(bool(r["pass"]) for r in rs) / len(rs), "passed": f"{sum(bool(r['pass']) for r in rs)}/{len(rs)}",
            "action": sum(bool(r["action_ok"]) for r in rs) / len(rs),
            "checks": f"{sum(bool(r['checks_ok']) for r in changes)}/{len(changes)}",
            "median": times[len(times) // 2] if any(times) else None},
            "after the absence fix (v4) for the five it reran" if sys_ == "agent" and fixed else "", sys_ == "intents"))
    return [_table(
        "head-to-head", "swords", "Pipeline vs agent on the same 41 requests",
        "The same demo department and week, reset before each request: the fixed pipeline with the intents adapter "
        "against the tool-calling agent loop.", at,
        [_col("passed", "Passed", "text"), _col("pass", "Pass rate", "pct", "high", bar=True),
         _col("action", "Right action", "pct", "high", "Changed, answered, asked or refused as it should"),
         _col("checks", "Timetable checks hold", "text", None, "For requests that change the timetable"),
         _col("median", "Median seconds", "num", "low")],
        rows)]


def agent_adapters(root: Path) -> list[dict]:
    doc = root / "docs" / "regression-matrix.md"
    if not doc.exists():
        return []
    text = doc.read_text(encoding="utf-8")
    sec = text.split("## Agent: passed / n per family", 1)
    if len(sec) < 2:
        return []
    lines = [x for x in sec[1].splitlines() if x.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    allrow = next((x for x in lines if x.startswith("| **all**")), None)
    if allrow is None:
        return []
    cells = [c.strip() for c in allrow.strip("|").split("|")]
    rows = []
    for name, cell in zip(head[1:], cells[1:], strict=False):
        m = re.match(r"\*\*(\d+)/(\d+)\*\* \[([\d.]+), ([\d.]+)\]", cell)
        if not m or "/" in name:  # only the main run of each adapter, not the reruns of a few families
            continue
        k, n = int(m[1]), int(m[2])
        rows.append(_row(f"nts-agent-{name}", {"passed": f"{k}/{n}", "rate": k / n,
                                               "ci": f"{float(m[3]):.2f}-{float(m[4]):.2f}"},
                         {"v6-dpo": "deployed", "v7-sft": "v7 data, SFT", "v7-dpo": "v7 data, SFT + DPO"}.get(name, ""),
                         name == "v6-dpo"))
    if not rows:
        return []
    at = datetime.fromtimestamp(doc.stat().st_mtime, UTC).isoformat()
    return [_table(
        "agent-adapters", "cpu", "Agent adapters: episodes passed",
        "49 request families x 2 departments x 5 episodes on Kaggle (docs/regression-matrix.md). The release rule keeps "
        "the old adapter unless a new one is better.", at,
        [_col("passed", "Passed", "text"), _col("rate", "Pass rate", "pct", "high", bar=True),
         _col("ci", "95% CI", "text")], rows,
        ["v7 did not pass the release rule (regressions on counter-offers, no reply, fallbacks), so v6-dpo stays."])]


def sections(root: Path, agentic: bool = False) -> list[dict]:
    """Every section after the main run and the model table, in page order; ``agentic``: also the agent
    system's own (pipeline vs agent, the agent adapters)."""
    runs = root / "runs"
    out = [*bias_check(runs), *ablations(runs), *fairness(runs), *rag(runs), *safety(runs)]
    if agentic:
        out = [*head_to_head(runs), *agent_adapters(root), *out]
    return out
