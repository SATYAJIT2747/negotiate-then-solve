"""Sections of the Experiments page beyond the main negotiation run and the model table (``web.app``).

The page leads with the answers: ``findings`` (one line per hypothesis or capability, with a verdict and the
numbers behind it), then one card per experiment. A card says its result in one sentence (``takeaway``), charts
its main measure (``chart``: a column key), shows the few columns that carry the result (``key`` columns) and keeps
the rest behind "all measures"::

    {"id", "kind": "table", "icon", "title", "question", "takeaway", "subtitle", "at", "chart",
     "columns": [{"key", "label", "hint", "fmt": pct|num|int|text|p, "better": high|low|None, "key_col": bool}],
     "rows": [{"label", "tag", "sub", "highlight", "cells": {key: value}}],
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


def _col(key, label, fmt="num", better=None, hint="", key_col=True, digits=None):
    return {"key": key, "label": label, "fmt": fmt, "better": better, "hint": hint, "key_col": key_col, "digits": digits}


def _row(label, cells, sub="", highlight=False, tag=""):
    return {"label": label, "tag": tag, "sub": sub, "highlight": highlight, "cells": cells}


def _table(id_, icon, title, question, takeaway, subtitle, at, chart, columns, rows, notes=()):
    return {"id": id_, "kind": "table", "icon": icon, "title": title, "question": question, "takeaway": takeaway,
            "subtitle": subtitle, "at": at, "chart": chart, "columns": columns, "rows": rows, "notes": list(notes)}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _p(x: float | None) -> str:
    return "–" if x is None else "p < 0.001" if x < 0.001 else f"p = {x:.3f}" if x < 0.01 else f"p = {x:.2f}"


# -- same model for both: ours vs B4 on Gemini ----------------------------------------------------------------


def bias_check(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-bias-gemini.json")
    main, _ = _load(runs / "negotiation-main-ours-4b.json")
    if not rep:
        return []
    ids = {r["scenario"] for r in rep["configs"]["ours-llm"]["rows"]}
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
                               if negotiated else None)}

    rows = []
    if main:
        sub = [r for r in main["configs"]["ours-llm"]["rows"] if r["scenario"] in ids and r.get("seed", 0) == 0]
        rows.append(_row("Our system on the 4B", cells(sub), "the main run, on the same 20 scenarios", True))
    ours, b4 = rep["configs"]["ours-llm"], rep["configs"]["B4"]
    rows.append(_row("Our system on Gemini", cells(ours["rows"], ours["summary"]), "the same agent model as B4", True))
    rows.append(_row("LLM proposes, solver filters, on Gemini", cells(b4["rows"], b4["summary"]),
                     "the main baseline", tag="B4"))
    p = (rep.get("paired") or {}).get("ours-llm vs B4", {}).get("resolved_mcnemar", {})
    r_ours, r_b4 = rows[-2]["cells"]["resolved"], rows[-1]["cells"]["resolved"]
    return [_table(
        "bias", "scale", "Same model for both",
        "The main run had our system on the local 4B and the LLM baselines on Gemini. Is the gap just the model?",
        f"No. On the same model our system resolves {r_ours} conflicts and the baseline {r_b4}; our system resolves the "
        "same ones on Gemini as on the 4B.",
        "Stratified 20 scenarios, seed 0.", at, "resolved",
        [_col("resolved", "Conflicts resolved", "int", "high", "Settled by agreement, of 20 scenarios"),
         _col("correct", "Right outcome", "pct", "high", "Ended as the oracle did"),
         _col("gap", "Gap to oracle", "num", "low", "Distance of the final timetable's cost from the oracle's"),
         _col("calls", "LLM calls per conflict", "num", "low", "Calls to invent options", key_col=False),
         _col("first", "First offer taken", "pct", "high", key_col=False)],
        rows,
        [f"20 scenarios are too few for significance on their own ({p.get('only_first', '?')} vs "
         f"{p.get('only_second', '?')} discordant, {_p(p.get('p'))}); the 180-pair main run is significant."])]


# -- ablations, paired ------------------------------------------------------------------------------------------

ABLATION_NAMES = {"A1": ("Without tiers or ledger", "A1"), "A2": ("Without grounded explanations", "A2"),
                  "A3": ("Without LLM reply reading", "A3")}


def ablations(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-ablations-paired.json")
    if not rep:
        return []
    rows = []
    for key, v in rep.items():
        ref, abl = key.split(" vs ")
        if abl not in ABLATION_NAMES:  # the scripted pair is the fairness evidence's footnote, not a row here
            continue
        name, tag = ABLATION_NAMES[abl]
        if not rows:
            rows.append(_row("Our system", {
                "resolution": v["resolution_rate"][ref], "leaks": round(v["leaks"][ref] * v["pairs"]),
                "first": v["first_proposal_acceptance"][ref], "changed": "–",
                "clarity": (v.get("judge_clarity") or {}).get(ref),
                "accept": (v.get("judge_acceptability") or {}).get(ref)},
                f"{v['pairs']} scenarios, seed 0", True))
        rows.append(_row(name, {
            "resolution": v["resolution_rate"][abl], "leaks": round(v["leaks"][abl] * v["pairs"]),
            "first": v["first_proposal_acceptance"][abl], "changed": v["different_conceders"],
            "clarity": (v.get("judge_clarity") or {}).get(abl),
            "accept": (v.get("judge_acceptability") or {}).get(abl)}, v["isolates"], tag=tag))
    leaks = {r["tag"]: r["cells"]["leaks"] for r in rows}
    return [_table(
        "ablations", "flask", "Ablations: our system minus one part",
        "Which parts matter? Each ablation removes one part and meets the same scenarios as our system.",
        f"Grounded explanations are what keep private reasons private: without them {leaks.get('A2', '?')} reached "
        "a colleague, against 0. Removing the other parts resolves just as many conflicts here (the notes say why).",
        "Stratified 30 scenarios, seed 0, paired with our system's run on the same scenarios.", at, "leaks",
        [_col("leaks", "Private reasons leaked", "int", "low", "Private reasons that reached a colleague"),
         _col("resolution", "Conflicts resolved", "pct", "high"),
         _col("first", "First offer taken", "pct", "high"),
         _col("changed", "Changed who gave way", "text", None, "Resolved conflicts where someone else conceded",
              key_col=False),
         _col("clarity", "Clarity (judge, 1-5)", "num", "high", "LLM judge; not yet validated against people",
              key_col=False),
         _col("accept", "Acceptability (judge, 1-5)", "num", "high", "LLM judge; not yet validated against people",
              key_col=False)],
        rows,
        ["Without grounded explanations (A2): a private reason leaked in every scenario that had one to leak (4 of 4).",
         "Without tiers or ledger (A1): the benchmark's conflicts rarely meet the same person twice, so it never "
         "changed who gave way; fairness is tested in its own experiment.",
         "Without LLM reply reading (A3): the replies here are almost all plain yes or no. On the 100-reply benchmark, "
         "which has counter-offers and vague replies, keyword rules get 66% of actions right against 100% for the "
         "fine-tuned 4B (Models)."])]


# -- fairness stress test ---------------------------------------------------------------------------------------

FAIRNESS_NAMES = {
    "ours": ("Our system", "with the concession ledger", True, ""),
    "no-ledger": ("Same policy, no ledger", "isolates the ledger", False, ""),
    "flat": ("Flat weights", "every option costs the same", False, "A1"),
    "ours-smooth": ("Smooth credit", "experimental, not in the live system", False, ""),
}


def fairness(runs: Path) -> list[dict]:
    rep, at = _load(runs / "fairness-final.json")
    if not rep:
        return []
    rows = []
    for k, s in rep["summary"].items():
        name, sub, hi, tag = FAIRNESS_NAMES.get(k, (k, "", False, ""))
        t = s["times"]
        rows.append(_row(name, {"gini": s["gini"], "max": s["max"], "most": max(t.values()), "least": min(t.values()),
                                "resolved": s["resolution_rate"], "students": s.get("students_per_conflict")},
                         sub, hi, tag))
    pv = rep.get("paired", {})

    def p(key, m):
        return pv.get(key, {}).get(m, {}).get("p")

    s = rep["summary"]
    return [_table(
        "fairness", "scale", "Fairness: who gives way over a semester",
        "Without memory, the same cheap-to-move teachers are asked again and again. Does the concession ledger spread "
        "the burden?",
        f"Yes. The ledger halves the unevenness (Gini {s['ours']['gini']:.2f} against {s['no-ledger']['gini']:.2f} "
        f"without it, {_p(p('ours vs no-ledger', 'gini'))}) and every conflict is still resolved.",
        f"One department, the same four teachers, {rep['conflicts']} lab clashes per semester, {rep['seeds']} "
        "semesters; two teachers have the smallest section, so they are always the cheapest to move.", at, "gini",
        [_col("gini", "Unevenness (Gini)", "num", "low", "0 = everyone gives way equally, 1 = one person does"),
         _col("most", "Most times one teacher gave way", "num", "low", "Mean over semesters, of 24 conflicts"),
         _col("least", "Fewest times", "num", "high", "Mean over semesters"),
         _col("resolved", "Conflicts resolved", "pct", "high"),
         _col("students", "Students affected per conflict", "num", "low",
              "Students in the section of the teacher who gave way", key_col=False),
         _col("max", "Largest weighted burden", "num", "low", key_col=False)],
        rows,
        [f"Flat weights are about as even (p = {p('ours vs flat', 'gini'):.2f}), but only because they ignore who is "
         f"affected: they move bigger sections ({s['flat']['students_per_conflict']:.1f} students per conflict against "
         f"{s['ours']['students_per_conflict']:.1f}, {_p(p('ours vs flat', 'students_per_conflict'))}).",
         f"Smooth credit would spread it further ({_p(p('ours-smooth vs ours', 'gini'))}): the live ledger's credit "
         "stops growing after about one concession.",
         "On the 60 benchmark scenarios flat weights changed who gave way in 1 of 22 conflicts: those conflicts "
         "rarely meet the same person twice, which is why this experiment exists."])]


# -- RAG --------------------------------------------------------------------------------------------------------


def rag(runs: Path) -> list[dict]:
    files = [("rag-4b-read100-k8.json", 8), ("rag-4b-read100.json", 5)]
    rows, at, by = [], None, {}
    for name, k in files:
        rep, a = _load(runs / name)
        if not rep:
            continue
        at = max(x for x in (at, a) if x)
        for cond, block in rep["conditions"].items():
            if cond == "no-rag" and "none" in by:
                continue
            s = block["summary"]
            by["none" if cond == "no-rag" else k] = s
            rows.append(_row("No retrieval" if cond == "no-rag" else f"Retrieval, {k} rules shown",
                             {"accuracy": s["answer_accuracy"], "retrieval": s["retrieval_recall"],
                              "abstain": s["abstention_on_unanswerable"],
                              "hallucinate": s["hallucination_on_unanswerable"],
                              "cite_precision": s["citation_precision"], "false_abstain": s["false_abstention"]},
                             "the model alone" if cond == "no-rag" else "as deployed" if k == 8 else "the earlier setting",
                             cond == "rag" and k == 8))
    rows.sort(key=lambda r: r["label"] == "No retrieval")
    out = []
    if rows:
        best, none = by.get(8) or by.get(5), by.get("none") or {}
        out.append(_table(
            "rag", "book", "Policy questions: retrieval (RAG)",
            "Does showing the model the institute's rules make its answers right, and does it admit when no rule "
            "applies?",
            f"Yes. With retrieval {_pct(best['answer_accuracy'])} of answers are right against "
            f"{_pct(none.get('answer_accuracy'))} without, and it never invents a rule (without retrieval it did "
            f"{_pct(none.get('hallucination_on_unanswerable'))} of the time).",
            "40 questions worded unlike the rules and 20 that no rule covers; policy agent on Qwen3.5-4B + LoRA; "
            "answers labelled by a Gemini judge.", at, "accuracy",
            [_col("accuracy", "Right answers", "pct", "high", "Answered, and consistent with the rule"),
             _col("abstain", "Says 'no rule' when none applies", "pct", "high", "On the 20 questions no rule covers"),
             _col("hallucinate", "Invents a rule", "pct", "low", "A substantive answer when no rule exists"),
             _col("retrieval", "Right rule shown", "pct", "high", "The rule among those retrieved", key_col=False),
             _col("cite_precision", "Citation precision", "pct", "high", key_col=False),
             _col("false_abstain", "Wrongly says 'no rule'", "pct", "low", key_col=False)],
            rows,
            ["Without retrieval the model makes up institute policy (\"10 days of casual leave\", \"75% attendance\").",
             "Most remaining misses had the right rule in front of the model, which cited a similar one beside it.",
             "Ordinary requests (109, policy test split): every verdict is right with 5 or 8 rules shown."]))
    rk, at = _load(runs / "retrieval-k.json")
    if rk:
        rec = rk["recall"]
        ks = sorted({int(x.split("@")[1]) for x in rec})
        rows = [_row({"bm25": "Keyword (BM25)", "dense": "Meaning (embeddings)", "hybrid": "Both combined"}[kind],
                     {f"k{k}": rec.get(f"{kind}@{k}") for k in ks}, "deployed" if kind == "hybrid" else "",
                     kind == "hybrid")
                for kind in ("bm25", "dense", "hybrid")]
        n = rk["queries"]
        out.append(_table(
            "retrieval", "search", "Retrieval: how many rules to show",
            "Is the right rule among those shown to the model, and how many should it see?",
            f"Combined retrieval with 8 rules finds the right one for {round(rec['hybrid@8'] * n)} of {n} questions, "
            f"against {round(rec['hybrid@5'] * n)} with 5; the deployed setting is now 8.",
            f"The same {n} paraphrased questions, no model calls.", at, "k8",
            [_col(f"k{k}", f"{k} rules shown", "pct", "high") for k in ks], rows))
    return out


# -- safety -----------------------------------------------------------------------------------------------------

SAFETY_RUNS = [
    ("safety-4b.json", "Checks on, fine-tuned 4B", "the system as deployed", True),
    ("safety-compromised.json", "Checks on, compromised model", "a model that obeys every instruction", False),
    ("safety-4b-noguards.json", "Checks off, fine-tuned 4B", "what the model resists on its own", False),
    ("safety-compromised-noguards.json", "Checks off, compromised model", "nothing stands", False),
    ("safety-gemini.json", "Checks on, Gemini Flash-Lite", "the earlier run with the prompted parser", False),
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
    by = {r["label"]: r["cells"] for r in rows}
    worst = by.get("Checks off, compromised model", {})
    return [_table(
        "safety", "shield", "Safety: 50 attacks",
        "Do the code checks (sender, authority, injection filter, human approval) stop attacks, even if the model is "
        "fooled?",
        "Yes. With the checks on nothing is published without approval, even with a model that obeys every "
        f"instruction; with them off, {worst.get('unapproved', '?')} attacks get published.",
        "Students and colleagues changing someone else's classes, claimed authority, \"I approve\" emails, prompt "
        "injections, questions about a colleague's private reason, spoofed senders.", at, "pass",
        [_col("pass", "Handled safely", "pct", "high"),
         _col("unapproved", "Published without approval", "int", "low"),
         _col("authority", "Authority violations", "int", "low"),
         _col("injection", "Injections obeyed", "int", "low", key_col=False),
         _col("leaks", "Private reasons leaked", "int", "low", key_col=False)],
        rows,
        ["Checks off means, for this test only: no sender verification, no injection filter, no authority check, and "
         "whatever reaches approval is approved at once.",
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
    systems = {"intents": "Fixed pipeline", "agent": "Agent loop"}
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
            "intents adapter" if sys_ == "intents" else "agent adapter, tool calls" + (
                "; five requests rerun after the absence fix" if fixed else ""), sys_ == "agent"))
    t = {r["label"]: r["cells"] for r in rows}
    pipe, agent = t.get("Fixed pipeline", {}), t.get("Agent loop", {})
    return [_table(
        "head-to-head", "swords", "Pipeline vs agent",
        "The same 41 requests, the same department and week: does the tool-calling agent do as well as the fixed "
        "pipeline?",
        f"As well, but slower: both pass {agent.get('passed', '?')}; the agent takes "
        f"{(agent.get('median') or 0):.0f} s at the median against {(pipe.get('median') or 0):.0f} s.",
        "Reset before each request.", at, "pass",
        [_col("pass", "Requests passed", "pct", "high"),
         _col("median", "Median seconds", "num", "low"),
         _col("action", "Right action", "pct", "high", "Changed, answered, asked or refused as it should",
              key_col=False),
         _col("checks", "Timetable checks hold", "text", None, "For requests that change the timetable",
              key_col=False)],
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
        rows.append(_row(f"Agent {name}", {"rate": k / n, "passed": f"{k}/{n}",
                                           "ci": f"{100 * float(m[3]):.0f}-{100 * float(m[4]):.0f}%"},
                         {"v6-dpo": "deployed", "v7-sft": "new data, SFT", "v7-dpo": "new data, SFT + DPO"}.get(name, ""),
                         name == "v6-dpo"))
    if not rows:
        return []
    at = datetime.fromtimestamp(doc.stat().st_mtime, UTC).isoformat()
    return [_table(
        "agent-adapters", "cpu", "Agent adapters",
        "Did retraining the agent (v7) beat the deployed adapter (v6)?",
        "No: v7 scored lower, so the deployed v6 stays (the release rule).",
        "49 request families x 2 departments x 5 episodes on Kaggle.", at, "rate",
        [_col("rate", "Episodes passed", "pct", "high"), _col("passed", "Count", "text"),
         _col("ci", "95% interval", "text", key_col=False)], rows,
        ["v7's losses were on counter-offers, no reply and fallbacks."])]


# -- the answers first --------------------------------------------------------------------------------------------


def findings(runs: Path) -> list[dict]:
    """One line per hypothesis or capability: the claim, a verdict and the numbers, read from the result files."""
    out = []

    def add(claim, verdict, evidence, section):
        out.append({"claim": claim, "verdict": verdict, "evidence": evidence, "section": section})

    ours, _ = _load(runs / "negotiation-main-ours-4b.json")
    base, _ = _load(runs / "negotiation-main-baselines-gemini.json")
    paired, _ = _load(runs / "negotiation-main-paired.json")
    if ours and base:
        o, b3, b4 = ours["configs"]["ours-llm"]["summary"], base["configs"]["B3"]["summary"], base["configs"]["B4"]["summary"]
        add("Options invented by an LLM cost more than options from the solver", "supported",
            f"{_pct(b3['invalid_candidate_rate'])} of unchecked invented options are impossible; filtering them takes "
            f"{b4['option_calls_per_scenario']:.1f} LLM calls per conflict (ours: none)", "negotiation-main")
        pv = (paired or {}).get("ours-llm vs B4", {})
        add("Our system beats the main baseline (LLM proposes, solver filters)", "partly",
            f"resolves {_pct(o['agreement_rate'])} vs {_pct(b4['agreement_rate'])} of conflicts "
            f"({_p(pv.get('resolved_mcnemar', {}).get('p'))}) and lands much closer to the oracle "
            f"({o['objective_gap_abs_mean']:.2f} vs {b4['objective_gap_abs_mean']:.2f}); not in fewer rounds "
            f"({o['rounds_mean']:.2f} vs {b4['rounds_mean']:.2f}, {_p(pv.get('rounds_wilcoxon', {}).get('p'))})",
            "negotiation-main")
    bias, _ = _load(runs / "negotiation-bias-gemini.json")
    if bias:
        s = {k: v["summary"] for k, v in bias["configs"].items()}
        add("The gap comes from the solver's options, not from the model", "supported",
            f"both on Gemini: ours resolves {_pct(s['ours-llm']['agreement_rate'])}, the baseline "
            f"{_pct(s['B4']['agreement_rate'])}; ours resolves the same conflicts on Gemini as on the 4B", "bias")
    abl, _ = _load(runs / "negotiation-ablations-paired.json")
    if abl and "ours-llm vs A2" in abl:
        v = abl["ours-llm vs A2"]
        add("Grounded explanations keep private reasons private, without lowering acceptance", "supported",
            f"{round(v['leaks']['A2'] * v['pairs'])} private reasons leaked without grounding, 0 with; first offer taken "
            f"{_pct(v['first_proposal_acceptance']['ours-llm'])} vs {_pct(v['first_proposal_acceptance']['A2'])}",
            "ablations")
    fair, _ = _load(runs / "fairness-final.json")
    if fair:
        s, pv = fair["summary"], fair.get("paired", {})
        add("The concession ledger spreads the burden of giving way", "supported",
            f"unevenness {s['ours']['gini']:.2f} with the ledger vs {s['no-ledger']['gini']:.2f} without "
            f"({_p(pv.get('ours vs no-ledger', {}).get('gini', {}).get('p'))}), no conflicts lost; flat weights are as "
            "even only by ignoring who is affected", "fairness")
    par, _ = _load(runs / "parsing-4b-nts-q8-paratest.json")
    pro, _ = _load(runs / "parsing-4b-base-prompted-paratest.json")
    if par and pro:
        add("Fine-tuning makes the small model read requests reliably", "supported",
            f"the 4B compiles {_pct(par['system_two']['compile_exact_match'])} of requests exactly with LoRA, "
            f"{_pct(pro['system_two']['compile_exact_match'])} prompted", "models")
    r8, _ = _load(runs / "rag-4b-read100-k8.json")
    if r8:
        s, n = r8["conditions"]["rag"]["summary"], r8["conditions"]["no-rag"]["summary"]
        add("Retrieval grounds policy answers and stops invented rules", "supported",
            f"{_pct(s['answer_accuracy'])} right with retrieval vs {_pct(n['answer_accuracy'])} without; invents a rule "
            f"{_pct(s['hallucination_on_unanswerable'])} vs {_pct(n['hallucination_on_unanswerable'])}", "rag")
    rep, _ = _load(runs / "replies-4b-nts-q8.json")
    rule, _ = _load(runs / "replies-rule.json")
    if rep and rule:
        add("The fine-tuned model turns replies into the right tool call", "supported",
            f"{_pct(rep['summary']['tool_accuracy'])} of 100 replies vs {_pct(rule['summary']['tool_accuracy'])} for "
            "keyword rules", "models")
    on, _ = _load(runs / "safety-4b.json")
    off, _ = _load(runs / "safety-compromised-noguards.json")
    if on and off:
        add("Safety comes from code checks, not from trusting the model", "supported",
            f"{_pct(on['pass_rate'])} of 50 attacks handled safely as deployed; {off['unapproved_publishes']} get "
            "published with the checks off and a compromised model", "safety")
    if not out:
        return []
    return [{"id": "findings", "kind": "findings", "title": "Key findings",
             "at": datetime.now(UTC).isoformat(), "items": out,
             "pending": ["The LLM judge's scores are not yet validated against people (two raters in Study).",
                         "No human pilot yet."]}]


def sections(root: Path, agentic: bool = False) -> list[dict]:
    """Every section after the main run and the model table; ``agentic``: also the agent system's own
    (pipeline vs agent, the agent adapters). The page puts them in its own order."""
    runs = root / "runs"
    out = [*findings(runs), *fairness(runs), *ablations(runs), *bias_check(runs), *rag(runs), *safety(runs)]
    if agentic:
        out += [*head_to_head(runs), *agent_adapters(root)]
    return out
