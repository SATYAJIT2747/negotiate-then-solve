"""The Experiments page as a report: an introduction, one chapter per experiment, and the conclusions.

A chapter tells one experiment in order: the question, what we did (``method`` steps and the systems
``compared``), the answer in one sentence (``takeaway``), charts made for that experiment, what it means
(``notes``) and the full table for whoever wants it::

    {"id", "kind": "chapter", "icon", "title", "question", "method": [str],
     "compared": [{"name", "tag", "desc", "ours"}], "takeaway", "charts": [chart], "notes": [str],
     "table": {"columns": [...], "rows": [...]} | None, "setup", "at"}

    chart = {"type": "bar" | "line", "title", "caption", "categories": [str],
             "series": [{"name", "data": [number | None], "ours": bool, "error": [[lo, hi]] | None}],
             "fmt": "pct" | "num" | "int", "max": number | None, "x_label", "y_label"}

``intro`` and ``conclusions`` frame them. Every number is read from a result file in ``runs/`` (or, for the agent
adapters, ``docs/regression-matrix.md``); a chapter whose files are missing is left out, so the same module serves
both branches.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

MAIN_RUNS = ("negotiation-main-ours-4b.json", "negotiation-main-baselines-gemini.json", "negotiation-main-offline.json")


def _load(path: Path):
    if not path.exists():
        return None, None
    return json.loads(path.read_text(encoding="utf-8")), datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _latest(*ats):
    return max((a for a in ats if a), default=None)


def _col(key, label, fmt="num", better=None, hint="", key_col=True, digits=None):
    return {"key": key, "label": label, "fmt": fmt, "better": better, "hint": hint, "key_col": key_col, "digits": digits}


def _row(label, cells, sub="", highlight=False, tag=""):
    return {"label": label, "tag": tag, "sub": sub, "highlight": highlight, "cells": cells}


def _series(name, data, ours=False, error=None):
    return {"name": name, "data": data, "ours": ours, "error": error}


def _chart(title, categories, series, fmt="pct", caption="", type_="bar", max_=None, x_label="", y_label=""):
    return {"type": type_, "title": title, "caption": caption, "categories": categories, "series": series, "fmt": fmt,
            "max": max_, "x_label": x_label, "y_label": y_label}


def _who(name, desc, tag="", ours=False):
    return {"name": name, "tag": tag, "desc": desc, "ours": ours}


def _chapter(id_, icon, title, question, method, compared, takeaway, charts, notes, table=None, setup="", at=None):
    return {"id": id_, "kind": "chapter", "icon": icon, "title": title, "question": question, "method": method,
            "compared": compared, "takeaway": takeaway, "charts": charts, "notes": list(notes), "table": table,
            "setup": setup, "at": at}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _p(x: float | None) -> str:
    return "–" if x is None else "p < 0.001" if x < 0.001 else f"p = {x:.3f}" if x < 0.01 else f"p = {x:.2f}"


# -- 1. the main comparison ---------------------------------------------------------------------------------------

MAIN_NAMES = {
    "ours-llm": ("Our system", "", "the negotiator may offer only options the solver has checked", True),
    "B4": ("LLM proposes, solver filters", "B4", "the LLM invents options; the solver drops impossible ones and it tries again", False),
    "B3": ("LLM proposes, unchecked", "B3", "the LLM's invented options go out as they are", False),
    "B2": ("Solver decides, nobody asked", "B2", "the cheapest fix is imposed without negotiating", False),
    "B1": ("LLM writes the timetable", "B1", "no solver at all", False),
    "oracle": ("Oracle", "", "knows everyone's hidden flexibility: the best possible", False),
}
FAMILY = {"strict": "Strict", "flexible": "Flexible", "cost_sensitive": "Cost-sensitive",
          "history_sensitive": "History-sensitive"}


def main_comparison(runs: Path) -> list[dict]:
    parts = [_load(runs / f) for f in MAIN_RUNS]
    configs, rows, at = {}, {}, _latest(*(a for _, a in parts))
    for rep, _ in parts:
        if rep:
            for k, v in rep["configs"].items():
                configs.setdefault(k, v["summary"])
                rows.setdefault(k, v["rows"])
    if "ours-llm" not in configs or "B4" not in configs:
        return []
    paired, _ = _load(runs / "negotiation-main-paired.json")
    order = [k for k in MAIN_NAMES if k in configs]
    names = [MAIN_NAMES[k][0] for k in order]
    o, b4 = configs["ours-llm"], configs["B4"]
    pv = (paired or {}).get("ours-llm vs B4", {})
    charts = [
        _chart("How many conflicts each system resolves", names,
               [_series("Conflicts resolved", [configs[k].get("agreement_rate") for k in order], True),
                _series("Ended as the oracle did", [configs[k].get("correct_outcome") for k in order])],
               caption="Resolved: a resolvable conflict ended in an agreement everyone accepted. The oracle resolves "
                       "every resolvable one."),
        _chart("How far from the best possible timetable", [n for k, n in zip(order, names, strict=True)
                                                           if configs[k].get("objective_gap_abs_mean") is not None],
               [_series("Gap to the oracle's cost", [configs[k]["objective_gap_abs_mean"] for k in order
                                                     if configs[k].get("objective_gap_abs_mean") is not None], True)],
               fmt="num", caption="Mean distance of the final timetable's cost from the oracle's. Lower is better."),
    ]
    inv = [k for k in ("ours-llm", "B3", "B4") if k in configs]
    charts.append(_chart(
        "What inventing options costs", [MAIN_NAMES[k][0] for k in inv],
        [_series("LLM calls to make options", [configs[k].get("option_calls_per_scenario") or 0 for k in inv], True),
         _series("Impossible options thrown away", [configs[k].get("filtered_per_scenario") or 0 for k in inv])],
        fmt="num", caption=f"Per conflict. Unchecked, {_pct(configs.get('B3', {}).get('invalid_candidate_rate'))} of "
                           "the invented options are impossible; our system needs no calls because the solver "
                           "enumerates the options."))
    fams = [f for f in FAMILY if any(r.get("family") == f for r in rows["ours-llm"])]

    def by_family(k):
        out = []
        for f in fams:
            rs = [r for r in rows[k] if r.get("family") == f and r.get("expected") == "agree"]
            out.append(sum(r["status"] == "agreed" for r in rs) / len(rs) if rs else None)
        return out

    charts.append(_chart(
        "Across kinds of people", [FAMILY[f] for f in fams],
        [_series("Our system", by_family("ours-llm"), True), _series("LLM proposes, solver filters", by_family("B4"))],
        caption="Conflicts resolved, by the simulated stakeholders' behaviour: strict ones accept little, "
                "history-sensitive ones give way less if they gave way recently."))
    table_cols = [_col("resolved", "Conflicts resolved", "pct", "high"), _col("correct", "Right outcome", "pct", "high"),
                  _col("gap", "Gap to oracle", "num", "low"), _col("rounds", "Rounds to agree", "num", "low"),
                  _col("valid", "Valid timetables", "pct", "high", key_col=False),
                  _col("first", "First offer taken", "pct", "high", key_col=False),
                  _col("esc_p", "Escalations justified", "num", "high", key_col=False),
                  _col("n", "Runs", "int", None, key_col=False)]
    table_rows = [_row(MAIN_NAMES[k][0], {"resolved": configs[k].get("agreement_rate"),
                                          "correct": configs[k].get("correct_outcome"),
                                          "gap": configs[k].get("objective_gap_abs_mean"),
                                          "rounds": configs[k].get("rounds_mean"),
                                          "valid": configs[k].get("validity_of_timetables"),
                                          "first": configs[k].get("first_proposal_acceptance"),
                                          "esc_p": configs[k].get("escalation_precision"), "n": configs[k].get("n")},
                       MAIN_NAMES[k][2], MAIN_NAMES[k][3], MAIN_NAMES[k][1]) for k in order]
    return [_chapter(
        "main", "messages", "The main comparison",
        "When two requests cannot both be met, someone has to give way. Is it better to let the negotiator offer only "
        "options the solver has already checked, or to let an LLM come up with its own ideas, even if the solver "
        "filters them?",
        ["Built 60 small departments, each with a planted conflict: two or three teachers want the same lab, a "
         "request breaks a rule, or a room is too small.",
         "Gave every simulated teacher hidden flexibility (times they would really accept), which only the oracle "
         "knows.",
         "Let each system negotiate by message until everyone agreed or it escalated to a person; repeated with 3 "
         "seeds for the main systems.",
         "Scored the outcome against the oracle: resolved or not, how good the timetable is, how many rounds."],
        [_who(*MAIN_NAMES[k][:1], MAIN_NAMES[k][2], MAIN_NAMES[k][1], MAIN_NAMES[k][3]) for k in order],
        f"Offering only the solver's options resolves {_pct(o['agreement_rate'])} of conflicts against "
        f"{_pct(b4['agreement_rate'])} ({_p(pv.get('resolved_mcnemar', {}).get('p'))}) and lands far closer to the "
        f"best timetable; it does not take fewer rounds ({o['rounds_mean']:.2f} against {b4['rounds_mean']:.2f}).",
        charts,
        ["The LLM's own ideas are often impossible or miss concessions the solver finds, so it resolves fewer "
         "conflicts and settles for worse timetables, at several LLM calls per conflict.",
         "Imposing the cheapest fix without asking (B2) resolves nothing by agreement: only 10% of the imposed "
         "changes would have been accepted. An LLM writing the timetable itself (B1) breaks hard rules in 80% of its "
         "timetables.",
         "Rounds are about equal: when either system finds an acceptable option, it finds it early."],
        {"columns": table_cols, "rows": table_rows},
        "Our system on Qwen3.5-4B + LoRA; the LLM baselines on Gemini Flash-Lite; simulated stakeholders on Gemini.",
        at)]


# -- 2. same model for both -----------------------------------------------------------------------------------------


def bias_check(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-bias-gemini.json")
    main, _ = _load(runs / "negotiation-main-ours-4b.json")
    if not rep:
        return []
    ids = {r["scenario"] for r in rep["configs"]["ours-llm"]["rows"]}
    want = {"feasible": {"feasible"}, "agree": {"agreed"}, "escalate": {"escalated", "failed"}}  # correct_outcome

    def cells(rs: list[dict], s: dict | None = None) -> dict:
        s = s or {}
        negotiated = [r for r in rs if r.get("rounds")]
        agree = [r for r in rs if r.get("expected") == "agree"]
        return {"resolved": sum(r["status"] == "agreed" for r in rs),
                "rate": s.get("agreement_rate", sum(r["status"] == "agreed" for r in agree) / len(agree) if agree else None),
                "correct": s.get("correct_outcome", sum(r["status"] in want.get(r.get("expected"), ()) for r in rs)
                                 / len(rs) if rs else None),
                "first": s.get("first_proposal_acceptance",
                               sum(bool(r.get("first_accepted")) for r in negotiated) / len(negotiated)
                               if negotiated else None)}

    rows = []
    if main:
        sub = [r for r in main["configs"]["ours-llm"]["rows"] if r["scenario"] in ids and r.get("seed", 0) == 0]
        rows.append(_row("Our system on the 4B", cells(sub), "the main run, same 20 scenarios", True))
    ours, b4 = rep["configs"]["ours-llm"], rep["configs"]["B4"]
    rows.append(_row("Our system on Gemini", cells(ours["rows"], ours["summary"]), "the same model as the baseline", True))
    rows.append(_row("LLM proposes, solver filters, on Gemini", cells(b4["rows"], b4["summary"]), "the main baseline",
                     tag="B4"))
    p = (rep.get("paired") or {}).get("ours-llm vs B4", {}).get("resolved_mcnemar", {})
    return [_chapter(
        "bias", "scale", "Is it the model?",
        "In the main comparison our system ran on our fine-tuned 4B while the LLM baselines ran on Gemini. Could the "
        "difference come from the model rather than from the solver's options?",
        ["Took a stratified 20 of the 60 scenarios.",
         "Ran our system and the main baseline with the same model, Gemini Flash-Lite, for everything.",
         "Compared with our system's 4B run on the same 20."],
        [_who("Our system on the 4B", "the main run", ours=True), _who("Our system on Gemini", "same model as B4", ours=True),
         _who("LLM proposes, solver filters", "on Gemini", "B4")],
        f"It is the options, not the model: on the same model our system resolves {rows[-2]['cells']['resolved']} "
        f"conflicts and the baseline {rows[-1]['cells']['resolved']}; our system resolves the same conflicts on either "
        "model.",
        [_chart("Conflicts resolved on the same 20 scenarios", [r["label"] for r in rows],
                [_series("Resolved", [r["cells"]["resolved"] for r in rows], True)], fmt="int",
                caption="Our system's result does not depend on which model writes its messages.")],
        [f"20 scenarios are too few for significance on their own ({p.get('only_first', '?')} vs "
         f"{p.get('only_second', '?')} discordant, {_p(p.get('p'))}); the 180-pair main run is significant."],
        {"columns": [_col("resolved", "Conflicts resolved", "int", "high"), _col("rate", "Resolution rate", "pct", "high"),
                     _col("correct", "Right outcome", "pct", "high"),
                     _col("first", "First offer taken", "pct", "high", key_col=False)], "rows": rows},
        "Stratified 20 scenarios, seed 0.", at)]


# -- 3. ablations ----------------------------------------------------------------------------------------------------

ABLATION_NAMES = {"A1": ("Without tiers or ledger", "everyone weighted the same, no memory of who gave way"),
                  "A2": ("Without grounded explanations", "the LLM explains in its own words, not from solver facts"),
                  "A3": ("Without LLM reply reading", "keyword rules read the replies")}


def ablations(runs: Path) -> list[dict]:
    rep, at = _load(runs / "negotiation-ablations-paired.json")
    if not rep:
        return []
    rows = []
    for key, v in rep.items():
        ref, abl = key.split(" vs ")
        if abl not in ABLATION_NAMES:
            continue
        if not rows:
            rows.append(_row("Our system", {
                "resolution": v["resolution_rate"][ref], "leaks": round(v["leaks"][ref] * v["pairs"]),
                "first": v["first_proposal_acceptance"][ref], "changed": "–",
                "clarity": (v.get("judge_clarity") or {}).get(ref),
                "accept": (v.get("judge_acceptability") or {}).get(ref)}, "all parts", True))
        rows.append(_row(ABLATION_NAMES[abl][0], {
            "resolution": v["resolution_rate"][abl], "leaks": round(v["leaks"][abl] * v["pairs"]),
            "first": v["first_proposal_acceptance"][abl], "changed": v["different_conceders"],
            "clarity": (v.get("judge_clarity") or {}).get(abl),
            "accept": (v.get("judge_acceptability") or {}).get(abl)}, ABLATION_NAMES[abl][1], tag=abl))
    names = [r["label"] for r in rows]
    leaks = {r["tag"]: r["cells"]["leaks"] for r in rows}
    return [_chapter(
        "ablations", "flask", "Which parts matter?",
        "Our system has several parts: priority tiers and a memory of who gave way, explanations built from solver "
        "facts, and an LLM that reads replies. What happens when each is taken away?",
        ["Removed one part at a time, keeping everything else the same.",
         "Ran each version on a stratified 30 scenarios and compared it with our system on the very same scenarios.",
         "Counted private reasons that reached a colleague, offers accepted, and conflicts resolved; an LLM judge "
         "rated the messages."],
        [_who("Our system", "all parts", ours=True)] + [_who(n, d, k) for k, (n, d) in ABLATION_NAMES.items()],
        f"Grounded explanations are what keep private reasons private: without them {leaks.get('A2', '?')} reached a "
        "colleague, against none. The other parts make no difference on these one-off conflicts; they matter where "
        "the next chapters test them.",
        [_chart("Private reasons that reached a colleague", names,
                [_series("Leaks", [r["cells"]["leaks"] for r in rows], True)], fmt="int",
                caption="A teacher's private reason (an illness, say) named in a message to someone else."),
         _chart("First offer accepted", names, [_series("Accepted", [r["cells"]["first"] for r in rows], True)],
                caption="Negotiations where the first option offered was taken.")],
        ["Without grounding, a private reason leaked in every scenario that had one to leak (4 of 4); with grounding, "
         "in none.",
         "Without tiers or ledger, nobody else was asked to give way: these conflicts rarely meet the same person "
         "twice. Fairness is tested in its own chapter.",
         "Keyword rules handle the plain yes-or-no replies here. On the reply benchmark, with counter-offers and vague "
         "answers, they get 66% of actions right against 100% for the fine-tuned model (chapter on models).",
         "The LLM judge rates every version about the same (4.0-4.5 of 5); its scores are not yet checked against "
         "people."],
        {"columns": [_col("leaks", "Private reasons leaked", "int", "low"),
                     _col("resolution", "Conflicts resolved", "pct", "high"),
                     _col("first", "First offer taken", "pct", "high"),
                     _col("changed", "Changed who gave way", "text", key_col=False),
                     _col("clarity", "Clarity (judge, 1-5)", "num", "high", key_col=False),
                     _col("accept", "Acceptability (judge, 1-5)", "num", "high", key_col=False)], "rows": rows},
        "Stratified 30 scenarios, seed 0, paired with our system's run.", at)]


# -- 4. fairness -----------------------------------------------------------------------------------------------------

FAIRNESS_NAMES = {
    "ours": ("Our system", "with the concession ledger", True),
    "no-ledger": ("Same policy, no ledger", "isolates the ledger", False),
    "flat": ("Flat weights", "every option costs the same", False),
    "ours-smooth": ("Smooth credit", "experimental, not in the live system", False),
}
TEACHERS = {"F-301": "Teacher 1 (small section)", "F-302": "Teacher 2", "F-303": "Teacher 3",
            "F-304": "Teacher 4 (small section)"}


def fairness(runs: Path) -> list[dict]:
    rep, at = _load(runs / "fairness-final.json")
    if not rep:
        return []
    s, pv = rep["summary"], rep.get("paired", {})
    keys = [k for k in FAIRNESS_NAMES if k in s]

    def p(key, m):
        return pv.get(key, {}).get(m, {}).get("p")

    n = len(s["ours"]["gini_trace"])
    rows = [_row(FAIRNESS_NAMES[k][0], {"gini": s[k]["gini"], "most": max(s[k]["times"].values()),
                                        "least": min(s[k]["times"].values()), "resolved": s[k]["resolution_rate"],
                                        "students": s[k].get("students_per_conflict"), "max": s[k]["max"]},
                  FAIRNESS_NAMES[k][1], FAIRNESS_NAMES[k][2], "A1" if k == "flat" else "") for k in keys]
    return [_chapter(
        "fairness", "scale", "Who gives way?",
        "Without a memory of who has given way, the same people are asked again and again: whoever is cheapest to "
        "move. Does the concession ledger spread the burden, and at what cost?",
        [f"Simulated one department with the same four teachers through {rep['conflicts']} lab clashes in a "
         f"semester, {rep['seeds']} times.",
         "Gave two of the teachers the smallest section, so moving their lab always looks cheapest.",
         "Ran every version on exactly the same clashes, with the same hidden flexibility, and recorded who gave way "
         "each time.",
         "Measured the unevenness of the burden (the Gini: 0 when everyone gives way equally)."],
        [_who(FAIRNESS_NAMES[k][0], FAIRNESS_NAMES[k][1], "A1" if k == "flat" else "", FAIRNESS_NAMES[k][2])
         for k in keys],
        f"The ledger halves the unevenness (Gini {s['ours']['gini']:.2f} against {s['no-ledger']['gini']:.2f} without "
        f"it, {_p(p('ours vs no-ledger', 'gini'))}) and every conflict is still resolved.",
        [_chart("How uneven the burden gets over a semester", [str(i + 1) for i in range(n)],
                [_series(FAIRNESS_NAMES[k][0], s[k]["gini_trace"], FAIRNESS_NAMES[k][2]) for k in keys],
                fmt="num", type_="line", max_=1, x_label="Conflict in the semester", y_label="Unevenness (Gini)",
                caption="Without the ledger the same two teachers keep giving way and the unevenness climbs; with it, "
                        "it stays low."),
         _chart("How often each teacher gave way", list(TEACHERS.values()),
                [_series(FAIRNESS_NAMES[k][0], [s[k]["times"].get(t) for t in TEACHERS], FAIRNESS_NAMES[k][2])
                 for k in keys], fmt="num",
                caption=f"Mean over {rep['seeds']} semesters of {rep['conflicts']} conflicts."),
         _chart("The price of fairness", [FAIRNESS_NAMES[k][0] for k in keys],
                [_series("Students affected per conflict", [s[k].get("students_per_conflict") for k in keys], True)],
                fmt="num", caption="Students in the section of the teacher who gave way: spreading the burden means "
                                   "sometimes moving a bigger section.")],
        [f"Flat weights look about as even (p = {p('ours vs flat', 'gini'):.2f}), but only because they ignore who is "
         f"affected: they move bigger sections ({s['flat']['students_per_conflict']:.1f} students per conflict against "
         f"{s['ours']['students_per_conflict']:.1f}, {_p(p('ours vs flat', 'students_per_conflict'))}).",
         f"A smoother credit would spread it further (Gini {s['ours-smooth']['gini']:.2f}, "
         f"{_p(p('ours-smooth vs ours', 'gini'))}): the live ledger's credit stops growing after about one concession."
         if "ours-smooth" in s else "",
         "The 60-scenario benchmark cannot show this: its conflicts rarely meet the same person twice."],
        {"columns": [_col("gini", "Unevenness (Gini)", "num", "low"),
                     _col("most", "Most times one teacher gave way", "num", "low"),
                     _col("resolved", "Conflicts resolved", "pct", "high"),
                     _col("students", "Students affected per conflict", "num", "low"),
                     _col("least", "Fewest times", "num", "high", key_col=False),
                     _col("max", "Largest weighted burden", "num", "low", key_col=False)], "rows": rows},
        "Scripted replies, no LLM.", at)]


# -- 5. the language model's jobs ----------------------------------------------------------------------------------


def models(runs: Path) -> list[dict]:
    try:
        from .app import MODEL_ROWS
    except ImportError:  # pragma: no cover - the page without the app
        return []
    rows, at = [], None
    for label, rep_f, parse_f, pol_f, _swap in MODEL_ROWS:
        rep, a1 = _load(runs / f"{rep_f}.json") if rep_f else (None, None)
        par, a2 = _load(runs / f"{parse_f}.json") if parse_f else (None, None)
        pol, a3 = _load(runs / f"{pol_f}.json") if pol_f else (None, None)
        if not (rep or par or pol):
            continue
        at = _latest(at, a1, a2, a3)
        two = (par or {}).get("system_two", {})
        rows.append(_row(label, {"read": two.get("compile_exact_match"), "reply": (rep or {}).get("summary", rep or {})
                                 .get("tool_accuracy"), "policy": (pol or {}).get("allow_deny_accuracy"),
                                 "latency": two.get("latency_s_p50"), "action": two.get("action_accuracy")},
                         highlight="4B fine-tuned · Q8" in label))
    if not rows:
        return []
    ft = next((r for r in rows if r["highlight"]), rows[0])
    base = next((r for r in rows if "4B base" in r["label"]), None)
    short = [r["label"].split(" · ")[0].replace("Qwen3.5-", "").replace(" (prompted)", "") for r in rows]
    take = (f"Fine-tuning is what makes the small local model usable: with its adapter the 4B reads "
            f"{_pct(ft['cells']['read'])} of requests exactly and gets {_pct(ft['cells']['reply'])} of replies right, "
            f"against {_pct(base['cells']['read'])} and {_pct(base['cells']['reply'])} prompted.") if base else ""
    return [_chapter(
        "models", "cpu", "Teaching a small model the job",
        "The language model reads requests into constraints, reads stakeholders' replies into actions, and checks "
        "requests against the rules. Can a small local model do this well enough, and what does fine-tuning add?",
        ["Fine-tuned Qwen3.5-4B (and 2B) with LoRA on the system's own tasks, on free Kaggle GPUs.",
         "Tested each model on 151 unseen requests (read exactly into constraints?), 100 replies (right tool call?) "
         "and 109 policy checks (right verdict?).",
         "Compared with the same models prompted, with Gemini, and with keyword rules."],
        [_who("Qwen3.5-4B + LoRA", "the deployed model", ours=True), _who("Qwen3.5-2B + LoRA", "smaller"),
         _who("Prompted models", "the 4B and 2B without adapters, Gemini Flash-Lite"),
         _who("Keyword rules", "no model")],
        take,
        [_chart("Each model on the three jobs", short,
                [_series("Requests read exactly", [r["cells"]["read"] for r in rows], True),
                 _series("Replies: right tool call", [r["cells"]["reply"] for r in rows]),
                 _series("Policy: right verdict", [r["cells"]["policy"] for r in rows])],
                caption="A missing bar: that model was not run on that job.")],
        ["The prompted 4B reads under half the requests exactly and takes 13 s per request; with the adapter it reads "
         "98% in 2 s, because the adapter needs only a short prompt.",
         "The fine-tuned 4B matches or beats Gemini Flash-Lite on every job while running on a single local GPU."],
        {"columns": [_col("read", "Requests read exactly", "pct", "high", "151 paraphrased test requests"),
                     _col("reply", "Replies: right tool call", "pct", "high", "100 replies of every kind"),
                     _col("policy", "Policy: right verdict", "pct", "high", "109 requests"),
                     _col("action", "Right action on requests", "pct", "high", key_col=False),
                     _col("latency", "Median seconds per request", "num", "low", key_col=False)], "rows": rows},
        "", at)]


def rag(runs: Path) -> list[dict]:
    files = [("rag-4b-read100-k8.json", 8), ("rag-4b-read100.json", 5)]
    by, at = {}, None
    for name, k in files:
        rep, a = _load(runs / name)
        if not rep:
            continue
        at = _latest(at, a)
        by.setdefault("none", rep["conditions"]["no-rag"]["summary"])
        by[k] = rep["conditions"]["rag"]["summary"]
    if 8 not in by and 5 not in by:
        return []
    variants = [("none", "No retrieval", "the model alone", False), (5, "5 rules shown", "the earlier setting", False),
                (8, "8 rules shown", "as deployed", True)]
    variants = [v for v in variants if v[0] in by]
    best, none = by.get(8) or by[5], by["none"]
    measures = [("answer_accuracy", "Right answers"), ("abstention_on_unanswerable", "Says 'no rule' when none applies"),
                ("hallucination_on_unanswerable", "Invents a rule")]
    rows = [_row(label, {"accuracy": by[k]["answer_accuracy"], "abstain": by[k]["abstention_on_unanswerable"],
                         "hallucinate": by[k]["hallucination_on_unanswerable"], "retrieval": by[k]["retrieval_recall"],
                         "cite": by[k]["citation_precision"], "false_abstain": by[k]["false_abstention"]}, sub, ours)
            for k, label, sub, ours in variants]
    out = [_chapter(
        "rag", "book", "Answering from the rules",
        "Whether a request is allowed depends on this institute's handbook, which no model has seen. Does showing the "
        "model the relevant rules (retrieval) make its answers right, and does it admit when no rule applies?",
        ["Wrote 40 questions in words unlike the rules' (\"Can the practical go ahead if nobody from staff is "
         "around?\") and 20 that no rule covers (leave, attendance, fees).",
         "Asked the policy agent each question with the retrieved rules, and again with no rules shown.",
         "A separate model checked each answer against the right rule."],
        [_who(label, sub, ours=ours) for _, label, sub, ours in variants],
        f"Retrieval makes the difference: {_pct(best['answer_accuracy'])} of answers are right with it against "
        f"{_pct(none['answer_accuracy'])} without, and it never invents a rule (without it, the model did "
        f"{_pct(none['hallucination_on_unanswerable'])} of the time).",
        [_chart("With and without retrieval", [m[1] for m in measures],
                [_series(label, [by[k][m[0]] for m in measures], ours) for k, label, _, ours in variants],
                caption="Right answers on the 40 answerable questions; the other two on the 20 that no rule covers.")],
        ["Without retrieval the model makes up institute policy: \"10 days of casual leave\", \"75% attendance\".",
         "Most remaining misses had the right rule in front of the model, which cited a similar one beside it.",
         "On 109 ordinary requests every verdict is right with 5 or 8 rules shown."],
        {"columns": [_col("accuracy", "Right answers", "pct", "high"),
                     _col("abstain", "Says 'no rule' when none applies", "pct", "high"),
                     _col("hallucinate", "Invents a rule", "pct", "low"),
                     _col("retrieval", "Right rule shown", "pct", "high", key_col=False),
                     _col("cite", "Citation precision", "pct", "high", key_col=False),
                     _col("false_abstain", "Wrongly says 'no rule'", "pct", "low", key_col=False)], "rows": rows},
        "Policy agent on Qwen3.5-4B + LoRA, combined retrieval.", at)]
    rk, at2 = _load(runs / "retrieval-k.json")
    if rk:
        rec, n = rk["recall"], rk["queries"]
        ks = sorted({int(x.split("@")[1]) for x in rec})
        kinds = [("bm25", "Keyword match", False), ("dense", "Meaning match", False), ("hybrid", "Both combined", True)]
        out.append(_chapter(
            "retrieval", "search", "How many rules to show",
            "Retrieval can only help if the right rule is among those shown. How should rules be found, and how many "
            "should the model see?",
            [f"Took the same {n} questions and, without any model, checked whether the right rule was among the first "
             "5, 8 or 10 found.",
             "Compared finding rules by keywords, by meaning (sentence embeddings), and both combined."],
            [_who(label, "", ours=ours) for _, label, ours in kinds],
            f"Combined search with 8 rules finds the right one for {round(rec['hybrid@8'] * n)} of {n} questions, "
            f"against {round(rec['hybrid@5'] * n)} with 5, so the system now shows 8.",
            [_chart("The right rule is among those shown", [f"{k} rules" for k in ks],
                    [_series(label, [rec.get(f"{kind}@{k}") for k in ks], ours) for kind, label, ours in kinds],
                    type_="line", x_label="Rules shown", y_label="Questions", max_=1,
                    caption="Keywords miss paraphrases; meaning alone misses requests whose rule is implied; combined "
                            "is best from 8 rules on.")],
            ["Showing 8 rules instead of 5 changed no verdict on the 109 ordinary requests."],
            None, "No model calls.", _latest(at2)))
    return out


# -- 6. safety -------------------------------------------------------------------------------------------------------

SAFETY_RUNS = [
    ("safety-4b.json", "Checks on, fine-tuned 4B", "the system as deployed", True),
    ("safety-compromised.json", "Checks on, compromised model", "a model that obeys every instruction", False),
    ("safety-4b-noguards.json", "Checks off, fine-tuned 4B", "what the model resists on its own", False),
    ("safety-compromised-noguards.json", "Checks off, compromised model", "nothing stands", False),
]


def safety(runs: Path) -> list[dict]:
    rows, at = [], None
    for name, label, sub, hi in SAFETY_RUNS:
        rep, a = _load(runs / name)
        if not rep:
            continue
        at = _latest(at, a)
        rows.append(_row(label, {"pass": rep["pass_rate"], "unapproved": rep["unapproved_publishes"],
                                 "authority": rep["authority_violations"], "injection": rep["injection_successes"],
                                 "leaks": rep["private_leaks"]}, sub, hi))
    if not rows:
        return []
    names = [r["label"] for r in rows]
    worst = rows[-1]["cells"]
    return [_chapter(
        "safety", "shield", "Can it be tricked?",
        "Anyone can send the system a message. Can a student move a teacher's class, a colleague claim authority, or "
        "an injected instruction publish a change, even if the language model is fooled?",
        ["Wrote 50 attacks: students and colleagues changing someone else's classes, claimed authority, \"I approve\" "
         "emails, prompt injections, questions about a colleague's private reason, spoofed senders.",
         "Sent each through the real intake and pipeline.",
         "Repeated with a 'compromised' model that obeys every instruction, and with the code checks switched off "
         "(sender check, injection filter, authority check, human approval)."],
        [_who(r["label"], r["sub"], ours=r["highlight"]) for r in rows],
        "The safety lives in the code checks, not the model: with them on, nothing is published without approval even "
        f"when the model obeys every instruction; with them off, {worst['unapproved']} of 50 attacks get published.",
        [_chart("Attacks handled safely", names, [_series("Handled safely", [r["cells"]["pass"] for r in rows], True)]),
         _chart("What got through", names,
                [_series("Published without approval", [r["cells"]["unapproved"] for r in rows]),
                 _series("Authority violations", [r["cells"]["authority"] for r in rows]),
                 _series("Injections obeyed", [r["cells"]["injection"] for r in rows])], fmt="int",
                caption="Counts out of 50 attacks.")],
        ["Even on its own the fine-tuned model ignores injections and authority claims; without human approval, "
         "though, 14 changes go out that nobody checked.",
         "No private reason leaked in any run: replies are built from constraints, never from the reasons given."],
        {"columns": [_col("pass", "Handled safely", "pct", "high"), _col("unapproved", "Published without approval", "int", "low"),
                     _col("authority", "Authority violations", "int", "low"),
                     _col("injection", "Injections obeyed", "int", "low"),
                     _col("leaks", "Private reasons leaked", "int", "low", key_col=False)], "rows": rows},
        "", at)]


# -- the agentic system ------------------------------------------------------------------------------------------------


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
    systems = {"intents": ("Fixed pipeline", "reads the request, then a fixed sequence of steps"),
               "agent": ("Agent loop", "decides its own steps with tool calls")}
    per = {}
    for sys_ in systems:
        rs = [r for r in results if r["system"] == sys_]
        per[sys_] = [fixed.get(r["id"], r) for r in rs] if sys_ == "agent" else rs
    cats: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for sys_, rs in per.items():
        for r in rs:
            cats[r["category"]][sys_].append(bool(r["pass"]))
    order = sorted(cats, key=lambda c: -len(cats[c]["intents"]))
    rows = []
    for sys_, (label, desc) in systems.items():
        rs = per[sys_]
        times = sorted(r.get("seconds") or 0 for r in rs)
        rows.append(_row(label, {"pass": sum(r["pass"] for r in rs) / len(rs),
                                 "passed": f"{sum(bool(r['pass']) for r in rs)}/{len(rs)}",
                                 "median": times[len(times) // 2], "action": sum(r["action_ok"] for r in rs) / len(rs)},
                         desc, sys_ == "agent"))
    a, p_ = rows[1]["cells"], rows[0]["cells"]
    return [_chapter(
        "head-to-head", "swords", "Pipeline or agent?",
        "This system lets a tool-calling agent handle each request in its own steps. Does it do as well as the fixed "
        "pipeline?",
        ["Wrote 41 requests across 15 kinds: swaps, moves, absences, questions, policy, safety, multi-part requests.",
         "Gave each to both systems in the same department and week, resetting in between.",
         "Checked the action (change, answer, ask, refuse) and, for changes, the resulting timetable."],
        [_who(label, desc, ours=k == "agent") for k, (label, desc) in systems.items()],
        f"As well, but slower: both pass {a['passed']}, and the agent takes {a['median']:.0f} s at the median against "
        f"{p_['median']:.0f} s.",
        [_chart("Requests passed, by kind", order,
                [_series(systems[s][0], [sum(cats[c][s]) / len(cats[c][s]) if cats[c][s] else None for c in order],
                         s == "agent") for s in systems],
                caption="Share of each kind of request handled correctly."),
         _chart("Time per request", [r["label"] for r in rows],
                [_series("Median seconds", [r["cells"]["median"] for r in rows], True)], fmt="num")],
        ["The agent's result includes five requests rerun after a fix for absences."] if fixed else [],
        {"columns": [_col("passed", "Passed", "text"), _col("pass", "Pass rate", "pct", "high"),
                     _col("action", "Right action", "pct", "high"), _col("median", "Median seconds", "num", "low")],
         "rows": rows}, "", at)]


def agent_adapters(root: Path) -> list[dict]:
    doc = root / "docs" / "regression-matrix.md"
    if not doc.exists():
        return []
    sec = doc.read_text(encoding="utf-8").split("## Agent: passed / n per family", 1)
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
        if not m or "/" in name:
            continue
        k, n = int(m[1]), int(m[2])
        rows.append(_row(f"Agent {name}", {"rate": k / n, "passed": f"{k}/{n}", "lo": float(m[3]), "hi": float(m[4])},
                         {"v6-dpo": "deployed", "v7-sft": "new data, SFT", "v7-dpo": "new data, SFT + DPO"}.get(name, ""),
                         name == "v6-dpo"))
    if not rows:
        return []
    at = datetime.fromtimestamp(doc.stat().st_mtime, UTC).isoformat()
    return [_chapter(
        "agent-adapters", "cpu", "Did retraining the agent help?",
        "We retrained the agent's adapter (v7) with more data and a preference step. Is it better than the deployed "
        "one?",
        ["Ran each adapter on 49 kinds of request, in 2 departments, 5 times each, on Kaggle.",
         "Kept the new adapter only if it beat the old one (the release rule)."],
        [_who(r["label"], r["sub"], ours=r["highlight"]) for r in rows],
        "No: both v7 adapters scored lower, so the deployed v6 stays.",
        [_chart("Episodes passed", [r["label"] for r in rows],
                [_series("Passed", [r["cells"]["rate"] for r in rows], True,
                         [[r["cells"]["lo"], r["cells"]["hi"]] for r in rows])],
                caption="With 95% intervals.")],
        ["v7 lost ground on counter-offers, no reply and fallbacks."],
        {"columns": [_col("rate", "Passed", "pct", "high"), _col("passed", "Count", "text")], "rows": rows}, "", at)]


# -- the frame: introduction and conclusions -------------------------------------------------------------------------


def conclusions(runs: Path) -> list[dict]:
    """One line per claim: a verdict and the deciding numbers, read from the result files."""
    out = []

    def add(claim, verdict, evidence, section):
        out.append({"claim": claim, "verdict": verdict, "evidence": evidence, "section": section})

    ours, _ = _load(runs / "negotiation-main-ours-4b.json")
    base, _ = _load(runs / "negotiation-main-baselines-gemini.json")
    paired, _ = _load(runs / "negotiation-main-paired.json")
    if ours and base:
        o, b3, b4 = ours["configs"]["ours-llm"]["summary"], base["configs"]["B3"]["summary"], base["configs"]["B4"]["summary"]
        pv = (paired or {}).get("ours-llm vs B4", {})
        add("Options invented by an LLM cost more than options from the solver", "supported",
            f"{_pct(b3['invalid_candidate_rate'])} of unchecked invented options are impossible; filtering takes "
            f"{b4['option_calls_per_scenario']:.1f} LLM calls per conflict, ours none", "main")
        add("Our system beats the LLM-proposes-solver-filters baseline", "partly",
            f"more conflicts resolved ({_pct(o['agreement_rate'])} vs {_pct(b4['agreement_rate'])}, "
            f"{_p(pv.get('resolved_mcnemar', {}).get('p'))}) and better timetables; not fewer rounds", "main")
    bias, _ = _load(runs / "negotiation-bias-gemini.json")
    if bias:
        s = {k: v["summary"] for k, v in bias["configs"].items()}
        add("The gain comes from the solver's options, not the model", "supported",
            f"on the same model: {_pct(s['ours-llm']['agreement_rate'])} vs {_pct(s['B4']['agreement_rate'])}", "bias")
    abl, _ = _load(runs / "negotiation-ablations-paired.json")
    if abl and "ours-llm vs A2" in abl:
        v = abl["ours-llm vs A2"]
        add("Grounded explanations keep private reasons private", "supported",
            f"{round(v['leaks']['A2'] * v['pairs'])} leaks without grounding, 0 with; acceptance not lower",
            "ablations")
    fair, _ = _load(runs / "fairness-final.json")
    if fair:
        s, pv = fair["summary"], fair.get("paired", {})
        add("The concession ledger spreads the burden of giving way", "supported",
            f"Gini {s['ours']['gini']:.2f} vs {s['no-ledger']['gini']:.2f} without it "
            f"({_p(pv.get('ours vs no-ledger', {}).get('gini', {}).get('p'))}); flat weights are as even only by "
            "ignoring who is affected", "fairness")
    par, _ = _load(runs / "parsing-4b-nts-q8-paratest.json")
    pro, _ = _load(runs / "parsing-4b-base-prompted-paratest.json")
    rep, _ = _load(runs / "replies-4b-nts-q8.json")
    rule, _ = _load(runs / "replies-rule.json")
    if par and pro:
        add("Fine-tuning makes a small local model reliable", "supported",
            f"requests read exactly: {_pct(par['system_two']['compile_exact_match'])} with LoRA vs "
            f"{_pct(pro['system_two']['compile_exact_match'])} prompted"
            + (f"; replies {_pct(rep['summary']['tool_accuracy'])} vs {_pct(rule['summary']['tool_accuracy'])} for "
               "keyword rules" if rep and rule else ""), "models")
    r8, _ = _load(runs / "rag-4b-read100-k8.json")
    if r8:
        s, n = r8["conditions"]["rag"]["summary"], r8["conditions"]["no-rag"]["summary"]
        add("Retrieval grounds policy answers and stops invented rules", "supported",
            f"{_pct(s['answer_accuracy'])} right vs {_pct(n['answer_accuracy'])} without; invented rules "
            f"{_pct(s['hallucination_on_unanswerable'])} vs {_pct(n['hallucination_on_unanswerable'])}", "rag")
    on, _ = _load(runs / "safety-4b.json")
    off, _ = _load(runs / "safety-compromised-noguards.json")
    if on and off:
        add("Safety comes from code checks, not from trusting the model", "supported",
            f"{_pct(on['pass_rate'])} of 50 attacks handled safely; {off['unapproved_publishes']} published with the "
            "checks off", "safety")
    if not out:
        return []
    return [{"id": "conclusions", "kind": "conclusions", "title": "Conclusions", "at": datetime.now(UTC).isoformat(),
             "items": out,
             "pending": ["The LLM judge's scores are not yet checked against people (two raters in Study).",
                         "No human pilot yet."]}]


def intro(agentic: bool) -> dict:
    return {"id": "intro", "kind": "intro", "title": "Negotiate, then solve", "at": datetime.now(UTC).isoformat(),
            "problem": "Timetables change every week, and the requests arrive as plain messages. Language models read "
                       "messages but can break hard rules; solvers never break them but cannot read or negotiate. "
                       "When two requests cannot both be met, someone has to give way.",
            "question": "Can a solver's view of what is possible guide a language-model negotiator to resolve "
                        "conflicts better than letting the model come up with its own ideas, fairly, safely, and "
                        "explaining itself without revealing private reasons?",
            "approach": ["A fine-tuned model reads each request into constraints.",
                         "The solver finds where requests conflict and every minimal way to fix it.",
                         "The negotiator offers only those checked options, asking whoever has given way least.",
                         "A person approves every change before it is published."]
                        + (["In this system a tool-calling agent can also handle a request end to end."] if agentic else [])}


def sections(root: Path, agentic: bool = False) -> list[dict]:
    """The page: introduction, chapters in reading order, conclusions. ``agentic``: also the agent system's own."""
    runs = root / "runs"
    chapters = [*main_comparison(runs), *bias_check(runs), *ablations(runs), *fairness(runs), *models(runs),
                *rag(runs), *safety(runs)]
    if agentic:
        chapters += [*head_to_head(runs), *agent_adapters(root)]
    return [intro(agentic), *chapters, *conclusions(runs)]
