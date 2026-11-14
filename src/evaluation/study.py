"""Human data (proposal Section 12.5): judge validation by the team and the
anonymised faculty pilot, collected in the web app (``/study``).

    uv run python -m evaluation.study build runs/negotiation-study-source.json   # the item sets
    uv run python -m evaluation.study add-rules runs/rag-4b-read100-k8.json      # + the rule answers
    uv run python -m evaluation.study report                                     # kappas, H3, live checks

Three labelling tasks are built from one negotiation report (``ours-llm`` and
``A2`` with ``explanations`` recorded, seed 0):

* ``claims``   100 explanation claims with the facts the explanation could use;
               the label is "supported by the facts or not". Machine label:
               the faithfulness verifier. Every claim the verifier rejected is
               kept (up to 40), the rest are sampled, half grounded, half free.
* ``replies``  100 stakeholder replies under the message they answer; the label
               is what the reply does (accept an option, counter, decline,
               other). Machine label: the recorded reply call.
* ``rules``    60 answers to questions about the rules (``evaluation.rag``: the assistant with retrieval and,
               blind, the same model without it), with the official rule text the question has (or none);
               the label is what the answer does: declines (no rule covers it), consistent with the rule, or
               not. Machine label: the Gemini judge's verdict, so the RAG results get the same check.
* ``ratings``  110 messages rated 1-5 for clarity and acceptability, blind to the
               configuration. Grounded (ours-llm) and free-form (A2) messages
               come in pairs from the same scenario; a 24-message subset is the
               pilot set (H3: "the human pilot rates clarity and acceptability");
               unpaired grounded messages fill the set to 110 for the judge kappa.

Team raters (codes ``T-n``) do all four; pilot participants (``P-nn``) rate
the pilot set and take part in a live session in the portal as one demo
faculty member, a tour of the system as it is used (``LIVE_STEPS``): they ask
the assistant about their timetable and about the rules and say whether each
answer was right, write a request in their own words and say whether it was
understood, take part in a practice clash (reply to the negotiator in their own
words, confirm the reading of the reply, rate the message), move or swap a class
from the timetable, and answer a short closing questionnaire (``SURVEY``).

Nothing identifies a participant beyond the code. Everything is stored under
``runs/study/`` (``participants.json``, ``labels.jsonl``, ``live.jsonl``);
``report`` writes ``human_ratings.csv`` in the format ``evaluation.judge --kappa``
reads. LLM-scored metrics are reported only where kappa >= 0.6 (Section 12.5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import threading
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from .stats import cohen_kappa, mean, wilcoxon

TASKS = ("claims", "replies", "ratings", "rules")
GROUNDED, FREE = "ours-llm", "A2"
N_CLAIMS, N_REPLIES, N_RATINGS, MAX_UNSUPPORTED, PILOT_PAIRS = 100, 100, 110, 40, 12
KAPPA_OK = 0.6
REPLY_LABELS = ("accept:A", "accept:B", "accept:C", "counter", "reject", "other")
RULE_LABELS = ("declines", "consistent", "not_consistent")
N_RULES = 60

# The pilot's live session: a tour of the system as a faculty member uses it. (step, record kind, title, what to
# do.) A step is done when a record of its kind (and, for checks, its step) exists in live.jsonl.
LIVE_STEPS = (
    ("timetable", "answer", "Ask the assistant about your timetable",
     "For example: \u201cWhen am I free on Thursday?\u201d or \u201cWhere is my next class?\u201d Then say whether "
     "the answer was right (a box appears under it)."),
    ("rules", "answer", "Ask about a rule",
     "For example: \u201cHow many students can be in a lab at once?\u201d or \u201cCan I teach a class online?\u201d "
     "Say whether the answer was right and backed by the rule it cites."),
    ("preference", "understood", "Tell the office a preference",
     "In your own words, for example times you would rather not teach. Open the request from its card and say "
     "whether it was understood."),
    ("practice", "rating", "The practice clash",
     "A colleague has claimed a lab you need. Ask for it, answer the office's message in your own words, confirm "
     "how your reply was read, and rate the message."),
    ("click", "understood", "Change a class from the timetable",
     "Click one of your classes on the timetable and move, swap or cancel it. Then open the request and say whether "
     "it was understood."),
    ("survey", "survey", "A few closing questions", "Six statements about the session, and anything you want to add."),
)
SURVEY = (
    ("understood", "The assistant understood what I wrote."),
    ("timetable", "Its answers about my timetable were correct."),
    ("rules", "Its answers about the rules were trustworthy."),
    ("clash", "The office's message explained the clash and my options clearly."),
    ("next", "I knew at each point what would happen to my request."),
    ("use", "I would use this instead of e-mailing the timetable office."),
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def judge_id(config: str, scenario: str, k: int, seed: int = 0) -> str:
    """The message id ``evaluation.judge`` uses, so human and judge ratings join."""
    return f"{config}/{scenario}/{k + 1}" if not seed else f"{config}/{scenario}/s{seed}/{k + 1}"


def reply_label(reply: dict) -> str:
    d = reply.get("decision")
    if d == "accept":
        return f"accept:{(reply.get('choice') or '').strip().upper()}"
    return d if d in ("counter", "reject") else "other"


# ---------------------------------------------------------------------------
# Item sets
# ---------------------------------------------------------------------------


def build_items(report: dict, seed: int = 0, rng_seed: int = 0) -> dict:
    rng = random.Random(rng_seed)
    rows = {c: [r for r in report["configs"][c]["rows"] if r.get("seed", 0) == seed] for c in (GROUNDED, FREE)}
    if any("explanations" not in r for rs in rows.values() for r in rs):
        raise SystemExit("the report has no recorded explanations: re-run evaluation.negotiation (it records them now)")

    claims = []
    for config, rs in rows.items():
        for r in rs:
            for k, e in enumerate(r["explanations"]):
                for j, c in enumerate(e["claims"]):
                    claims.append({"config": config, "mode": e["mode"], "scenario": r["scenario"], "message": k,
                                   "claim": c["text"], "cited": c["facts"], "facts": e["facts"],
                                   "machine": bool(c["supported"]), "key": f"{config}/{r['scenario']}/{k}/{j}"})
    unsupported = [c for c in claims if not c["machine"]]
    picked = rng.sample(unsupported, min(MAX_UNSUPPORTED, len(unsupported)))
    for mode in ("grounded", "free"):
        pool = [c for c in claims if c["machine"] and c["mode"] == mode]
        picked += rng.sample(pool, min(len(pool), (N_CLAIMS - len(picked)) // (2 if mode == "grounded" else 1)))
    rng.shuffle(picked)
    claims = [{"id": f"CL-{i + 1:03d}", **c} for i, c in enumerate(picked)]

    replies = []
    for config, rs in rows.items():
        for r in rs:
            for k, (text, rep) in enumerate(zip(r["messages"], r["replies"], strict=False)):
                if rep.get("text") and rep.get("decision") != "no_reply":
                    replies.append({"config": config, "scenario": r["scenario"], "message": text,
                                    "reply": rep["text"], "machine": reply_label(rep), "key": f"{config}/{r['scenario']}/{k}"})
    by_label = defaultdict(list)
    for x in replies:
        by_label[x["machine"]].append(x)
    picked = []
    while len(picked) < min(N_REPLIES, len(replies)):  # round-robin over labels: rare ones are all kept
        for xs in by_label.values():
            if xs and len(picked) < N_REPLIES:
                picked.append(xs.pop(rng.randrange(len(xs))))
    rng.shuffle(picked)
    replies = [{"id": f"RE-{i + 1:03d}", **x} for i, x in enumerate(picked)]

    msgs = {c: {(r["scenario"], k): m for r in rs for k, m in enumerate(r["messages"])} for c, rs in rows.items()}
    paired = sorted(set(msgs[GROUNDED]) & set(msgs[FREE]))
    ratings = []
    for config in (GROUNDED, FREE):
        for sc, k in paired:
            ratings.append({"id": judge_id(config, sc, k, seed), "config": config,
                            "mode": "grounded" if config == GROUNDED else "free", "scenario": sc, "message": k,
                            "text": msgs[config][(sc, k)], "pilot": False})
    extra = [p for p in sorted(msgs[GROUNDED]) if p not in set(paired)]
    for sc, k in rng.sample(extra, min(len(extra), max(0, N_RATINGS - len(ratings)))):  # judge kappa needs >= 100
        ratings.append({"id": judge_id(GROUNDED, sc, k, seed), "config": GROUNDED, "mode": "grounded",
                        "scenario": sc, "message": k, "text": msgs[GROUNDED][(sc, k)], "pilot": False})
    firsts = [p for p in paired if p[1] == 0]
    kinds = defaultdict(list)
    for p in firsts:
        kinds[p[0].split("-", 2)[-1]].append(p)
    pilot: list[tuple[str, int]] = []
    while len(pilot) < min(PILOT_PAIRS, len(firsts)):  # spread over conflict kinds
        for xs in kinds.values():
            if xs and len(pilot) < PILOT_PAIRS:
                pilot.append(xs.pop(0))
    for it in ratings:
        it["pilot"] = (it["scenario"], it["message"]) in pilot
    return {"built": _now(), "seed": seed, "claims": claims, "replies": replies, "ratings": ratings}


def rule_label(row: dict) -> str:
    """The judge's verdict on a rule answer, as the rater's choice: declines, consistent or not."""
    if row.get("no_rule") or row.get("abstains"):
        return "declines"
    return "consistent" if row.get("agrees") else "not_consistent"


def build_rule_items(report: dict, rules: dict, rng_seed: int = 0) -> list[dict]:
    """Rule answers for the team: with retrieval (the assistant as deployed) and without, blind to which, half
    answerable, half not, so all three labels occur. Each carries the official rule text the judge saw."""
    rng = random.Random(rng_seed)
    picked = []
    for cond, n in (("rag", N_RULES * 2 // 3), ("no-rag", N_RULES // 3)):
        rows = [r for r in report["conditions"].get(cond, {}).get("rows", [])
                if r.get("abstains") is not None or r.get("no_rule")]
        for gold in (True, False):
            pool = [r for r in rows if bool(r["gold"]) == gold]
            picked += [(cond, r) for r in rng.sample(pool, min(len(pool), n // 2))]
    rng.shuffle(picked)
    return [{"id": f"RU-{i + 1:03d}", "condition": cond, "query": r["query"], "answer": r["answer"],
             "rules": [{"id": g, "title": rules[g].title, "text": rules[g].text} for g in r["gold"] if g in rules],
             "machine": rule_label(r), "key": r["key"]} for i, (cond, r) in enumerate(picked)]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class Study:
    """Participants, labels and live-session records under one directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- items ----------------------------------------------------------------------------

    @property
    def items_path(self) -> Path:
        return self.root / "items.json"

    def items(self) -> dict | None:
        return json.loads(self.items_path.read_text(encoding="utf-8")) if self.items_path.exists() else None

    def save_items(self, items: dict) -> None:
        self.items_path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")

    def tasks_for(self, kind: str) -> list[str]:
        if kind != "team":
            return ["ratings"]
        items = self.items() or {}
        return [t for t in TASKS if items.get(t)] or list(TASKS)

    def team(self) -> list[str]:
        return sorted(c for c, p in self.active().items() if p["kind"] == "team")

    def assigned(self, code: str, task: str) -> set[str]:
        """Team work is shared: every item goes to exactly two raters, rotating through every pair
        of active team raters, so each does about 2/N of a task (all of it when N <= 2)."""
        items = (self.items() or {}).get(task, [])
        team = self.team()
        if code not in team:
            return set()
        if len(team) <= 2:
            return {i["id"] for i in items}
        pairs = list(combinations(team, 2))
        return {it["id"] for n, it in enumerate(items) if code in pairs[n % len(pairs)]}

    def queue(self, code: str, kind: str, task: str, labels: dict | None = None) -> list[dict]:
        """The rater's items for a task, in an order of their own (blind to configuration): a pilot
        participant's are the pilot set; a team rater's are their share, plus anything they already
        labelled, minus items two others have already finished (after the team changed size)."""
        items = (self.items() or {}).get(task, [])
        if kind == "pilot":
            items = [i for i in items if i["pilot"]] if task == "ratings" else []
        else:
            labels = self.labels() if labels is None else labels
            mine = labels.get((task, code), {})
            others = Counter(i for (t, c), vals in labels.items() if t == task and c != code and c in self.team()
                             for i in vals)
            share = self.assigned(code, task)
            items = [i for i in items if i["id"] in mine or (i["id"] in share and others[i["id"]] < 2)]
        seed = int(hashlib.sha256(f"{code}/{task}".encode()).hexdigest()[:8], 16)
        items = list(items)
        random.Random(seed).shuffle(items)
        return items

    def coverage(self, labels: dict | None = None) -> dict[str, dict]:
        """How many items of each task have the two team labels agreement needs."""
        labels = self.labels() if labels is None else labels
        team = set(self.team())
        out = {}
        for task in TASKS:
            items = (self.items() or {}).get(task, [])
            n = Counter(i for (t, c), vals in labels.items() if t == task and c in team for i in vals)
            out[task] = {"items": len(items), "double": sum(n[i["id"]] >= 2 for i in items),
                         "single": sum(n[i["id"]] == 1 for i in items)}
        return out

    # -- participants ---------------------------------------------------------------------

    @property
    def participants_path(self) -> Path:
        return self.root / "participants.json"

    def participants(self) -> dict[str, dict]:
        p = self.participants_path
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def _save_participants(self, ps: dict) -> None:
        self.participants_path.write_text(json.dumps(ps, indent=1), encoding="utf-8")

    def add_participants(self, kind: str, n: int, persona: str | None = None) -> list[str]:
        if kind not in ("team", "pilot"):
            raise ValueError("kind must be team or pilot")
        with self._lock:
            ps = self.participants()
            prefix = "T-" if kind == "team" else "P-"
            taken = [int(c[2:]) for c in ps if c.startswith(prefix)]
            start = max(taken, default=0) + 1
            codes = [f"{prefix}{i}" if kind == "team" else f"{prefix}{i:02d}" for i in range(start, start + n)]
            for c in codes:
                ps[c] = {"kind": kind, "persona": persona if kind == "pilot" else None, "created": _now(),
                         "consented": None}
            self._save_participants(ps)
        return codes

    def active(self) -> dict[str, dict]:
        """Participants whose code still works; a revoked one counts in no result."""
        return {c: p for c, p in self.participants().items() if not p.get("revoked")}

    def get(self, code: str) -> dict | None:
        code = code.strip().upper()
        p = self.active().get(code)
        return p and {"code": code, **p}

    def revoke(self, code: str, delete_answers: bool = False) -> dict:
        """The code stops working and its answers leave every result. With
        ``delete_answers`` (a withdrawal) its labels and live records are erased
        from disk, and the code cannot be restored."""
        with self._lock:
            ps = self.participants()
            if code not in ps:
                raise KeyError(code)
            ps[code]["revoked"] = ps[code].get("revoked") or _now()
            if delete_answers:
                for name in ("labels.jsonl", "live.jsonl"):
                    rows = [r for r in self._read(name) if r.get("code") != code]
                    (self.root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                  encoding="utf-8")
                ps[code]["answers_deleted"] = _now()
            self._save_participants(ps)
        return {"code": code, **ps[code]}

    def clear(self) -> dict:
        """Start over: every participant, label and live record goes; the item sets stay."""
        with self._lock:
            counts = {"participants": len(self.participants()), "labels": len(self._read("labels.jsonl")),
                      "live": len(self._read("live.jsonl"))}
            for name in ("participants.json", "labels.jsonl", "live.jsonl"):
                (self.root / name).unlink(missing_ok=True)
        return counts

    def restore(self, code: str) -> dict:
        with self._lock:
            ps = self.participants()
            if ps[code].get("answers_deleted"):
                raise ValueError("this participant withdrew and their answers were deleted")
            ps[code].pop("revoked", None)
            self._save_participants(ps)
        return {"code": code, **ps[code]}

    def consent(self, code: str) -> dict:
        with self._lock:
            ps = self.participants()
            ps[code]["consented"] = ps[code]["consented"] or _now()
            self._save_participants(ps)
        return {"code": code, **ps[code]}

    # -- labels and live records -----------------------------------------------------------

    def _append(self, name: str, row: dict) -> dict:
        row = {"at": _now(), **row}
        with self._lock, (self.root / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def _read(self, name: str) -> list[dict]:
        p = self.root / name
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x] if p.exists() else []

    def label(self, code: str, task: str, item: str, value: dict) -> dict:
        return self._append("labels.jsonl", {"code": code, "task": task, "item": item, "value": value})

    def labels(self) -> dict[tuple[str, str], dict[str, dict]]:
        """{(task, code): {item: value}}, the latest label of each item."""
        out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        for r in self._read("labels.jsonl"):
            out[(r["task"], r["code"])][r["item"]] = r["value"]
        return out

    def live(self, code: str, persona: str, kind: str, **fields) -> dict:
        return self._append("live.jsonl", {"code": code, "persona": persona, "kind": kind, **fields})

    def live_rows(self) -> list[dict]:
        return self._read("live.jsonl")

    def progress(self, code: str, kind: str) -> dict[str, dict]:
        labels = self.labels()
        out = {}
        for task in self.tasks_for(kind):
            q = self.queue(code, kind, task, labels)
            done = labels.get((task, code), {})
            out[task] = {"done": sum(i["id"] in done for i in q), "total": len(q)}
        return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _kappa(a: list, b: list, weights: str | None = None) -> float | None:
    if len(a) < 2:
        return None
    k = cohen_kappa(a, b, weights=weights)
    return None if math.isnan(k) else round(k, 3)


def _pairwise(by_rater: dict[str, dict[str, object]], weights: str | None = None) -> dict:
    """Agreement between raters. With the work shared, different pairs label different items, so the
    headline is pooled: each item's first two labels (by rater code) form one pair, over all items.
    Pairs of raters that share at least 10 items are listed as well."""
    out = {}
    humans = sorted(r for r in by_rater if r != "llm_judge")  # the judge labels everything: never pooled
    pooled_a, pooled_b = [], []
    for item in sorted({i for r in humans for i in by_rater[r]}):
        got = [by_rater[r][item] for r in humans if item in by_rater[r]]
        if len(got) >= 2:
            pooled_a.append(got[0])
            pooled_b.append(got[1])
    if len(humans) > 2 and len(pooled_a) >= 2:
        out["all raters (pooled)"] = {"n": len(pooled_a), "kappa": _kappa(pooled_a, pooled_b, weights)}
    least = 2 if len(humans) <= 2 else 10
    for a, b in combinations(sorted(by_rater), 2):
        common = sorted(set(by_rater[a]) & set(by_rater[b]))
        if len(common) >= least:
            out[f"{a} vs {b}"] = {"n": len(common),
                                  "kappa": _kappa([by_rater[a][i] for i in common], [by_rater[b][i] for i in common],
                                                  weights)}
    return out


def _vs_machine(by_rater: dict[str, dict[str, object]], machine: dict[str, object]) -> dict:
    out = {}
    for r, labels in sorted(by_rater.items()):
        common = sorted(set(labels) & set(machine))
        if common:
            h, m = [labels[i] for i in common], [machine[i] for i in common]
            k = _kappa(h, m)
            out[r] = {"n": len(common), "kappa": k, "agreement": round(mean([x == y for x, y in zip(h, m, strict=True)]), 3),
                      "ok": k is not None and k >= KAPPA_OK}
    return out


def analyse(study: Study, judge_ratings: dict[str, dict] | None = None) -> dict:
    items = {t: [] for t in TASKS} | (study.items() or {})
    ps = study.active()  # a revoked participant's answers count nowhere
    labels = study.labels()
    team = {c for c, p in ps.items() if p["kind"] == "team"}
    pilot = {c for c, p in ps.items() if p["kind"] == "pilot"}

    def rater_labels(task: str, codes: set[str], field: str) -> dict[str, dict[str, object]]:
        return {c: {i: v[field] for i, v in labels.get((task, c), {}).items() if field in v}
                for c in codes if labels.get((task, c))}

    claim_h = rater_labels("claims", team, "supported")
    reply_h = rater_labels("replies", team, "label")
    rule_h = rater_labels("rules", team, "label")
    report: dict = {
        "coverage": study.coverage(labels),
        "participants": {"team": len(team), "pilot": len(pilot),
                         "pilot_consented": sum(bool(ps[c]["consented"]) for c in pilot)},
        "claims": {"items": len(items["claims"]),
                   "human_vs_verifier": _vs_machine(claim_h, {i["id"]: i["machine"] for i in items["claims"]}),
                   "human_vs_human": _pairwise(claim_h)},
        "replies": {"items": len(items["replies"]),
                    "human_vs_recorded": _vs_machine(reply_h, {i["id"]: i["machine"] for i in items["replies"]}),
                    "human_vs_human": _pairwise(reply_h)},
        "rules": {"items": len(items.get("rules", [])),
                  "human_vs_judge": _vs_machine(rule_h, {i["id"]: i["machine"] for i in items.get("rules", [])}),
                  "human_vs_human": _pairwise(rule_h)},
    }

    rating_rows = []  # (code, kind, item, clarity, acceptability)
    for (task, code), vals in labels.items():
        if task == "ratings" and code in ps:
            rating_rows += [(code, ps[code]["kind"], i, v["clarity"], v["acceptability"]) for i, v in vals.items()]
    by_rater = defaultdict(dict)
    for code, kind, i, c, a in rating_rows:
        if kind == "team":
            by_rater[code][i] = (c, a)
    if judge_ratings:
        by_rater["llm_judge"] = {i: (v["clarity"], v["acceptability"]) for i, v in judge_ratings.items()}
    kappas = {}
    for k, idx in (("clarity", 0), ("acceptability", 1)):
        for pair, v in _pairwise({r: {i: x[idx] for i, x in d.items()} for r, d in by_rater.items()}, "quadratic").items():
            kappas.setdefault(pair, {"n": v["n"]})[k] = v["kappa"]
    meta = {i["id"]: i for i in items["ratings"]}
    report["ratings"] = {"items": len(items["ratings"]), "pilot_items": sum(i["pilot"] for i in items["ratings"]),
                         "kappa": kappas, "h3_pilot": _h3(rating_rows, meta, "pilot"),
                         "h3_team": _h3(rating_rows, meta, "team")}

    live = [r for r in study.live_rows() if r["code"] in ps]
    understood = Counter(r["value"] for r in live if r["kind"] == "understood")
    reply_checks = [r for r in live if r["kind"] == "reply"]
    live_ratings = [r for r in live if r["kind"] == "rating"]
    report["live"] = {
        "requests_checked": sum(understood.values()), "understood": dict(understood),
        "replies": len(reply_checks),
        "reply_reading_confirmed": mean([r["confirmed"] for r in reply_checks]),
        "message_ratings": len(live_ratings),
        "clarity": mean([r["value"]["clarity"] for r in live_ratings]),
        "acceptability": mean([r["value"]["acceptability"] for r in live_ratings]),
    }
    for topic in ("timetable", "rules"):  # "was this answer right?" on the assistant's answers
        got = Counter(r["value"] for r in live if r["kind"] == "answer" and r.get("step") == topic)
        report["live"][f"{topic}_answers"] = dict(got)
    surveys = [r["value"] for r in live if r["kind"] == "survey" and isinstance(r.get("value"), dict)]
    report["live"]["survey"] = {"n": len(surveys), **{k: mean([v[k] for v in surveys if isinstance(v.get(k), int)])
                                                       for k, _ in SURVEY}}
    report["live"]["steps_done"] = {code: sorted(steps_done(live, code)) for code in sorted(pilot)}
    return report


def steps_done(live: list[dict], code: str) -> set[str]:
    """The live-session steps a pilot participant has finished (``LIVE_STEPS``): a check of the step's kind,
    for the checks that more than one step uses (answers, "was it understood?") tagged with the step."""
    mine = [r for r in live if r["code"] == code]
    return {step for step, kind, *_ in LIVE_STEPS
            if any(r["kind"] == kind and (r.get("step") == step if kind in ("answer", "understood") else True)
                   for r in mine)}


def _h3(rows: list[tuple], meta: dict[str, dict], kind: str) -> dict:
    """Grounded vs free-form ratings from one group of raters, paired on
    (rater, scenario, message)."""
    by = defaultdict(dict)
    for code, k, i, c, a in rows:
        if k == kind and i in meta:
            m = meta[i]
            by[(code, m["scenario"], m["message"])][m["mode"]] = (c, a)
    pairs = [v for v in by.values() if "grounded" in v and "free" in v]
    out: dict = {"raters": len({code for code, k, *_ in rows if k == kind}), "pairs": len(pairs)}
    for name, idx in (("clarity", 0), ("acceptability", 1)):
        g = [p["grounded"][idx] for p in pairs]
        f = [p["free"][idx] for p in pairs]
        out[name] = {"grounded": mean(g), "free": mean(f), "wilcoxon": wilcoxon(g, f) if pairs else None}
    return out


def write_ratings_csv(study: Study, path: Path) -> int:
    """Team ratings in the ``evaluation.judge --kappa`` format."""
    ps = study.active()
    rows = [(i, code, v["clarity"], v["acceptability"]) for (task, code), vals in study.labels().items()
            if task == "ratings" and ps.get(code, {}).get("kind") == "team" for i, v in vals.items()]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["message_id", "rater", "clarity", "acceptability"])
        w.writerows(sorted(rows))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the item sets from a negotiation report")
    b.add_argument("report", type=Path)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--force", action="store_true", help="replace items that already have labels")
    ar = sub.add_parser("add-rules", help="add the rule-answer items from an evaluation.rag report")
    ar.add_argument("report", type=Path)
    ar.add_argument("--force", action="store_true", help="replace rule items that already have labels")
    r = sub.add_parser("report", help="kappas, H3 and the live checks")
    r.add_argument("--judge", type=Path, default=Path("runs/judge-ratings.json"))
    for p in (b, ar, r):
        p.add_argument("--dir", type=Path, default=Path("runs/study"))
    args = ap.parse_args()
    study = Study(args.dir)
    if args.cmd == "add-rules":
        from agents.documents import load_corpus

        items = study.items()
        if items is None:
            raise SystemExit("build the item sets first")
        if any(t == "rules" for t, _ in study.labels()) and not args.force:
            raise SystemExit("rule answers already have labels; pass --force to replace them")
        corpus, _ = load_corpus(Path("data/handbook.md"), Path("data/policies"), None)
        items["rules"] = build_rule_items(json.loads(args.report.read_text(encoding="utf-8")),
                                          {x.id: x for x in corpus})
        items["rules_source"] = str(args.report)
        study.save_items(items)
        print(f"{len(items['rules'])} rule answers -> {study.items_path}:",
              dict(Counter(i["machine"] for i in items["rules"])), dict(Counter(i["condition"] for i in items["rules"])))
        return
    if args.cmd == "build":
        if study.items() and study.labels() and not args.force:
            raise SystemExit("labels exist for the current items; pass --force to rebuild (labels would no longer match)")
        items = build_items(json.loads(args.report.read_text(encoding="utf-8")), seed=args.seed)
        items["source"] = str(args.report)
        study.save_items(items)
        print(f"{len(items['claims'])} claims, {len(items['replies'])} replies, {len(items['ratings'])} messages "
              f"({sum(i['pilot'] for i in items['ratings'])} in the pilot set) -> {study.items_path}")
        print("claims by verifier label:", dict(Counter(i["machine"] for i in items["claims"])),
              "| replies by label:", dict(Counter(i["machine"] for i in items["replies"])))
    else:
        judge = json.loads(args.judge.read_text(encoding="utf-8")) if args.judge.exists() else None
        print(json.dumps(analyse(study, judge), indent=1))
        n = write_ratings_csv(study, study.root / "human_ratings.csv")
        print(f"{n} team ratings -> {study.root / 'human_ratings.csv'} (evaluation.judge --kappa)")


if __name__ == "__main__":
    main()
