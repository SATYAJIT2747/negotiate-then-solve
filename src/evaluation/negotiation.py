"""Negotiation experiments (proposal brief Section 5): our system against
baselines B1-B4 and the oracle, and ablations A1-A3, on the 60 scenarios.

    uv run python -m evaluation.negotiation --offline          # no API calls
    # the whole reduced plan on the local model (agent and simulators)
    uv run python -m evaluation.negotiation --plan --local --sim-local
    # one configuration, one seed
    uv run python -m evaluation.negotiation --configs ours-llm,B4 --local --sim-local --seeds 0
    # Gemini check of the main comparison on a stratified subset
    uv run python -m evaluation.negotiation --configs ours-llm,B4 --subset 20
    # the same scenarios through the request pipeline the app runs (submit, negotiate, approve)
    uv run python -m evaluation.negotiation --offline --pipeline

Configurations (brief Table, Section 5.2):

* ``ours``      MCS options, tiers + ledger, template explanations, scripted replies (offline)
* ``ours-llm``  the same with grounded LLM explanations; simulators voice replies
                with the LLM and each reply becomes one typed tool call
* ``B1``        control: the LLM writes the timetable directly; no solver
* ``B2``        control: the same tiers, weights, objective and solver as ours, but
                the cheapest correction set is imposed without asking anyone
* ``B3``        unverified: as ours-llm, but the LLM invents the options and they
                are offered unchecked (their feasibility is measured)
* ``B4``        filtered (primary baseline): as B3, but CP-SAT drops infeasible
                inventions and the LLM regenerates (at most 3 calls per round)
* ``oracle``    CP-SAT with every stakeholder's hidden flexibility; upper bound
* ``A1``        ours-llm with flat stakeholder weights, no tiers or ledger (H4)
* ``A2``        ours-llm with free-form instead of grounded explanations (H3)
* ``A3``        ours-llm with a keyword/regex reply handler instead of LLM tool calls
* ``A1-offline`` A1 with template explanations and scripted replies (no API)

Reduced plan (``--plan``). Each scenario carries one stakeholder family
(balanced within each kind), rather than every scenario meeting all four,
and only the comparison the research question rests on is repeated:

* ours-llm, B3, B4: all 60 scenarios x 3 seeds (540 runs)
* A1, A2, A3: a stratified 30, seed 0 (90 runs)
* B1: a stratified 20, seed 0 (20 runs)
* B2 and the oracle: all 60, no LLM

That is 650 LLM runs instead of the 5,040 of a full 60 x 4 x 3 grid.

A scenario is *resolved* when every affected stakeholder accepted an offered
alternative and the combined repair re-solves feasibly; otherwise it ends
escalated (deadlock, no reply, the round limit, a Tier 1-2 change, or a reply
that needs a person). Outcomes are scored with the oracle objective
(``benchmark.objective``). Scenarios run in order with one concession ledger
per configuration and seed, so the weighted concession burden reflects a
semester of conflicts among the same four faculty IDs. Results are reported
pooled and per stakeholder family.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from agents.explainer import Explainer, describe
from agents.ledger import ConcessionLedger
from agents.negotiation import Negotiator, ReplyParser, RuleReplyParser
from agents.priority import PriorityModel
from core.schemas import ConcessionEntry, Placement, Tier
from core.validator import verify_timetable
from language.llm import DailyQuotaReached, GeminiClient, default_model

from .benchmark import SEMESTER, Scenario, load, objective, original_ids, simulators
from .stats import bootstrap_ci, mcnemar, mean, wilcoxon

STAKEHOLDERS = ["F-301", "F-302", "F-303", "F-304"]
WORKERS = 1  # single-threaded CP-SAT: paired configurations must be offered the same options
OFFLINE = ["ours", "A1-offline", "B2", "oracle"]
LLM = ["ours-llm", "B3", "B4", "A1", "A2", "A3", "B1"]
PRIMARY = ["ours-llm", "B3", "B4"]  # repeated over every seed
SEEDS = [0, 1, 2]
ABLATION_SUBSET = 30
B1_SUBSET = 20
PLAN = ["ours-llm", "B4", "B3", "A1", "A2", "A3", "B1", "B2", "oracle"]

CONFIGS = {
    # option_source, flat priorities (no tiers, no ledger), explanation mode, replies
    # replies: "scripted" (structured), "llm" (voiced, LLM tool calls), "rule" (voiced, regex handler)
    "ours": ("mcs", False, "template", "scripted"),
    "ours-llm": ("mcs", False, "grounded", "llm"),
    "B3": ("llm", False, "grounded", "llm"),
    "B4": ("llm-filtered", False, "grounded", "llm"),
    "A1": ("mcs", True, "grounded", "llm"),
    "A2": ("mcs", False, "free", "llm"),
    "A3": ("mcs", False, "grounded", "rule"),
    "A1-offline": ("mcs", True, "template", "scripted"),
}


class _Timetable(BaseModel):
    class Item(BaseModel):
        session: str
        day: str
        slot: int
        room: str

    placements: list[Item]


B1_PROMPT = """You are a university timetabler. Place every session: choose a day, a start
slot and a room. Rules: a room, a faculty member or a section can hold one
session at a time; a session of duration d occupies d consecutive slots on
one day; labs need lab rooms with the equipment listed; respect every
constraint listed as far as possible. Output every session exactly once."""


def validity(sc: Scenario, assignment: dict[str, Placement] | None, extra=()) -> bool:
    """Hard-constraint validity (H1a): physical rules plus every Tier 0-2
    constraint, and any constraints the system itself claims to enforce."""
    if not assignment:
        return False
    must = [c for c in sc.constraints if c.hard and c.tier <= Tier.COMMITMENT] + list(extra)
    return not verify_timetable(sc.instance, assignment, must, week=sc.week)


def _moved(sc: Scenario, assignment) -> int:
    return sum(1 for s, p in sc.baseline.items() if assignment and assignment.get(s) != p)


def _scored(sc: Scenario, relaxed: list[str], moved: int | None) -> dict:
    """The oracle objective of an outcome, when it produced a timetable."""
    if moved is None:
        return {"objective": None, "tiers": None}
    tiers, cost = objective(sc, relaxed, moved)
    return {"objective": cost, "tiers": tiers}


def _negotiator(sc: Scenario, config: str, ledger: ConcessionLedger, agent) -> Negotiator:
    source, flat, mode, replies = CONFIGS[config]
    pm = PriorityModel(sc.instance, ledger=ledger, semester=SEMESTER, baseline=sc.baseline, flat=flat)
    explainer = Explainer(sc.instance, mode=mode, client=agent if mode != "template" else None)
    parser = {"llm": ReplyParser(agent, sc.instance) if replies == "llm" else None,
              "rule": RuleReplyParser(sc.instance), "scripted": None}[replies]
    return Negotiator(sc.instance, priority=pm, explainer=explainer, ledger=ledger, semester=SEMESTER,
                      reply_parser=parser, option_source=source, option_client=agent, use_ledger=not flat,
                      time_limit=20, workers=WORKERS)


def _simulators(sc: Scenario, config: str, sim, run_seed: int):
    voiced = CONFIGS[config][3] != "scripted"
    return simulators(sc, client=sim if voiced else None, structured=not voiced, run_seed=run_seed)


def run_negotiation(sc: Scenario, config: str, shared: ConcessionLedger, agent, sim, run_seed: int = 0) -> dict:
    ledger = ConcessionLedger([*shared.entries, *sc.ledger])
    neg = _negotiator(sc, config, ledger, agent)
    out = neg.resolve(sc.constraints, _simulators(sc, config, sim, run_seed), baseline=sc.baseline, week=sc.week,
                      private=sc.private, raw_requests=sc.raw_requests)
    if out.status != "escalated":  # an escalated conflict changes nothing yet: what was agreed on the way is no burden
        shared.entries.extend(out.concessions)
    assignment = out.result.assignment if out.result and out.result.ok else None
    return _row(sc, out, assignment)


class _Given:
    """A request whose constraints are already read (as a form gives them): the scenario's own, so the
    pipeline is scored on what happens after reading."""

    def __init__(self, sc: Scenario) -> None:
        self.by_request: dict[str, list] = {}
        for c in sc.planted:
            self.by_request.setdefault(c.source.request or "R-?", []).append(c)

    def parse(self, request):
        from language.parsing import ParseOutput, ParseResult

        return ParseResult(request=request, output=ParseOutput(request_type="preference", action="compile"),
                           constraints=list(self.by_request[request.id]))


def run_pipeline(sc: Scenario, config: str, shared: ConcessionLedger, agent, sim, run_seed: int = 0) -> dict:
    """The scenario through the request pipeline the app runs (``Orchestrator``), not the negotiator alone:
    the base timetable is published, each planted request is submitted in turn with its constraints as read,
    and every change waiting for approval is approved in order. The row describes what was published."""
    from core.schemas import Channel, Request, Role
    from pipeline.orchestrator import Orchestrator
    from pipeline.store import Store

    store = Store()
    for e in [*shared.entries, *sc.ledger]:
        store.record_concession(e)
    sims = _simulators(sc, config, sim, run_seed)
    orch = Orchestrator(sc.instance, store, parser=_Given(sc), negotiator=_negotiator(sc, config, ConcessionLedger(), agent),
                        responders=sims, semester=SEMESTER)
    orch.bootstrap(sc.base, sc.baseline)
    before = len(store.ledger())
    cases = []
    for rid, cons in _Given(sc).by_request.items():
        owner = cons[0].owner
        role = (sc.instance.faculty_by_id[owner].role if owner in sc.instance.faculty_by_id else
                Role.LAB_INCHARGE if owner else Role.EXAM_CELL)
        cases.append(orch.submit(Request(id=rid, channel=Channel.PORTAL, sender_id=owner or "S-EXAM", role=role,
                                         raw_text=sc.raw_requests.get(rid, ""), received_at=datetime(2026, 8, 3))))
    # a requirement already met has no proposal: confirmed first, it publishes nothing
    for case in sorted(orch.pending(), key=lambda c: c.proposal.version if c.proposal else 0):
        orch.approve(case.id, "C-TT")
    shared.entries.extend(store.ledger()[before:])
    outcomes = [c.outcome for c in cases if c.outcome is not None]
    out = _merge(outcomes, escalated=any(c.status.value == "escalated" for c in cases))
    v = store.current_version(sc.week) or store.current_version()
    assignment = v.assignment if out.status != "escalated" and v is not None else None
    row = _row(sc, out, assignment)
    row["pipeline"] = {c.id: c.status.value for c in cases}
    return row


def _merge(outcomes: list, escalated: bool):
    """One scenario's outcome from the negotiations of its requests: escalated if any request was,
    agreed if any was settled by negotiation, else feasible."""
    from agents.negotiation import Outcome

    out = Outcome(status="escalated" if escalated else
                  "agreed" if any(o.status == "agreed" for o in outcomes) else "feasible", step=1)
    for o in outcomes:
        out.rounds += o.rounds
        out.messages += o.messages
        out.replies += o.replies
        out.relaxed += o.relaxed
        out.concessions += o.concessions
        out.constraints = o.constraints
        out.escalation = out.escalation or o.escalation
        out.option_calls += o.option_calls
        out.filtered += o.filtered
        out.proposals += o.proposals
        out.proposals_verified += o.proposals_verified
        out.clarifications += o.clarifications
    return out


def _row(sc: Scenario, out, assignment: dict[str, Placement] | None) -> dict:
    agreed = [c for c in out.constraints if c.hard]
    conceders = {c.stakeholder for c in out.concessions}
    moved = _moved(sc, assignment) if assignment else None
    # a clarification re-sends the same options: count each offer once
    offers = list({id(o): o for m in out.messages for o in m.offers}.values())
    accepted = [next((o for o in m.offers if o.key == (r.choice or "").strip().upper()), None)
                for m, r in zip(out.messages, out.replies, strict=False) if r.decision == "accept"]
    accepted = [o for o in accepted if o is not None]
    return {
        "status": out.status,
        "rounds": out.rounds,
        # candidate-level validity (H1a): MCS options are solver-verified by construction
        "offers": len(offers),
        "invalid_offers": sum(o.feasible is False for o in offers),
        "accepted_offers": len(accepted),
        "accepted_invalid": sum(o.feasible is False for o in accepted),
        # H1: what invented options cost (B3, B4)
        "option_calls": out.option_calls,
        "filtered": out.filtered,
        # tool calls made from replies (brief Section 4.3)
        "tools": dict(Counter(r.tool for r in out.replies if r.tool)),
        "tool_retries": sum(r.retries for r in out.replies),
        "proposals": out.proposals,
        "proposals_verified": out.proposals_verified,
        "clarifications": out.clarifications,
        "valid": validity(sc, assignment, agreed) if assignment else None,
        "moved": moved,
        "concessions": len(conceders),
        "conceders": sorted(conceders),
        "relaxed": original_ids(sc, out.relaxed),
        **_scored(sc, out.relaxed, moved),
        "escalated_to": out.escalation.to if out.escalation else None,
        "escalation_reason": out.escalation.reason if out.escalation else None,
        "faithfulness": mean([m.faithfulness for m in out.messages]),
        "leaks": sum(len(m.leaks) for m in out.messages),
        "rejected_claims": sum(m.rejected_claims for m in out.messages),
        "invented_times": sum(m.invented_times for m in out.messages),
        "omitted_options": sum(m.omitted_options for m in out.messages),
        "explanation_modes": dict(Counter(m.explanation_mode for m in out.messages)),
        "first_accepted": bool(out.replies) and out.replies[0].decision == "accept",
        "messages": [m.text for m in out.messages],
        # what each explanation could use and what it claimed, for the human claim labels (evaluation.study)
        "explanations": [{"to": m.to, "mode": m.explanation_mode, "facts": m.facts, "claims": m.claims}
                         for m in out.messages],
        "replies": [r.model_dump(exclude_none=True) for r in out.replies],
    }


def run_imposed(sc: Scenario, shared: ConcessionLedger) -> dict:
    """B2: the same priority model (tiers, weights, ledger), objective and
    solver as ours, but the cheapest correction set is imposed without asking
    anyone. We record whether each imposed change would have been acceptable
    to its owner, from the hidden profiles."""
    ledger = ConcessionLedger([*shared.entries, *sc.ledger])
    pm = PriorityModel(sc.instance, ledger=ledger, semester=SEMESTER, baseline=sc.baseline)
    neg = Negotiator(sc.instance, priority=pm, explainer=Explainer(sc.instance), ledger=ledger, semester=SEMESTER,
                     time_limit=20, workers=WORKERS)
    out = neg.impose(sc.constraints, baseline=sc.baseline, week=sc.week)
    shared.entries.extend(out.concessions)
    assignment = out.result.assignment if out.result and out.result.ok else None
    moved = _moved(sc, assignment) if assignment else None
    losers = {e.stakeholder for e in out.concessions}
    sims = simulators(sc)
    acceptable = []
    for owner in sorted(losers):
        sim = sims.get(owner)
        own = {s.id for c in sc.planted if c.owner == owner
               for s in sc.instance.sessions if c.scope.session == s.id or c.scope.faculty == s.faculty}
        acceptable.append(bool(sim) and all(sim.acceptable(s, assignment[s]) for s in own if s in assignment))
    return {"status": out.status, "rounds": 0, "valid": validity(sc, assignment) if assignment else None,
            "moved": moved, "concessions": len(losers), "conceders": sorted(losers),
            "relaxed": original_ids(sc, out.relaxed), **_scored(sc, out.relaxed, moved),
            "escalation_reason": out.escalation.reason if out.escalation else None,
            "imposed_acceptable": mean(acceptable) if acceptable else None}


def run_llm_only(sc: Scenario, agent) -> dict:
    inst = sc.instance
    sessions = "\n".join(
        f"{s.id}: faculty {s.faculty}, sections {','.join(s.groups)}, duration {s.duration}, "
        f"{'lab' if s.room_type.value == 'lab' else 'lecture room'}"
        + (f", needs {','.join(s.equipment)}" if s.equipment else "") for s in inst.sessions)
    rooms = "\n".join(f"{r.id}: {r.type.value}" + (f", equipment {','.join(r.equipment)}" if r.equipment else "")
                      for r in inst.rooms)
    cons = "\n".join(f"- {describe(inst, c)}" for c in sc.constraints if c.active_in(sc.week))
    prompt = (f"Days: {', '.join(inst.calendar.days)}; slots 0-{inst.calendar.slots_per_day - 1} "
              f"(slot 4 is lunch).\n\nRooms:\n{rooms}\n\nSessions:\n{sessions}\n\nConstraints:\n{cons}")
    got = agent.generate(B1_PROMPT, prompt, _Timetable)
    assignment = {p.session: Placement(day=p.day, slot=p.slot, room=p.room) for p in got.placements}
    problems = verify_timetable(inst, assignment, [c for c in sc.constraints if c.hard], week=sc.week)
    return {"status": "llm_timetable", "rounds": 0, "valid": validity(sc, assignment),
            "all_hard_ok": not problems, "moved": _moved(sc, assignment), "concessions": 0,
            "problems": problems[:5]}


def run_oracle(sc: Scenario) -> dict:
    o = sc.oracle
    return {"status": {"agree": "agreed", "escalate": "escalated"}.get(o.status, o.status), "rounds": 0,
            "valid": o.status != "escalate" or None, "moved": o.moved, "concessions": o.concessions,
            "relaxed": o.relaxed, "objective": o.objective, "tiers": o.tiers or None}


def correct_outcome(row: dict, sc: Scenario) -> bool:
    """Status agreement with the oracle: feasible without negotiation,
    resolved (agreed), or escalated."""
    want = {"feasible": {"feasible"}, "agree": {"agreed"}, "escalate": {"escalated", "failed"}}[sc.expected]
    return row["status"] in want


def objective_gaps(rows: list[dict], scenarios: dict[str, Scenario]) -> tuple[list[float], list[bool]]:
    """Gap to the oracle on the option cost, J(A_sys) - J(A*), for scenarios
    the oracle resolves and the system turned into a timetable (agreed or
    imposed); and whether the system gave up more in a higher-priority tier
    (lexicographically worse, whatever the cost gap)."""
    gaps, tier_worse = [], []
    for r in rows:
        o = scenarios[r["scenario"]].oracle
        if o.status != "agree" or o.objective is None or r.get("objective") is None:
            continue
        if r["status"] not in ("agreed", "imposed"):
            continue
        gaps.append(r["objective"] - o.objective)
        tier_worse.append(list(r["tiers"]) > list(o.tiers))
    return gaps, tier_worse


def _spread(ledgers: list[ConcessionLedger], people: list[str] = STAKEHOLDERS) -> dict:
    """Burden dispersion per run's ledger (one semester per seed), averaged."""
    per = [led.dispersion(people, SEMESTER) for led in ledgers]
    return {k: mean([p.get(k) for p in per]) for k in ("gini", "max", "cv", "total")} if per else {}


def summarise(rows: list[dict], scenarios: dict[str, Scenario], ledgers: list[ConcessionLedger] | None) -> dict:
    need_agree = [r for r in rows if scenarios[r["scenario"]].expected == "agree"]
    resolved = [r for r in need_agree if r["status"] == "agreed"]
    gaps, tier_worse = objective_gaps(rows, scenarios)
    escalate_gold = [scenarios[r["scenario"]].expected == "escalate" for r in rows]
    escalate_pred = [r["status"] in ("escalated", "failed") for r in rows]
    tp = sum(g and p for g, p in zip(escalate_gold, escalate_pred, strict=True))
    produced = [r["valid"] for r in rows if r.get("valid") is not None]
    # the four faculty of the small departments, plus anyone else the run's conflicts involve (hard tier)
    owners = {p.owner for r in rows for p in scenarios[r["scenario"]].profiles}
    spread = _spread(ledgers or [], sorted(owners | set(STAKEHOLDERS)))
    tools = sum((Counter(r.get("tools", {})) for r in rows), Counter())
    return {
        "n": len(rows),
        "seeds": sorted({r.get("seed", 0) for r in rows}),
        "correct_outcome": mean([correct_outcome(r, scenarios[r["scenario"]]) for r in rows]),
        "validity_of_timetables": mean(produced),  # final repairs; 1 - this is the final violation rate
        "timetables_produced": len(produced),
        # candidate-level validity (H1a): share of offers the solver rejects, share of accepted
        # offers the final re-solve rejects, and invalid offers per resolved scenario
        "invalid_candidate_rate": (sum(r.get("invalid_offers", 0) for r in rows)
                                   / sum(r.get("offers", 0) for r in rows)) if sum(r.get("offers", 0) for r in rows) else None,
        "solver_rejection_rate": (sum(r.get("accepted_invalid", 0) for r in rows)
                                  / sum(r.get("accepted_offers", 0) for r in rows))
        if sum(r.get("accepted_offers", 0) for r in rows) else None,
        "invalid_candidates_per_resolved": mean([r.get("invalid_offers", 0) for r in resolved]),
        # H1: invention calls (B3: one per round; B4: with regenerations) and filtered inventions
        "option_calls_per_scenario": mean([r.get("option_calls", 0) for r in rows if r.get("rounds")]),
        "filtered_per_scenario": mean([r.get("filtered", 0) for r in rows if r.get("rounds")]),
        # tool calls from replies
        "tool_calls": dict(tools),
        "tool_retries": sum(r.get("tool_retries", 0) for r in rows),
        "proposals_verified": f"{sum(r.get('proposals_verified', 0) for r in rows)}"
                              f"/{sum(r.get('proposals', 0) for r in rows)}",
        "clarifications": sum(r.get("clarifications", 0) for r in rows),
        "agreement_rate": mean([r["status"] == "agreed" for r in need_agree]),  # resolution rate
        "rounds_mean": mean([r["rounds"] for r in resolved]),  # over resolved scenarios only
        "rounds_median": statistics.median([r["rounds"] for r in resolved]) if resolved else None,
        "rounds_ci": bootstrap_ci([r["rounds"] for r in resolved]),
        # reported beside rounds so a system cannot look efficient by escalating hard cases early
        "escalation_rate": mean([r["status"] in ("escalated", "failed") for r in rows]),
        # |gap| is the distance H1c compares: an imposed repair (B2) can undercut the
        # oracle only by forcing changes owners would not accept (see imposed_acceptable)
        "objective_gap_abs_mean": mean([abs(g) for g in gaps]),
        "objective_gap_mean": mean(gaps),
        "objective_gap_ci": bootstrap_ci(gaps),
        "tier_worse_than_oracle": mean(tier_worse),
        "escalation_precision": tp / sum(escalate_pred) if sum(escalate_pred) else None,
        "escalation_recall": tp / sum(escalate_gold) if sum(escalate_gold) else None,
        "escalation_reasons": dict(Counter(r["escalation_reason"] for r in rows if r.get("escalation_reason"))),
        "concession_gini": spread.get("gini"),
        "burden_max": spread.get("max"),
        "burden_cv": spread.get("cv"),
        "burden_total": spread.get("total"),
        "first_proposal_acceptance": mean([r.get("first_accepted") for r in rows if r.get("rounds")]),
        "explanation_faithfulness": mean([r.get("faithfulness") for r in rows]),
        "private_leaks": sum(r.get("leaks", 0) for r in rows),
        # H3: does the prose agree with the options listed under it?
        "invented_times": sum(r.get("invented_times", 0) for r in rows),
        "omitted_options": sum(r.get("omitted_options", 0) for r in rows),
        "explanation_modes": dict(sum((Counter(r.get("explanation_modes", {})) for r in rows), Counter())),
        "imposed_acceptable": mean([r.get("imposed_acceptable") for r in rows]),
        "status": dict(Counter(r["status"] for r in rows)),
        "wall_s": sum(r["wall_s"] for r in rows),
    }


def _run_one(config: str, sc: Scenario, shared: ConcessionLedger, agent, sim, seed: int, pipeline: bool) -> dict:
    if config == "B1":
        return run_llm_only(sc, agent)
    if config == "B2":
        return run_imposed(sc, shared)
    if config == "oracle":
        return run_oracle(sc)
    return (run_pipeline if pipeline else run_negotiation)(sc, config, shared, agent, sim, run_seed=seed)


def compare(a: list[dict], b: list[dict], scenarios) -> dict:
    """Paired tests of configuration a against b on the same scenarios."""
    def key(r):
        return r["scenario"], r.get("seed", 0)

    ka, kb = {key(r): r for r in a}, {key(r): r for r in b}
    ids = [s for s in ka if s in kb]  # paired on scenario and seed
    ok_a = [correct_outcome(ka[s], scenarios[s[0]]) for s in ids]
    ok_b = [correct_outcome(kb[s], scenarios[s[0]]) for s in ids]
    agree = [s for s in ids if scenarios[s[0]].expected == "agree"]
    both = [s for s in agree if ka[s]["status"] == kb[s]["status"] == "agreed"]  # rounds per resolved conflict
    scored = [s for s in agree if ka[s].get("objective") is not None and kb[s].get("objective") is not None]
    return {"pairs": len(ids),
            "correct_outcome_mcnemar": mcnemar(ok_a, ok_b),
            "resolved_mcnemar": mcnemar([ka[s]["status"] == "agreed" for s in agree],
                                        [kb[s]["status"] == "agreed" for s in agree]),
            "rounds_wilcoxon": wilcoxon([ka[s]["rounds"] for s in both], [kb[s]["rounds"] for s in both]),
            "objective_wilcoxon": wilcoxon([ka[s]["objective"] for s in scored],
                                           [kb[s]["objective"] for s in scored])}


def stratified(scenarios: list[Scenario], n: int) -> list[Scenario]:
    """A deterministic subset of ``n`` scenarios spread evenly over (kind,
    family): one from each stratum in turn, keeping the file order."""
    strata: dict[tuple[str, str], list[Scenario]] = {}
    for sc in scenarios:
        strata.setdefault((sc.kind, sc.family or ""), []).append(sc)
    picked: set[str] = set()
    while len(picked) < min(n, len(scenarios)):
        for group in strata.values():
            left = [s for s in group if s.id not in picked]
            if left and len(picked) < n:
                picked.add(left[0].id)
    return [s for s in scenarios if s.id in picked]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios.jsonl"))
    ap.add_argument("--configs", default=None, help="comma-separated; default: all offline configs")
    ap.add_argument("--plan", action="store_true",
                    help=f"the reduced experiment plan: {','.join(PLAN)}, seeds {SEEDS} for {','.join(PRIMARY)}")
    ap.add_argument("--offline", action="store_true", help="only configurations that need no API")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated scenario ids")
    ap.add_argument("--agent-model", default=None,
                    help="model for explanations, reply parsing, invented options and B1 (default: agent model); "
                         "use the simulator model for development when the agent quota is spent")
    ap.add_argument("--local", action="store_true",
                    help="run the agent (explanations, reply parsing, invented options, B1) on the local "
                         "model behind an OpenAI-compatible server instead of Gemini")
    ap.add_argument("--sim-local", action="store_true", help="run the stakeholder simulators locally too")
    ap.add_argument("--local-url", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None, help="model to run on a router-mode server (default NTS_LOCAL_MODEL)")
    ap.add_argument("--local-adapter", default=None,
                    help="LoRA adapter on the local server's base model for the agent and local simulators "
                         "(its GGUF name, e.g. nts-read100); the same one for every configuration")
    ap.add_argument("--local-max-tokens", type=int, default=2048)
    ap.add_argument("--subset", type=int, default=None,
                    help="run every configuration on a stratified subset of this many scenarios (Tier 2)")
    ap.add_argument("--ablation-subset", type=int, default=ABLATION_SUBSET,
                    help="stratified subset for the ablations A1-A3 (0: all scenarios)")
    ap.add_argument("--b1-subset", type=int, default=B1_SUBSET,
                    help="stratified subset for B1, the LLM writing whole timetables (0: all scenarios)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated run seeds for the stakeholders' stochastic replies; only "
                         f"{','.join(PRIMARY)} (and offline ours) repeat over them, the rest use the first "
                         f"(default: {','.join(map(str, SEEDS))} with --plan, else 0)")
    ap.add_argument("--ablation-all-seeds", action="store_true",
                    help="repeat A1-A3 (and A1-offline) over every seed too, so each pairs with ours seed by seed")
    ap.add_argument("--pipeline", action="store_true",
                    help="run the negotiating configurations through the request pipeline the app runs "
                         "(Orchestrator: submit, solve and negotiate, approve) instead of the negotiator alone")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in <out>.rows.jsonl (each finished row is appended there) and run "
                         "only the rest; a run stopped by the daily API quota continues where it stopped")
    args = ap.parse_args()

    scenarios = load(args.scenarios)
    if args.only:
        wanted = set(args.only.split(","))
        scenarios = [s for s in scenarios if s.id in wanted]
    scenarios = scenarios[: args.limit]
    if args.subset:
        scenarios = stratified(scenarios, args.subset)
    by_id = {s.id: s for s in scenarios}
    configs = args.configs.split(",") if args.configs else PLAN if args.plan else OFFLINE
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS if args.plan else [0]
    if args.offline:
        configs = [c for c in configs if c in OFFLINE]
    agent = sim = None
    if any(c in LLM for c in configs):
        local = None
        if args.local or args.sim_local:
            from language.local import LocalClient

            local = LocalClient(base_url=args.local_url, max_tokens=args.local_max_tokens, timeout=600,
                                server_model=args.local_model, adapter=args.local_adapter)
        agent = local if args.local else GeminiClient(args.agent_model or default_model("agent"))
        sim = local if args.sim_local else GeminiClient(default_model("simulator"))

    report: dict = {"scenarios": str(args.scenarios), "pipeline": args.pipeline, "configs": {}, "seeds": seeds,
                    "subset": args.subset, "ablation_subset": args.ablation_subset, "b1_subset": args.b1_subset,
                    "models": {"agent": getattr(agent, "model", None), "simulator": getattr(sim, "model", None)}}
    all_rows: dict[str, list[dict]] = {}
    out = args.out or Path("runs") / f"negotiation-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".rows.jsonl")  # every finished row, as it finishes
    done: dict[tuple, dict] = {}
    if args.resume and log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["config"], r["seed"], r["scenario"])] = r
        print(f"resuming: {len(done)} rows from {log}", flush=True)
    elif log.exists():
        log.unlink()
    stopped = None
    for config in configs:
        runs = scenarios
        if config in ("A1", "A2", "A3") and args.ablation_subset:
            runs = stratified(scenarios, args.ablation_subset)
        elif config == "B1" and args.b1_subset:
            runs = stratified(scenarios, args.b1_subset)
        rows, ledgers = [], []
        repeat = [*PRIMARY, "ours"] + (["A1", "A2", "A3", "A1-offline"] if args.ablation_all_seeds else [])
        for seed in seeds if config in repeat else seeds[:1]:
            shared = ConcessionLedger()  # one semester of conflicts per seed
            ledgers.append(shared)
            for sc in runs:
                if (kept := done.get((config, seed, sc.id))) is not None:
                    # the semester so far: later scenarios meet the concessions this one recorded
                    shared.entries.extend(ConcessionEntry(**e) for e in kept["ledger_added"])
                    rows.append(kept)
                    continue
                if stopped is not None:
                    continue
                t = time.perf_counter()
                n0 = len(shared.entries)
                try:
                    row = _run_one(config, sc, shared, agent, sim, seed, args.pipeline)
                except DailyQuotaReached as e:
                    stopped = str(e)
                    print(f"stopped: {e}. Run again with --resume when the quota is back.", flush=True)
                    continue
                row.update(scenario=sc.id, seed=seed, kind=sc.kind, family=sc.family, expected=sc.expected,
                           wall_s=time.perf_counter() - t, config=config,
                           ledger_added=[e.model_dump(mode="json") for e in shared.entries[n0:]])
                with log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                rows.append(row)
                print(f"[{config} s{seed}] {sc.id} expected={sc.expected} got={row['status']} "
                      f"rounds={row['rounds']} valid={row.get('valid')}", flush=True)
        all_rows[config] = rows
        uses_ledger = config in CONFIGS or config == "B2"
        families = sorted({r["family"] for r in rows if r["family"]})
        report["configs"][config] = {
            "summary": summarise(rows, by_id, ledgers if uses_ledger else None),
            # burden is a property of the whole run's ledger, so it is reported pooled only
            "by_family": {f: summarise([r for r in rows if r["family"] == f], by_id, None) for f in families},
            "rows": rows,
        }
    if "ours" in all_rows or "ours-llm" in all_rows:
        ref = "ours-llm" if "ours-llm" in all_rows else "ours"
        report["paired"] = {f"{ref} vs {c}": compare(all_rows[ref], all_rows[c], by_id)
                            for c in all_rows if c != ref}
    for c, v in report["configs"].items():
        print(f"\n== {c}\n" + json.dumps(v["summary"], indent=1, default=str))
    if "paired" in report:
        print("\npaired:", json.dumps(report["paired"], indent=1))
    if stopped is not None:
        report["stopped"] = stopped  # a partial report: the rows log has what finished
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}" + (" (partial: run again with --resume)" if stopped else ""))
    if stopped is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
