"""Run a suite of scenarios with one model, judge each episode, and record the verdicts as telemetry.

Each scenario runs on a fork of the same store, so every one starts from the same state (plus the
scenario's own setup), with a fresh copy of its cast to answer the agent's asks. A submitted
draft is approved (committed through the kernel) so the checks can look at the committed state; the
approval and the verdict are both recorded on the episode's trace ("approval", "evaluation" spans),
which is what labels the episode for training.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from nts.harness import AgentConfig, Harness, approve
from nts.models import Model
from nts.state import Store
from nts.telemetry import Tracer
from nts.tools import ALL_TOOLS, Registry

from .scenarios import Scenario


@dataclass
class Verdict:
    scenario: str
    family: str
    episode_id: str
    status: str
    passed: bool
    failures: list[str]
    turns: int
    tool_calls: int
    refused: int
    tokens: int
    seconds: float


def run_suite(suite: list[Scenario], model: Model, store: Store, agent: AgentConfig, tracer: Tracer,
              *, progress: bool = False, handbook=None) -> list[Verdict]:
    harness = Harness(Registry(ALL_TOOLS), model, tracer, handbook=handbook)
    verdicts = []
    for i, sc in enumerate(suite, 1):
        fork = store.fork()
        origin = fork.version  # the store's own version: a setup commit gives the episode one the store never had
        if sc.setup:
            draft = fork.draft("system", f"setup-{sc.name}")
            for ch in sc.setup:
                draft.add(ch)
            fork.commit(draft)
        before = fork.world
        out = harness.run(agent, sc.request, fork, sc.actor, sc.week, people=copy.deepcopy(sc.cast))
        after = approve(fork, out, tracer, by="eval") if out.status == "submitted" else None
        failures = sc.judge(out, before, after)
        v = Verdict(sc.name, sc.family, out.episode_id, out.status, not failures, failures, out.turns, out.tool_calls,
                    out.refused, out.usage.prompt_tokens + out.usage.completion_tokens, out.seconds)
        with tracer.span("evaluation", sc.family, trace_id=out.episode_id, scenario=sc.name, family=sc.family,
                         passed=v.passed, failures=failures, expect=sorted(sc.expect), store_version=origin):
            pass
        verdicts.append(v)
        if progress:
            mark = "ok  " if v.passed else "FAIL"
            print(f"[{i}/{len(suite)}] {mark} {sc.name}: {out.status}; {'; '.join(failures)}", flush=True)
    return verdicts


def summary(verdicts: list[Verdict]) -> dict:
    by: dict[str, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by[v.family].append(v)

    def stats(vs: list[Verdict]) -> dict:
        n = len(vs)
        return {"n": n, "passed": sum(v.passed for v in vs), "rate": round(sum(v.passed for v in vs) / n, 3) if n else 0,
                "turns": round(sum(v.turns for v in vs) / n, 2) if n else 0,
                "refused": sum(v.refused for v in vs), "tokens": sum(v.tokens for v in vs),
                "seconds": round(sum(v.seconds for v in vs), 1)}

    return {"all": stats(verdicts), "families": {f: stats(vs) for f, vs in sorted(by.items())}}


def save(path: Path, model: str, verdicts: list[Verdict]) -> dict:
    report = {"model": model, "summary": summary(verdicts), "verdicts": [asdict(v) for v in verdicts]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
