"""System One benchmark: every classifier on every test set, with the same numbers for each.

    uv run python -m evaluation.system_one                         # all models, all settings
    uv run python -m evaluation.system_one --models logreg,hybrid --settings para,noisy
    uv run python -m evaluation.system_one --no-intent-labels      # trained on the request corpus alone

Settings (training data -> test set); tau is tuned on the val split of the training data:

* ``corpus``: templates + paraphrases -> the template test split
* ``para``: templates + paraphrases -> the paraphrased test split
* ``unseen-wording``: templates only -> the paraphrased test split (no paraphrase seen in training)
* ``intents`` / ``typo`` / ``noisy``: templates + paraphrases -> the 417-message intent test set
  (``data/compiler-intents/test.jsonl``), and the same with typos / noisy English (``data/suites/``)

Every setting also trains on the kinds the corpus is missing (``system_one.intent_examples``: swaps,
cancellations, extra classes, ...) unless ``--no-intent-labels``.

The intent sets are labelled for the intent reader, not for System One, so they are scored by what
the routing needs. The kinds the request corpus has (preference, unavailability, room_issue,
clash_report, policy_question, out_of_scope) give the accuracies, a predicted ``deny`` counting as
``compile`` (the intent data leaves denial to the policy agent). The kinds it is missing
(``INTENT_TYPES``) must stay off the fast path, which cannot read them: ``fast_path_leak`` is the share
of them sent there, and ``new_kind_type_accuracy`` how many get their own type.

Writes ``runs/system_one-<date>.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from sklearn.metrics import f1_score

from language.corpus import CorpusExample, ExpectedAction, load_jsonl
from language.system_one import INTENT_TYPES, MODELS, Decision, Labelled, SystemOne, intent_examples, read_intents, tune_tau

from .metrics import ece, percentile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INTENT_SETS = {"intents": DATA / "compiler-intents" / "test.jsonl",
               "typo": DATA / "suites" / "test-typo.jsonl",
               "noisy": DATA / "suites" / "test-noisy.jsonl"}
SETTINGS = ("corpus", "para", "unseen-wording", *INTENT_SETS)


def from_corpus(examples: Sequence[CorpusExample]) -> list[Labelled]:
    return [Labelled(e.request, e.request_type, e.expected_action, e.request_type.value, e.split) for e in examples]


def _routing_action(a: ExpectedAction, scoring_intents: bool) -> ExpectedAction:
    return ExpectedAction.COMPILE if scoring_intents and a == ExpectedAction.DENY else a


def score(model: SystemOne, tau: float, cases: Sequence[Labelled], intents: bool = False) -> dict:
    decisions: list[Decision] = [model.decide(c.request) for c in cases]
    known = [(d, c) for d, c in zip(decisions, cases, strict=True)
             if c.expected_action is not None and c.kind not in INTENT_TYPES]
    new = [(d, c) for d, c in zip(decisions, cases, strict=True) if c.kind in INTENT_TYPES]
    type_ok = [d.request_type == c.request_type for d, c in known]
    action_ok = [_routing_action(d.action, intents) == c.expected_action for d, c in known]
    routed = [t and a for (d, _), t, a in zip(known, type_ok, action_ok, strict=True) if d.fast_path(tau)]
    gold = [c.expected_action.value for _, c in known]
    latencies = [d.latency_ms for d in decisions]

    def share(n: int) -> float | None:  # None on a set of new kinds only
        return n / len(known) if known else None

    return {
        "n": len(cases),
        "type_accuracy": share(sum(type_ok)),
        "action_accuracy": share(sum(action_ok)),
        "action_macro_f1": f1_score(gold, [_routing_action(d.action, intents).value for d, _ in known],
                                    labels=sorted(set(gold)), average="macro", zero_division=0) if known else None,
        "action_ece": ece([d.action_p for d, _ in known], action_ok) if known else None,
        "fast_path_share": share(sum(d.fast_path(tau) for d, _ in known)),
        "fast_path_precision": sum(routed) / len(routed) if routed else None,
        "new_kinds": len(new),
        "new_kind_type_accuracy": sum(d.request_type == c.request_type for d, c in new) / len(new) if new else None,
        "fast_path_leak": sum(d.fast_path(tau) for d, _ in new) / len(new) if new else None,
        "leaked_kinds": dict(Counter(c.kind for d, c in new if d.fast_path(tau))),
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def fit(name: str, examples: Sequence[CorpusExample | Labelled]) -> tuple[SystemOne, float, float]:
    """A model trained on the train split, its tau tuned on the val split, and the seconds it took."""
    t = time.perf_counter()
    model = SystemOne(model=name).fit([e for e in examples if e.split == "train"])
    val = [e for e in examples if e.split == "val"]
    tau = float(tune_tau([model.decide(e.request) for e in val], val))
    return model, tau, time.perf_counter() - t


def benchmark(models: Sequence[str] = MODELS, settings: Sequence[str] = SETTINGS,
              templates: Sequence[CorpusExample] | None = None,
              paraphrases: Sequence[CorpusExample] | None = None,
              intent_labels: bool = True) -> dict[str, dict[str, dict]]:
    """{model: {setting: metrics}}. ``templates`` / ``paraphrases`` default to the corpus files."""
    templates = list(templates if templates is not None else load_jsonl(DATA / "requests.jsonl"))
    paraphrases = list(paraphrases if paraphrases is not None else load_jsonl(DATA / "requests-para.jsonl"))
    extra = intent_examples("train") + intent_examples("val") if intent_labels else []
    tests = {"corpus": from_corpus([e for e in templates if e.split == "test"]),
             "para": from_corpus([e for e in paraphrases if e.split == "test"])}
    tests["unseen-wording"] = tests["para"]
    for s in settings:
        if s in INTENT_SETS:
            tests[s] = read_intents(INTENT_SETS[s], "test")
    out: dict[str, dict[str, dict]] = {}
    for name in models:
        both = fit(name, templates + paraphrases + extra)
        only = fit(name, templates + extra) if "unseen-wording" in settings else None
        out[name] = {}
        for s in settings:
            model, tau, seconds = only if s == "unseen-wording" else both
            out[name][s] = {"tau": tau, "fit_seconds": round(seconds, 2),
                            **score(model, tau, tests[s], intents=s in INTENT_SETS)}
    return out


COLUMNS = [("type_accuracy", "type"), ("action_accuracy", "action"), ("action_macro_f1", "macroF1"),
           ("action_ece", "ECE"), ("fast_path_share", "fast"), ("fast_path_precision", "fastPrec"),
           ("new_kind_type_accuracy", "newType"), ("fast_path_leak", "leak"), ("tau", "tau"),
           ("latency_ms_p50", "ms")]


def table(results: dict[str, dict[str, dict]]) -> str:
    lines = []
    settings = list(next(iter(results.values())))
    for s in settings:
        lines.append(f"\n{s} (n={results[next(iter(results))][s]['n']})")
        lines.append(f"  {'model':<8}" + "".join(f"{h:>9}" for _, h in COLUMNS))
        for name, by in results.items():
            m = by[s]
            cells = ["-" if m[k] is None else (f"{m[k]:.1f}" if k.startswith("latency") else f"{m[k]:.3f}")
                     for k, _ in COLUMNS]
            lines.append(f"  {name:<8}" + "".join(f"{c:>9}" for c in cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(MODELS), help=f"comma-separated, from {', '.join(MODELS)}")
    ap.add_argument("--settings", default=",".join(SETTINGS), help=f"comma-separated, from {', '.join(SETTINGS)}")
    ap.add_argument("--no-intent-labels", action="store_true", help="train on the request corpus alone")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / f"system_one-{date.today():%Y%m%d}.json")
    args = ap.parse_args(argv)
    results = benchmark(args.models.split(","), args.settings.split(","), intent_labels=not args.no_intent_labels)
    print(table(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
