"""More training rows from the local model's own mistakes, judged by the code that makes the gold.

    uv run python -m training.mine intents --local URL --local-model Qwen3.5-4B-Finetune-NTS-Read100-Q8_0
    uv run python -m training.mine policy --local URL --local-model Qwen3.5-4B-Finetune-NTS-Read100-Q8_0
    uv run python -m training.distill export --readable --compiler-dir data/compiler-intents \\
        --out data/multitask-readable                             # the mined rows join the training set

* ``intents``: a fresh pool from the intent generators (``language.intent_corpus``, new seeds, never a
  test wording) is read by the model and scored as ``evaluation.intents`` scores it. Each request it
  gets wrong is kept with its gold, with ``--twins`` more of the same wording (other classes, days,
  times) and ``--contrast`` of the same generator whose gold differs in a field the model got wrong (a
  must beside a wish, a morning beside an afternoon). Writes ``data/compiler-intents/mined.jsonl``.
* ``policy``: fresh rule-breaking requests and their legal look-alikes (``extra_denials``, a new seed),
  shown with the gold parse, a misread one or none, as ``distill denials`` shows them. ``check_policy``
  judges the model's answer; each one it gets wrong is kept with the canonical answer (``distill
  denials``) as target, with ``--twins`` of the same rule and verdict and ``--contrast`` look-alikes of
  the other verdict. Writes ``data/distill/mined-policy.jsonl``.

The gold always comes from the generator or the corpus label, never from the model, and nothing is
mined from a val or test request. A report of every failure and the fields it got wrong goes to
``runs/``. Model answers are cached (``runs/llm_cache``), so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from agents.policy import SYSTEM_PROMPT as POLICY_PROMPT
from agents.policy import PolicyAgent, PolicyOutput, load_handbook, query_hints
from core.instance import Instance
from evaluation.intents import Reader, score
from language.adapters import add_args, local_client, registry
from language.compiler import gold_output
from language.corpus import CorpusExample, ExpectedAction, core_text, extra_denials, load_jsonl
from language.intent_compiler import export_rows
from language.intent_corpus import GENERATORS, TEST_TEMPLATE, TRAIN_ONLY, IntentExample, _examples
from language.llm import LLMError
from language.parsing import postprocess
from training.distill import DIR, _chat, _read, _write, canonical_answers, check_policy, misread

TRAIN_TEMPLATES = tuple(t for t in range(5) if t != TEST_TEMPLATE)


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


def wrong_fields(pred, gold, path: str = "") -> list[str]:
    """Where a predicted ``IntentOutput`` (as JSON) differs from the gold: ``intents.0.hard``,
    ``intents.0.time.before``; ``*`` when there is no prediction."""
    if isinstance(pred, dict) and isinstance(gold, dict):
        return [w for k in sorted(set(pred) | set(gold))
                for w in wrong_fields(pred.get(k), gold.get(k), f"{path}.{k}" if path else k)]
    if isinstance(pred, list) and isinstance(gold, list) and len(pred) == len(gold):
        return [w for i, (a, b) in enumerate(zip(pred, gold)) for w in wrong_fields(a, b, f"{path}.{i}")]
    return [] if pred == gold else [path or "*"]


def field(out, path: str):
    for k in path.split("."):
        if isinstance(out, list):
            out = out[int(k)] if k.isdigit() and int(k) < len(out) else None
        elif isinstance(out, dict):
            out = out.get(k)
        else:
            return None
    return out


def intent_pool(instance: Instance, per_generator: int, seed: int) -> list[tuple[str, IntentExample]]:
    """``(wording key, example)``: every generator on its train wordings and every train-only one, with
    IDs that cannot collide with the exported data's."""
    rng = random.Random(seed)
    made = _examples(instance, rng, {k: (k, g) for k, g in GENERATORS.items()}, per_generator, None, TRAIN_TEMPLATES)
    made += _examples(instance, rng, TRAIN_ONLY, per_generator, None)
    out = []
    for ex in made:
        key = ex.id.removeprefix("I-").rsplit("-", 1)[0]
        new_id = f"M{seed}-{ex.id}"
        out.append((key, replace(ex, id=new_id, split="train", request=ex.request.model_copy(update={"id": new_id}))))
    return out


def mine_intents(instance: Instance, client, *, pool: int, bank: int, twins: int, contrast: int, seed: int,
                 held: set[str], workers: int) -> tuple[list[dict], dict]:
    found = [(k, ex) for k, ex in intent_pool(instance, pool, seed) if core_text(ex.request.raw_text) not in held]
    key_of = {ex.id: k for k, ex in found}
    by_id = {ex.id: ex for _, ex in found}
    rows = [row for _, row in export_rows(instance, [], [ex for _, ex in found])]
    report = score(rows, instance, Reader("local", instance, client), workers)
    failures = [r for r in report["rows"] if not r["match"]]

    spare: dict[str, list[tuple[str, IntentExample]]] = defaultdict(list)  # same generator: to draw twins from
    for k, ex in intent_pool(instance, bank, seed + 1):
        if core_text(ex.request.raw_text) not in held:
            spare[ex.generator].append((k, ex))
    used: set[str] = set()
    chosen: list[tuple[IntentExample, dict]] = []
    fields: Counter = Counter()
    for f in failures:
        ex = by_id[f["id"]]
        wrong = wrong_fields(f["pred"], f["gold"])
        fields.update(w.split(".", 2)[-1] if w.startswith("intents.") else w for w in wrong)
        f["wrong_fields"] = wrong
        tag = {"from": ex.id, "fields": wrong}
        chosen.append((ex, {**tag, "as": "failure"}))
        free = [(k, b) for k, b in spare[ex.generator] if b.id not in used]
        same = [b for k, b in free if k == key_of[ex.id] and b.template == ex.template]
        other = [b for k, b in free if b not in same
                 and any(field(_gold(b), w) != field(f["gold"], w) for w in wrong if w != "*")]
        for b, role in [*((b, "variant") for b in same[:twins]), *((b, "contrast") for b in other[:contrast])]:
            used.add(b.id)
            chosen.append((b, {**tag, "as": role}))
    out = []
    for (ex, tag), (_, row) in zip(chosen, export_rows(instance, [], [ex for ex, _ in chosen])):
        out.append({**row, "source": "mined", "mined": tag})
    summary = {"pool": len(rows), "failures": len(failures), "outcome_match": report["summary"]["outcome_match"],
               "rows": len(out), "as": dict(Counter(r["mined"]["as"] for r in out)),
               "failures_by_kind": dict(Counter(f["kind"] for f in failures).most_common()),
               "wrong_fields": dict(fields.most_common())}
    return out, {"summary": summary, "by_kind": report["by_kind"], "failures": failures}


def _gold(ex: IntentExample) -> dict:
    return ex.gold.model_dump(mode="json", exclude_defaults=True)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _rule(instance: Instance, ex: CorpusExample) -> str:
    """The rule a request breaks, or the one its legal look-alike stays inside."""
    return ex.violates_rule or ("P-MAXCONSEC" if len(ex.targets[0].when.slots) > 1 else "P-LUNCH")


def mine_policy(instance: Instance, client, handbook: Path, answers: dict[str, str], *, pool: int, twins: int,
                contrast: int, seed: int, held: list[CorpusExample], workers: int) -> tuple[list[dict], dict]:
    agent = PolicyAgent(load_handbook(handbook), client, instance)
    made = [ex.model_copy(update={"id": f"M{seed}-{ex.id}"})
            for ex in extra_denials(instance, pool, seed=seed, exclude=held, near_miss=0.5)]
    reasons: Counter = Counter()
    lock = threading.Lock()

    def view(ex: CorpusExample, shown: str) -> tuple[str, set[str]] | None:
        """The policy prompt for ``ex`` with its parse shown as ``shown``, and the rules it shows."""
        gold = gold_output(ex, instance)
        if gold is None:
            return None
        parse = postprocess(instance, ex.request, gold)
        if shown == "misread":
            parse = misread(instance, parse, ex.violates_rule)
        elif shown == "none":
            parse = None
        text = ex.request.raw_text
        retrieved = agent.retrieve(f"{text} {query_hints(instance, parse)}")
        ids = {r.id for r in retrieved}
        return (agent.build_prompt(text, parse, retrieved), ids) if set(ex.rules) <= ids else None

    jobs = [(ex, s) for ex in made for s in ("gold", "none", *(("misread",) if ex.violates_rule else ()))]

    def run(job: tuple[CorpusExample, str]) -> dict | None:
        ex, shown = job
        v = view(ex, shown)
        if v is None:
            with lock:
                reasons["gold rule not retrieved"] += 1
            return None
        prompt, ids = v
        try:
            out = client.generate(POLICY_PROMPT, prompt, PolicyOutput)
            target, why = check_policy(ex, out, ids)
            got = out.model_dump(exclude_none=True)
        except (LLMError, ValueError) as e:
            target, why, got = None, "invalid answer", str(e)[:120]
        with lock:
            reasons[why] += 1
        if target is not None:
            return None
        return {"id": ex.id, "shown": shown, "rule": _rule(instance, ex), "gold": ex.expected_action.value,
                "why": why, "text": ex.request.raw_text, "pred": got}

    with ThreadPoolExecutor(max(1, workers)) as pool_:
        failures = [f for f in pool_.map(run, jobs) if f]

    by_id = {ex.id: ex for ex in made}
    used: set[str] = {f["id"] for f in failures}
    out: list[dict] = []

    def add(ex: CorpusExample, shown: str, tag: dict) -> bool:
        key = ex.violates_rule if ex.expected_action == ExpectedAction.DENY else "allowed"
        v = view(ex, shown)
        if v is None or key not in answers:
            return False
        out.append({"id": f"{ex.id}-{shown}-{tag['as']}", "task": "policy", "split": "train", "mined": tag,
                    "messages": _chat(POLICY_PROMPT, v[0], answers[key])})
        return True

    for f in failures:
        ex = by_id[f["id"]]
        tag = {"from": f"{ex.id}-{f['shown']}", "why": f["why"]}
        add(ex, f["shown"], {**tag, "as": "failure"})
        for role, n, same_verdict in (("variant", twins, True), ("contrast", contrast, False)):
            for b in made:
                if n == 0:
                    break
                if (b.id in used or _rule(instance, b) != f["rule"]
                        or (b.expected_action == ex.expected_action) != same_verdict):
                    continue
                shown = f["shown"] if f["shown"] != "misread" or b.violates_rule else "gold"
                if add(b, shown, {**tag, "as": role}):
                    used.add(b.id)
                    n -= 1
    summary = {"pool": len(made), "prompts": len(jobs), "failures": len(failures), "rows": len(out),
               "as": dict(Counter(r["mined"]["as"] for r in out)), "outcomes": dict(reasons.most_common()),
               "failures_by": dict(Counter(f"{f['rule']} {f['gold']} ({f['shown']} parse)"
                                           for f in failures).most_common())}
    return out, {"summary": summary, "failures": failures}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["intents", "policy"])
    ap.add_argument("--local", metavar="URL", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None, help="model on a router-mode server (default NTS_LOCAL_MODEL)")
    ap.add_argument("--pool", type=int, default=40,
                    help="intents: requests per generator; policy: rule-breaking requests and look-alikes")
    ap.add_argument("--bank", type=int, default=40, help="intents: spare requests per generator to draw twins from")
    ap.add_argument("--twins", type=int, default=2, help="more of the same wording (rule and verdict) per failure")
    ap.add_argument("--contrast", type=int, default=2,
                    help="per failure, look-alikes that differ where the model went wrong")
    ap.add_argument("--seed", type=int, default=7, help="pool seed (the exported data uses 0, 1, 2 and 100)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", type=Path, default=Path("data/compiler-intents"))
    ap.add_argument("--corpus", type=Path, nargs="+",
                    default=[Path("data/requests.jsonl"), Path("data/requests-para.jsonl")])
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--handbook", type=Path, default=Path("data/handbook.md"))
    add_args(ap)
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/compiler-intents/mined.jsonl or data/distill/mined-policy.jsonl")
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    client = local_client("compile" if args.stage == "intents" else "policy", base_url=args.local,
                          server_model=args.local_model, adapter=args.adapter, reg=registry(args.adapters))
    name = client.model.split(":")[-1].removesuffix(".gguf")
    corpora = [load_jsonl(p, instance) for p in args.corpus]
    held = [e for c in corpora for e in c if e.split in ("val", "test")]
    if args.stage == "intents":
        texts = {core_text(r["request"]["raw_text"]) for s in ("val", "test") for r in _read(args.data / f"{s}.jsonl")}
        texts |= {core_text(t) for e in held for t in (e.request.raw_text, e.template_text) if t}
        rows, report = mine_intents(instance, client, pool=args.pool, bank=args.bank, twins=args.twins,
                                    contrast=args.contrast, seed=args.seed, held=texts, workers=args.workers)
        out = args.out or args.data / "mined.jsonl"
    else:
        answers = canonical_answers(_read(DIR / "policy.jsonl"))
        rows, report = mine_policy(instance, client, args.handbook, answers, pool=args.pool, twins=args.twins,
                                   contrast=args.contrast, seed=args.seed, held=held, workers=args.workers)
        out = args.out or DIR / "mined-policy.jsonl"
    _write(out, rows)
    path = Path("runs") / f"mine-{args.stage}-{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": name, **report}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {len(rows)} rows to {out}; failures in {path}")


if __name__ == "__main__":
    main()
