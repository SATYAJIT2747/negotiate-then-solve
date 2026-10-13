"""The intent reader's training data, built from a recipe, gated and recorded, so later runs build on it.

    uv run python -m training.dataset build                    # data/recipes/intents.toml
    uv run python -m training.dataset build --check            # build and gate, write nothing
    uv run python -m training.dataset build --refreeze         # accept a changed test split (say why)

A recipe (``data/recipes/*.toml``) names the sources and how much of each:

* ``base``: the first data (``intent_compiler.base_rows``: the corpus, ``intent_corpus``'s generators and
  train-only wordings); its test split is the one every model so far was scored on;
* ``generators``: ``intent_generators.REGISTRY``, each generator's held-out wordings going to its suite;
* ``stored``: rows that cannot be rebuilt (mined from a model's mistakes, labelled by people), read from
  files, migrated to the current schema (``language.migrations``) and put in the current prompt;
* ``augment``: noisy twins of training rows (typos, shorthand, casual wording: ``evaluation.robustness``,
  with seeds of its own), each kept only if its gold resolves as the original's does;
* ``suites``: the evaluation suites written beside the test split (``data/suites``), held-out wordings and
  the test split rewritten at fixed noise levels.

Every row passes ``training.quality``'s gates, and the build stops if one fails. The test split is
**frozen**: the recipe holds its fingerprint and the build refuses to change it, so every model is scored on
the same requests. ``manifest.json`` beside the data records the recipe, the code version, each source's
rows and each file's fingerprint; the training notebook copies it into the adapter's own manifest, so an
adapter can always be traced to the exact data it read, and an update (``INIT_FROM``) reads only the rows
it has not seen.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import random
import subprocess
import tomllib
from collections import Counter
from pathlib import Path

from core.instance import Instance
from evaluation.intents import _context, outcome, same
from evaluation.robustness import protected, rewrite_row, rewritten
from language.intent_compiler import INTENT_PROMPT, base_rows, export_rows, label_placements
from language.intents import SCHEMA_VERSION, IntentOutput, compile_intents
from language.migrations import migrate_row

from . import human, quality

RECIPE = Path("data/recipes/intents.toml")
SPLITS = ("train", "val", "test")


def fingerprint(rows: list[dict]) -> str:
    """What a split asks and expects, in any order: each message as the model reads it, its gold, and what the
    gold resolves against. Not the system prompt: that is how the task is put, and changes with the
    vocabulary, while the requests and their answers stay."""
    keys = ("request", "timetable", "current_week")
    # an id is not unique (a corpus request and its paraphrase share one): order by the whole entry
    entries = sorted(json.dumps([r["id"], r["messages"][1:], *(r.get(k) for k in keys)], sort_keys=True,
                                ensure_ascii=False) for r in rows)
    blob = "\n".join(entries)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _git() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def _current_prompt(row: dict) -> dict:
    row = copy.deepcopy(row)
    row["messages"][0]["content"] = INTENT_PROMPT
    return row


def stored_rows(instance: Instance, spec: dict) -> list[tuple[str, dict]]:
    """``(split, row)`` from files: migrated, in the current prompt, labelled by the project's rule for moves
    when ``relabel`` (rows made by code under an older rule), never for rows people labelled."""
    paths = sorted(Path(p) for pattern in spec["paths"] for p in glob.glob(pattern))
    if spec.get("kind") == "human":
        return [(split, _current_prompt(r)) for split, r in human.load(instance, paths)]
    out = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = _current_prompt(migrate_row(json.loads(line)))
            if spec.get("relabel", False):
                out_ = label_placements(IntentOutput.model_validate_json(row["messages"][-1]["content"]),
                                        row["request"]["raw_text"])
                from language.intent_compiler import completion

                row["messages"][-1]["content"] = completion(out_)
            out.append((spec.get("split", "train"), row))
    return out


def augmented(instance: Instance, rows: list[dict], spec: dict) -> tuple[list[dict], Counter]:
    """Noisy twins of ``rows``: a ``fraction`` of them, each at one of ``levels``, kept only when the twin's
    gold resolves to what the original's does (a reworded message must still mean the same)."""
    rng = random.Random(spec.get("seed", "augment"))
    keep, why = protected(instance), Counter()
    out = []
    for row in rows:
        if rng.random() >= spec["fraction"]:
            continue
        level = rng.choice(spec["levels"])
        twin = rewrite_row(row, level, f"aug:{spec.get('seed', '')}:{row['id']}", keep)
        if twin["request"]["raw_text"] == row["request"]["raw_text"]:
            why["unchanged"] += 1
            continue
        gold = IntentOutput.model_validate_json(row["messages"][-1]["content"])
        got = []
        for r in (row, twin):
            request, timetable, week = _context(r)
            got.append(outcome(instance, compile_intents(instance, request, gold, timetable=timetable,
                                                         current_week=week)))
        if not same(instance, got[1], got[0]):
            why["changed meaning"] += 1
            continue
        twin.update(id=f"{row['id']}~{level}", origin="augment", augment=level, parent=row["id"],
                    parent_text=row["request"]["raw_text"])
        out.append(twin)
        why["kept"] += 1
    return out, why


def build(recipe: dict, *, refreeze: bool = False) -> tuple[dict[str, list[dict]], dict]:
    """Every split and suite of ``recipe``, gated, and the manifest."""
    instance = Instance.model_validate_json(Path(recipe["instance"]).read_text(encoding="utf-8"))
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    sources: dict[str, dict] = {}

    def add(origin: str, pairs) -> None:
        c = Counter()
        for split, row in pairs:
            row["origin"] = origin
            splits.setdefault(split, []).append(row)
            c[split] += 1
        sources[origin] = dict(sorted(c.items()))

    b = recipe["base"]
    kept, _ = base_rows(instance, [Path(p) for p in b["corpus"]], extra_denials_n=b["extra_denials"],
                        min_per_action=b["min_per_action"], per_generator=b["per_generator"],
                        max_per_kind=b["max_per_kind"], train_only=b["train_only"])
    add("base", kept)

    from language.intent_generators import generated

    g = recipe["generators"]
    add("generators", export_rows(instance, [], generated(instance, per_generator=g["per_generator"],
                                                          held_per_generator=g["held_per_generator"],
                                                          seed=g["seed"], names=g.get("names"))))
    for name, spec in recipe.get("stored", {}).items():
        add(name, stored_rows(instance, spec))
    if a := recipe.get("augment"):
        pool = [r for r in splits["train"] if r["origin"] in a.get("origins", ["base", "generators"])]
        twins, why = augmented(instance, pool, a)
        splits["train"] += twins
        sources["augment"] = {"train": len(twins), **{k: v for k, v in why.items() if k != "kept"}}
    for level in recipe.get("suites", {}).get("noisy_levels", []):
        rows = rewritten(splits["test"], level, instance)
        for r, parent in zip(rows, splits["test"], strict=True):
            r["origin"], r["augment"], r["parent_text"] = "suite", level, parent["request"]["raw_text"]
        splits[f"test-{level}"] = rows

    for name, rows in splits.items():  # a fixed order within a split, whatever order the sources came in
        random.Random(f"order:{name}").shuffle(rows)

    held = [s for s in splits if s not in ("train", "val")]
    failures = quality.check(instance, splits, held=held)
    # what may be dropped rather than stop the build: a training copy of a held-out request (leak), and a
    # stored row that no longer passes (made by older code, it cannot be rebuilt); anything built by code
    # that fails a gate is a bug in the code, and stops the build
    stored = set(recipe.get("stored", {}))
    dropped: dict[str, list[str]] = {}
    for gate, fails in failures.items():
        drop = [f for f in fails if f["split"] in ("train", "val") and (gate == "leak" or f["origin"] in stored)]
        if drop:
            bad = {(f["split"], f["id"]) for f in drop}
            for name in ("train", "val"):
                splits[name] = [r for r in splits[name] if (name, r["id"]) not in bad]
            dropped[gate] = sorted(f"{f['split']}:{f['id']}" for f in drop)
            failures[gate] = [f for f in fails if f not in drop]
    frozen = recipe.get("frozen", {})
    got = fingerprint(splits["test"])
    if frozen.get("test") and frozen["test"] != got and not refreeze:
        raise SystemExit(f"the test split changed (fingerprint {got}, recipe {frozen['test']}): every model so "
                         "far was scored on the frozen one. Find what changed it, or pass --refreeze and say why.")
    manifest = {
        "schema": SCHEMA_VERSION, "recipe": recipe.get("name"), "recipe_sha": recipe["_sha"], "code": _git(),
        "sources": sources, "fingerprints": {s: fingerprint(r) for s, r in splits.items()},
        "gates": {gate: len(f) for gate, f in failures.items()}, "dropped": dropped,
        "report": quality.report(splits),
    }
    return splits, {"manifest": manifest, "failures": failures}


def write(splits: dict[str, list[dict]], manifest: dict, out: Path, suites: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    suites.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out / f"{name}.jsonl" if name in SPLITS else suites / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def load_recipe(path: Path) -> dict:
    raw = path.read_bytes()
    recipe = tomllib.loads(raw.decode("utf-8"))
    recipe["_sha"] = hashlib.sha256(raw).hexdigest()[:16]
    return recipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build"])
    ap.add_argument("--recipe", type=Path, default=RECIPE)
    ap.add_argument("--check", action="store_true", help="build and gate only; write nothing")
    ap.add_argument("--refreeze", action="store_true", help="accept a changed test split")
    ap.add_argument("--allow-failures", action="store_true", help="write even if a gate fails (look at them!)")
    args = ap.parse_args()

    recipe = load_recipe(args.recipe)
    splits, result = build(recipe, refreeze=args.refreeze)
    m, failures = result["manifest"], result["failures"]
    print(f"recipe {m['recipe']} ({m['recipe_sha']}), schema {m['schema']}, code {m['code']}")
    for name, rows in splits.items():
        print(f"  {name:16} {len(rows):5} rows  {m['fingerprints'][name]}")
    print("sources:", json.dumps(m["sources"]))
    print("gates:", quality.summary(failures))
    for gate, ids in m["dropped"].items():
        print(f"  dropped ({gate}): {len(ids)}: {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}")
    if any(failures.values()):
        print(quality.dump_failures(failures))
    if args.check:
        return
    if any(failures.values()) and not args.allow_failures:
        raise SystemExit("a gate failed: nothing written")
    out, suites = Path(recipe["output"]["dir"]), Path(recipe["output"]["suites"])
    write(splits, m, out, suites)
    print(f"wrote {out}/ and {suites}/ (manifest.json beside the data)")


if __name__ == "__main__":
    main()
