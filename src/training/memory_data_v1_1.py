"""Training data for nts-memory-v1_1, continuing from nts-memory-v1's adapter.

    uv run python -m training.memory_data_v1_1      # writes data/memory-train-v1_1/

What it adds to v1's data (``training.memory_data``, unchanged; its test set stays the one v1 is measured on):

- **plural** conversations: "these days", "those times", "the same time" (a plural word means all of it);
- more of the kinds v1 missed most on its test set: ``list`` (the first / second one of a list), ``two_back``
  (what it refers to is two exchanges back) and ``ask`` (asked back when the conversation lacks it);
- **replay**: a sample of v1's training rows and of Gemini's rewordings, so nothing v1 knows is forgotten.

The new conversations are made in v1's departments with other seeds (``tag``), so none repeats one of v1's and
the test set's names stay its own. ``test-plural.jsonl``: the new kind on the test templates and wordings, for
``evaluation.memory_set --test``. About 2,100 training rows: ~3 h on Kaggle (v1's 3,221 took ~4–4.5 h).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .lookup_data import ROOT
from .memory_data import OUT as V1, build

OUT = ROOT / "data" / "memory-train-v1_1"
# the new rows' kinds, in these proportions; "same" keeps the share of messages left as they are near v1's (24%):
# without it the new rows, all to rewrite, would teach rewriting what needs none
NEW = {"plural": 6, "list": 2, "two_back": 1, "ask": 1, "same": 3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", type=int, default=1000, help="new conversations (a tenth kept for validation)")
    ap.add_argument("--replay", type=int, default=950, help="v1's training rows again (a sample)")
    ap.add_argument("--distilled", type=int, default=400, help="Gemini's rewordings of v1's rows again (a sample)")
    ap.add_argument("--test", type=int, default=64, help="conversations of the new kind for the test set")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from web.app import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    new, cover, skipped = build(w.instance, w.store, a.new, test=False, seed=a.seed, shares=NEW, tag="v1_1-")
    test, test_cover, test_skipped = build(w.instance, w.store, a.test, test=True, seed=a.seed,
                                           shares={"plural": 1}, tag="v1_1-")
    rng = random.Random(f"v1_1-{a.seed}")
    rng.shuffle(new)
    new_val, new_train = new[: len(new) // 10], new[len(new) // 10:]
    load = lambda name: [json.loads(x) for x in (V1 / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    old_train, old_distilled = load("train"), load("distilled")
    train = new_train + rng.sample(old_train, min(a.replay, len(old_train)))
    rng.shuffle(train)
    files = {"train": train, "val": new_val + load("val"),
             "distilled": rng.sample(old_distilled, min(a.distilled, len(old_distilled))), "test-plural": test}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {"task": "memory", "version": 1.1, "start": "nts-memory-v1", "seed": a.seed,
                **{name: len(rows) for name, rows in files.items()}, "new": len(new), "new_kinds": dict(cover),
                "replayed": a.replay, "test_kinds": dict(test_cover),
                "train_kinds": dict(Counter(r["kind"] for r in train).most_common())}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(train)} train ({len(new_train)} new), {len(files['val'])} val, {len(files['distilled'])} distilled, "
          f"{len(test)} test-plural in {OUT}")
    print("new kinds:", dict(cover), "| test:", dict(test_cover))
    print("skipped:", dict(skipped + test_skipped))


if __name__ == "__main__":
    main()
