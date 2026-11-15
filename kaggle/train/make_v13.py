"""nts-intents-v13: v12's notebook (its code and Docker image) on the intents-v13 data, with the recipe Phase 0's
training runs settled on (docs/improvement-plan.md; kaggle-logs/cos1, cos3):

    python kaggle/train/make_v13.py     # kaggle/push/data-intents-v13/ (the dataset) and train-v13/ (the notebook)

* a cosine schedule to zero (CURVE = []: shuffled rows, one adapter at the end). v12 trained at a constant rate;
  the same data with cosine reached a lower validation loss at every epoch (cos3 0.0052 / 0.0023 / 0.0021 against
  v12's 0.0045 / 0.0055 / 0.0026, and v12's rose in epoch 2);
* 2 epochs: cos3's third epoch moved the validation loss by 0.0002 of 0.0023; the new kinds are upsampled in the
  data (the recipe's boost) instead;
* a larger end-of-run check (40 validation rows, against 15), printed by kind.

    python kaggle/train/make_v13.py b   # nts-intents-v13b: v12's own recipe on the same data (train-v13b/)

v13 scored worse than v12 on compound (65 vs 97.5) and out_for_me (54 vs 95), with fields of neighbouring kinds
carried into plain requests (a phantom ``"id":"mon"`` cancel, a dropped "rest of the term") that v12's data has
just as often: the regression is in the recipe, which no suite had checked (cos1 and cos3 ended in ERROR, so their
adapters cannot be attached). v13b keeps v12's settings (3 epochs at a constant rate, an adapter after each).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "v12-pulled"
PUSH = HERE.parent / "push"
ACCOUNT = "shubhadeepmandal"
ADAPTER = "nts-intents-v13"
DATASET = "nts-multitask-intents-v13"
SETTINGS = {"ADAPTER": f'"{ADAPTER}"', "EPOCHS": "2", "CURVE": "[]", "LR": "2e-4"}
VARIANTS = {"": SETTINGS, "b": {"ADAPTER": f'"{ADAPTER}b"', "EPOCHS": "3", "CURVE": "[1/3, 2/3, 1.0]", "LR": "2e-4"}}


def dataset() -> Path:
    d = PUSH / "data-intents-v13"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("train.jsonl", "val.jsonl", "compile-manifest.json"):
        shutil.copyfile(ROOT / "data" / "multitask-readable" / f, d / f)
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET, "id": f"{ACCOUNT}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    return d


def notebook(variant: str = "") -> Path:
    name = ADAPTER + variant
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    nb = json.loads((SRC / meta["code_file"]).read_text(encoding="utf-8"))
    src = "".join(nb["cells"][4]["source"])
    for key, value in VARIANTS[variant].items():
        src, n = re.subn(rf"^{key} = [^#\n]*", f"{key} = {value}  ", src, flags=re.M)
        assert n == 1, key
    nb["cells"][4]["source"] = src.splitlines(keepends=True)
    check = "".join(nb["cells"][13]["source"])
    check, n = re.subn(r"else 15\b", "else 40", check)
    assert n == 1, "check size"
    nb["cells"][13]["source"] = check.splitlines(keepends=True)
    for c in nb["cells"]:  # markdown cells have no outputs: a None there fails papermill after the run (cos1)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        else:
            c.pop("outputs", None)
            c.pop("execution_count", None)
    d = PUSH / f"train-v13{variant}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    m = {k: v for k, v in meta.items() if k != "id_no"}
    m.update(id=f"{ACCOUNT}/{name}", title=name, code_file=f"{name}.ipynb", kernel_sources=[],
             dataset_sources=[f"{ACCOUNT}/{DATASET}"])
    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
    return d


if __name__ == "__main__":
    import sys

    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    print(notebook(variant) if variant else (dataset(), notebook()))
