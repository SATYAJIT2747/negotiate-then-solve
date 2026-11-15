"""Training variants of nts-intents-v12 for Phase 3's recipe questions (docs/improvement-plan.md, R1, R2, R4).

    python kaggle/train/make_variants.py      # writes kaggle/push/train-<name>/ (notebook + kernel-metadata.json)

Each is v12's own notebook (pulled from shubhadeepmandal/nts-intents-v12: the same data, code and Docker image) with
only the settings changed. CURVE = [] makes the notebook train as a normal run: shuffled order and a cosine schedule
that decays to zero, one adapter at the end (gguf/<ADAPTER>.gguf). v12 itself (constant rate, curve points after
epochs 1, 2, 3) is the comparison.

    cos3        cosine, 3 epochs      vs v12 read-100 (constant, 3 epochs): does annealing help?      R1
    cos2, cos1  cosine, 2 and 1 epochs                                       how many epochs?          R2
    cos2-lr1e4  cosine, 2 epochs, lr 1e-4                                    a lower rate?             R4
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v12-pulled"
PUSH = HERE.parent / "push"
VARIANTS = {
    "cos3": {"EPOCHS": "3", "CURVE": "[]", "LR": "2e-4"},
    "cos2": {"EPOCHS": "2", "CURVE": "[]", "LR": "2e-4"},
    "cos1": {"EPOCHS": "1", "CURVE": "[]", "LR": "2e-4"},
    "cos2-lr1e4": {"EPOCHS": "2", "CURVE": "[]", "LR": "1e-4"},
}


def main() -> None:
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    nb = json.loads((SRC / meta["code_file"]).read_text(encoding="utf-8"))
    for name, sets in VARIANTS.items():
        adapter = f"nts-intents-v12-{name}"
        cells = json.loads(json.dumps(nb))
        src = "".join(cells["cells"][4]["source"])
        src, n = re.subn(r'^ADAPTER = "[^"]*"', f'ADAPTER = "{adapter}"', src, flags=re.M)
        assert n == 1
        for key, value in sets.items():
            src, n = re.subn(rf"^{key} = [^#\n]*", f"{key} = {value}  ", src, flags=re.M)
            assert n == 1, key
        cells["cells"][4]["source"] = src.splitlines(keepends=True)
        for c in cells["cells"]:
            c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs")
            if c["cell_type"] == "code":
                c["execution_count"] = None
        d = PUSH / f"train-{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{adapter}.ipynb").write_text(json.dumps(cells, indent=1), encoding="utf-8")
        m = {k: v for k, v in meta.items() if k not in ("id_no",)}
        m.update(id=f"shubhadeepmandal/{adapter}", title=adapter, code_file=f"{adapter}.ipynb", kernel_sources=[])
        (d / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
        print(name, "->", d.name, sets)


if __name__ == "__main__":
    main()
