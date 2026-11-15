"""The Kaggle notebook for nts-memory-v1_1, made from v1's (``notebooks/train_memory_kaggle.ipynb``, unchanged): the
same set-up, row check and quick check; continuing from v1's adapter on ``data/memory-train-v1_1``.

    uv run python scripts/make_memory_notebook.py   # writes notebooks/train_memory_v1_1_kaggle.ipynb
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "train_memory_kaggle.ipynb"
OUT = ROOT / "notebooks" / "train_memory_v1_1_kaggle.ipynb"

HEADER = """# nts-memory-v1_1: continuing from v1 (Qwen3.5-4B + LoRA)

The memory adapter reads the **newest chat message with the last 4 messages before it** and answers `{"same": true}`,
`{"message": "..."}` (the message in full) or `{"ask": "..."}` (a question back), as v1 does.

**What v1.1 adds** (`python -m training.memory_data_v1_1`):

| | |
|---|---|
| plural words mean all of it | "is Section 2 free on **these days** too?" → every day just asked about; "is Dr Rao free at **those times**?" → every free span the reply gave ("then" after several is still asked back); "the **same time** on Friday / next week / for Section 3" |
| more of what v1 missed | `list` ("move the second one"), `two_back` (two exchanges back), `ask` (asked back) |
| replay | 950 of v1's rows and 400 of Gemini's rewordings, so nothing v1 knows is forgotten |

**The set-up is v1's notebook's**; what differs: it **continues from nts-memory-v1's adapter** at a lower learning
rate (5e-5, as lookup v3 continued from v2), and reads `data/memory-train-v1_1`.

**Inputs to add (Add Input):**
1. `data/memory-train-v1_1`;
2. nts-memory-v1's **`lora/` folder** (`adapter_config.json` + `adapter_model.safetensors`, the output of v1's run);
3. only to resume: this run's earlier `checkpoints/`.

About 2,250 rows of ~560 tokens (a fifth of them messages to leave as they are, as in v1): **~3 h**. First `TRIAL = True` (a few minutes), then `TRIAL = False` and
*Save Version → Save & Run All*.

---
"""


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def replace(text: str, old: str, new: str) -> None:
        hits = [c for c in cells if text in "".join(c["source"])]
        assert len(hits) == 1, (text, len(hits))
        src = "".join(hits[0]["source"])
        assert old in src, (text, old)
        hits[0]["source"] = src.replace(old, new).splitlines(keepends=True)

    cells[0]["source"] = HEADER.splitlines(keepends=True)
    settings = "SESSION_START = time.time()"
    replace(settings, 'ADAPTER = "nts-memory-v1"', 'ADAPTER = "nts-memory-v1_1"')
    replace(settings, "LR = 2e-4                   # a fresh adapter (a later version continuing from this one: 5e-5)",
            "LR = 5e-5                   # continuing from v1: a lower rate adjusts what it knows instead of overwriting it")
    replace(settings, '''def _task(train_file):
    m = Path(train_file).parent / "manifest.json"
    return json.loads(m.read_text()).get("task") if m.exists() else None


found = [f for f in sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)) if _task(f) == "memory"]
assert found, "no memory train.jsonl: add the data/memory-train folder as a dataset (Add Input)"''',
            '''def _manifest(train_file):
    m = Path(train_file).parent / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else {}


found = [f for f in sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))
         if _manifest(f).get("task") == "memory" and _manifest(f).get("version") == 1.1]
assert found, "no memory v1.1 train.jsonl: add the data/memory-train-v1_1 folder as a dataset (Add Input)"''')
    replace('"task": "memory", "train_rows"', '"version": 1,', '"version": 1.1,')
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
