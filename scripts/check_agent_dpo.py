"""A local check of the training notebook's DPO mode with a tiny random model: the collator, seq_logp, the loss,
training, saving and the held-out accuracy (CPU or one GPU, a few seconds). It runs the script the notebook writes
(the %%writefile cell), so it checks what Kaggle will run:

    python scripts/check_agent_dpo.py      # needs torch, transformers>=5, peft, datasets, accelerate
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "train_agent_kaggle.ipynb"
cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
script = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("%%writefile /tmp/agent_train.py"))
where = Path(tempfile.mkdtemp())
(where / "agent_train.py").write_text(script.split("\n", 1)[1], encoding="utf-8")
sys.path.insert(0, str(where))
import agent_train

torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128)
base = LlamaForCausalLM(cfg)
model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

rng = random.Random(0)


def pair():
    prompt = [rng.randrange(3, 64) for _ in range(rng.randint(8, 20))]
    good = [5, 6, 7, 2]  # "chosen" always ends 5 6 7; "rejected" 9 9 9: learnable
    bad = [9, 9, 9, 2]
    ids_c, ids_r = prompt + good, prompt + bad
    return {"chosen_ids": ids_c, "chosen_labels": [-100] * len(prompt) + good,
            "rejected_ids": ids_r, "rejected_labels": [-100] * len(prompt) + bad}


rows = [pair() for _ in range(48)]
collate = agent_train.pair_collator(0)
with torch.no_grad():  # the reference: the model before DPO
    for x in rows:
        b = collate([{**x, "ref_chosen": 0.0, "ref_rejected": 0.0}])
        lp = agent_train.seq_logp(model, b["input_ids"], b["attention_mask"], b["labels"])
        x["ref_chosen"], x["ref_rejected"] = float(lp[0]), float(lp[1])
assert all(x["ref_chosen"] < 0 and x["ref_rejected"] < 0 for x in rows)

# padding does not change a row's log-probability (chosen and rejected differ in length here)
x = {**rows[0], "rejected_ids": rows[0]["rejected_ids"] + [9], "rejected_labels": rows[0]["rejected_labels"] + [9],
     "ref_chosen": 0.0, "ref_rejected": 0.0}
b = collate([x])
lp = agent_train.seq_logp(model, b["input_ids"], b["attention_mask"], b["labels"])
alone = agent_train.seq_logp(model, b["input_ids"][:1, :len(x["chosen_ids"])], b["attention_mask"][:1, :len(x["chosen_ids"])],
                             b["labels"][:1, :len(x["chosen_ids"])])
assert torch.allclose(lp[0], alone[0], atol=1e-4), (lp, alone)

out = Path(os.environ.get("OUT_DIR") or tempfile.mkdtemp())
C = {"OUT": str(out), "PRECISION": "fp32", "DPO_BETA": 0.1, "DPO_LR": 5e-3, "DPO_EPOCHS": 3, "DPO_ACCUM": 4,
     "SEED": 0, "PAD": 0, "SESSION_START": 0, "TIME_BUDGET_H": 1e9, "TRIAL": False}
val = Dataset.from_list(rows[40:])
before = agent_train.pair_accuracy(model, val, 0)
world = int(os.environ.get("WORLD_SIZE", "1"))
R = agent_train.train_dpo(C, model, None, Dataset.from_list(rows[:40]), val, processes=world)
if int(os.environ.get("RANK", "0")) == 0:
    print(json.dumps(R))
    assert (out / "lora-dpo" / "adapter_config.json").exists() and (out / "result-dpo.json").exists()
    assert R["val_accuracy"] > before and R["val_accuracy"] >= 0.75, (before, R)
    print("DPO check ok: held-out accuracy", before, "->", R["val_accuracy"])
