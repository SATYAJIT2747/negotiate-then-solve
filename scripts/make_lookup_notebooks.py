"""The Kaggle notebooks for nts-lookup-v3_2 and nts-lookup-v4, made from v3's (``notebooks/train_lookup_v3_kaggle.ipynb``,
unchanged): the same set-up and training, so the two runs differ only in the answer format they learn.

    uv run python scripts/make_lookup_notebooks.py   # writes notebooks/train_lookup_v3_2_kaggle.ipynb and _v4_

What changes from v3's notebook: the header, the settings (the adapter, the data's version, continuing from v3's
adapter), the check of every row (v3.2: up to 8 lookups; v4: each part's own fields, from ``agents.lookups_v4``'s
schema, so the notebook can't drift from the code), the quick check (the answer's list and nested fields) and
the manifest's version.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.lookups_v4 import MAX_PARTS, TOOLS, OtherPart, Request, Rule  # noqa: E402

SRC = ROOT / "notebooks" / "train_lookup_v3_kaggle.ipynb"

HEADER = """# {adapter}: continuing from v3 (Qwen3.5-4B + LoRA)

One of two runs on **the same rows** (`python -m training.lookup_data_v3_2`), in two answer formats; everything
else in the two notebooks is the same, so a difference in results is the format's:

| | nts-lookup-v3_2 | nts-lookup-v4 |
|---|---|---|
| Answer | v3's: `{{"lookups": [...]}}`, flat lookups; a request or rule question keeps its words in `request` | the lookup vocabulary (`agents/lookups_v4.py`): `{{"parts": [...]}}`, each tool only its fields; clock times, `next_week`; `request` / `rule` / `other` parts with their words |
| Prompt | `SYSTEM_V3_2` (v3's, plus one line) | `SYSTEM_V4` (about a third shorter) |
| Data folder | `data/lookup-train-v3_2` | `data/lookup-train-v4` |

**This notebook: {this}.** What both learn that v3 didn't: mixed messages (questions with 1 to 3 requests, rule
questions), long messages (up to 8 parts), and the slips v3 made ("a 2 hr slot", "is Rao away", "the exam hall",
"any section free for a guest session", "list the teachers", "when are we and Dr X both free"), with v3's rows
replayed so nothing is forgotten.

**Inputs to add (Add Input):**
1. `{data}` (train, val, distilled: Gemini's rewordings, used as training rows);
2. nts-lookup-v3's **`lora/` folder** (`adapter_config.json` + `adapter_model.safetensors`): the run continues from
   it. Add only v3's, not v2's;
3. only to resume: this run's earlier **`checkpoints/`** (save the earlier session's output as a dataset).

First `TRIAL = True` (a few minutes: installs, loads, 20 steps, exports), then `TRIAL = False` and
*Save Version → Save & Run All*.

---
"""

CHECK_V3_2 = '''from collections import Counter

ANSWER = "lookups"
TOOLS = {"free_hours", "classes_of", "free_rooms", "who_is_free", "describe", "changes", "calendar", "my_requests",
         "my_rules", "clash_check", "move_options", "counts", "gaps", "my_fairness", "bookings", "what_if",
         "unanswerable", "none"}
LIST_FIELDS = {"teachers", "sections", "rooms", "days", "dates", "hours", "equipment", "unknown", "ambiguous",
               "roles", "room_for", "weeks"}
OTHER_FIELDS = {"tool", "reason", "course", "part", "length", "kind", "min_capacity", "next", "count", "of",
                "what", "week", "first", "change", "which", "request_id", "measure", "order", "class_day",
                "class_kind", "club", "request", "separately"}
DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri"}


def problem_of(answer):
    # one lookup in the list
    if not isinstance(answer, dict) or answer.get("tool") not in TOOLS:
        return f"unknown tool {answer.get('tool')!r}"
    if not isinstance(answer.get("week"), int):
        return "no week number"
    if extra := set(answer) - LIST_FIELDS - OTHER_FIELDS:
        return f"unknown fields {sorted(extra)}"
    if bad := [k for k in LIST_FIELDS & set(answer) if not isinstance(answer[k], list)]:
        return f"not lists: {bad}"
    if bad := [d for d in answer.get("days", []) if d not in DAYS]:
        return f"days {bad}"
    if bad := [h for h in answer.get("hours", []) if not isinstance(h, int) or not 0 <= h <= 7]:
        return f"hours {bad}"
    if answer["tool"] == "none" and "request" in answer and not answer["request"].strip():
        return "a request with no words"
    return None
'''

CHECK_V4 = '''from collections import Counter
import re

ANSWER = "parts"
# every part's fields, and the fields inside who / when / room / which / to (agents/lookups_v4.py's schema)
FIELDS = __FIELDS__
NESTED = __NESTED__
TOOLS = set(FIELDS)
DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri"}
CLOCK = re.compile(r"^([01]\\d|2[0-3]):[0-5]\\d$")


def problem_of(answer):
    # one part in the list
    if not isinstance(answer, dict) or answer.get("tool") not in TOOLS:
        return f"unknown tool {answer.get('tool')!r}"
    if extra := set(answer) - FIELDS[answer["tool"]]:
        return f"{answer['tool']} has no {sorted(extra)}"
    for key in ("who", "when", "room", "which", "to"):
        if key in answer and not (key == "which" and answer["tool"] != "move_options"):  # my_requests' which: a word
            if not isinstance(answer[key], dict):
                return f"{key} is not an object"
            if extra := set(answer[key]) - NESTED[key]:
                return f"{key} has no {sorted(extra)}"
    when = answer.get("when") or answer.get("to") or {}
    if bad := [d for d in when.get("days", []) if d not in DAYS]:
        return f"days {bad}"
    if bad := [t for t in when.get("at", []) + [when.get("after"), when.get("before")] if t and not CLOCK.match(t)]:
        return f"times {bad}"
    if answer["tool"] in ("request", "rule", "what_if") and not str(answer.get("text", "")).strip():
        return f"a {answer['tool']} part with no words"
    return None
'''

CHECK_COMMON = '''

def problem(row):
    # Why a row must not be trained on, or None when it is fine.
    msgs = row.get("messages")
    if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return "messages are not system, user, assistant"
    if "<question>" not in msgs[1]["content"]:
        return "the user message has no <question>"
    try:
        items = json.loads(msgs[2]["content"]).get(ANSWER)
    except (json.JSONDecodeError, AttributeError):
        return "the answer is not a JSON object"
    if not isinstance(items, list) or not 1 <= len(items) <= __MAX__:
        return f"the answer has no list of 1 to __MAX__ {ANSWER}"
    for answer in items:
        if why := problem_of(answer):
            return why
    return None


def load(name):
    rows, dropped = [], []
    for n, line in enumerate(open(DATA / f"{name}.jsonl", encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped.append((n, "not a JSON line"))
            continue
        why = problem(row)
        if why:
            dropped.append((n, why))
        else:
            rows.append(row)
    print(f"{name}: {len(rows)} rows kept, {len(dropped)} left out")
    for n, why in dropped[:20]:
        print(f"   line {n}: {why}")
    return rows


train_rows, val_rows = load("train"), load("val")
if (DATA / "distilled.jsonl").exists():  # Gemini's rewordings: training rows too
    train_rows += load("distilled")
assert train_rows and val_rows, "no usable rows"
if MAX_TRAIN_ROWS and len(train_rows) > MAX_TRAIN_ROWS:  # a shorter run: a random part, every kind still in it
    train_rows = random.Random(SEED).sample(train_rows, MAX_TRAIN_ROWS)
    print(f"training on {len(train_rows)} of the rows (MAX_TRAIN_ROWS)")

# What the data covers: every tool, every kind of message, both kinds of asker, several departments.
def tools_of(row):
    return [x["tool"] for x in json.loads(row["messages"][2]["content"])[ANSWER]]

print("\\ntools:      ", dict(Counter(t for r in train_rows for t in tools_of(r)).most_common()))
print(f"{ANSWER} per message:", dict(sorted(Counter(len(tools_of(r)) for r in train_rows).items())))
print("kinds:      ", dict(Counter(r.get("kind") for r in train_rows).most_common()))
print("departments:", dict(Counter(r.get("department") for r in train_rows).most_common()))
fields = Counter(k for r in train_rows for x in json.loads(r["messages"][2]["content"])[ANSWER] for k in x)
print("fields:     ", dict(fields.most_common()))
missing = TOOLS - {t for r in train_rows for t in tools_of(r)} - {"unanswerable"}
assert not missing, f"no training rows for {missing}: make the data again"
'''

QUICK = '''model.eval()
model.config.use_cache = True


def norm(o):
    # the same answer: empty fields don't count, lists in any order, nested objects alike
    if isinstance(o, dict):
        return {k: norm(v) for k, v in o.items() if v not in (None, [], False, {})}
    if isinstance(o, list):
        return sorted((norm(x) for x in o), key=lambda x: json.dumps(x, sort_keys=True))
    return o


rows = random.Random(SEED).sample(val_rows, min(CHECK_ROWS, len(val_rows)))
score, misses = Counter(), []
for row in rows:
    inputs = tok(prompt_of(row["messages"]), return_tensors="pt").to("cuda:0")  # the first layers are on GPU 0
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=700, do_sample=False, use_cache=True)
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    want = json.loads(row["messages"][2]["content"])[ANSWER]
    score["questions"] += 1
    try:
        got = json.loads(text)
        score["valid JSON"] += 1
        ok = (isinstance(got, dict) and len(got.get(ANSWER, [])) == len(want)
              and all(norm(g) == norm(w) for g, w in zip(got[ANSWER], want)))
    except json.JSONDecodeError:
        ok = False
    score["right"] += ok
    for t in {x["tool"] for x in want}:
        score[f"right {t}"] += ok
        score[f"asked {t}"] += 1
    score[f"right {len(want)} parts"] += ok
    score[f"asked {len(want)} parts"] += 1
    kind = (row.get("kind") or "").split("+")[0].split()[0]
    score[f"right kind {kind}"] += ok
    score[f"asked kind {kind}"] += 1
    if not ok:
        misses.append((row["messages"][1]["content"].split("<question>")[1].split("</question>")[0], want, text))

print(f"{score['right']}/{score['questions']} right, {score['valid JSON']} valid JSON")
for n in range(1, __MAX__ + 1):
    if score[f"asked {n} parts"]:
        print(f"   {n} part(s)    {score[f'right {n} parts']}/{score[f'asked {n} parts']}")
for k in sorted(k[len("asked kind "):] for k in score if k.startswith("asked kind ")):
    print(f"   kind {k:10} {score[f'right kind {k}']}/{score[f'asked kind {k}']}")
for t in sorted(TOOLS):
    if score[f"asked {t}"]:
        print(f"   {t:13} {score[f'right {t}']}/{score[f'asked {t}']}")
for question, want, text in misses[:15]:
    print(f"\\nQ: {question}\\n   want {json.dumps(want)}\\n   got  {text}")
'''


def v4_fields() -> tuple[dict, dict]:
    from agents.lookups_v4 import MoveOptions, RoomNeed, TheClass, When, Who

    fields = {tool: sorted(m.model_fields) for tool, m in TOOLS.items()}
    fields |= {m.model_fields["tool"].annotation.__args__[0]: sorted(m.model_fields) for m in (Request, Rule, OtherPart)}
    nested = {"who": sorted(Who.model_fields), "when": sorted(When.model_fields), "to": sorted(When.model_fields),
              "room": sorted(RoomNeed.model_fields), "which": sorted(TheClass.model_fields)}
    assert set(MoveOptions.model_fields) >= {"which", "to"}
    return fields, nested


def make(adapter: str, version, data: str, this: str, check: str) -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def code(text: str) -> dict:
        hits = [c for c in cells if text in "".join(c["source"])]
        assert len(hits) == 1, (text, len(hits))
        return hits[0]

    def replace(text: str, old: str, new: str) -> None:
        c = code(text)
        src = "".join(c["source"])
        assert old in src, (text, old)
        c["source"] = src.replace(old, new).splitlines(keepends=True)

    def put(text: str, src: str) -> None:
        code(text)["source"] = src.splitlines(keepends=True)

    cells[0]["source"] = HEADER.format(adapter=adapter, this=this, data=data).splitlines(keepends=True)
    replace("SESSION_START = time.time()", 'ADAPTER = "nts-lookup-v3"', f'ADAPTER = "{adapter}"')
    replace("SESSION_START = time.time()", "# continuing from v2: a lower rate", "# continuing from v3: a lower rate")
    replace("SESSION_START = time.time()", "v3's rows are about 1,300", "rows are about 1,300 (v3.2) or 900 (v4)")
    replace("SESSION_START = time.time()", 'if _version(f) == 3]', f'if _version(f) == {version}]')
    replace("SESSION_START = time.time()", '"no v3 train.jsonl: add the data/lookup-train-v3 folder',
            f'"no {data} train.jsonl (manifest version {version}): add the {data} folder')
    replace("SESSION_START = time.time()", "nts-lookup-v2's lora/ folder", "nts-lookup-v3's lora/ folder")
    replace("SESSION_START = time.time()", "another adapter's final lora/ (v2's)", "another adapter's final lora/ (v3's)")
    replace("SESSION_START = time.time()", "v2's checkpoints, in the same input as v2's lora/, are not v3's.",
            "v3's checkpoints, in the same input as v3's lora/, are not this run's.")
    put("TOOLS = {", check + CHECK_COMMON.replace("__MAX__", str(MAX_PARTS)))
    replace("model = get_peft_model(model", "# v2's adapter, trained further: it starts knowing everything v2 knows",
            "# v3's adapter, trained further: it starts knowing everything v3 knows")
    put("def same(a, b):", QUICK.replace("__MAX__", str(MAX_PARTS)))
    replace('"version": 3,', '"version": 3,', f'"version": {version},')
    out = ROOT / "notebooks" / f"train_lookup_{adapter.removeprefix('nts-lookup-')}_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


def main() -> None:
    fields, nested = v4_fields()
    make("nts-lookup-v3_2", 3.2, "data/lookup-train-v3_2", "v3.2, the flat format", CHECK_V3_2)
    make("nts-lookup-v4", 4, "data/lookup-train-v4", "v4, the lookup vocabulary",
         CHECK_V4.replace("__FIELDS__", "{\n    " + ",\n    ".join(
             f'"{t}": {set(f)!r}' for t, f in sorted(fields.items())) + "\n}")
         .replace("__NESTED__", json.dumps(nested).replace("[", "{").replace("]", "}")))


if __name__ == "__main__":
    main()
