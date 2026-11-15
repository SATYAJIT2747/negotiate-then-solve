"""NTS intent reader on GLiNER2 (Fastino): zero-shot, then fine-tuned. Kaggle script kernel.

The same data, scorer and gate as ``nts_encoder.py`` (imported from it, or concatenated into one kernel file
by ``make_kernel.py``), but GLiNER2's own schema and heads instead of ours:

* ``action``: a classification over the six actions;
* ``choice``: a multi-label classification (any of, ranked, fallback, none);
* every request is one instance of a structure named by its kind (``move``, ``avoid`` ...): its role spans
  are list fields ("class day", "target day", ...), its attributes and "same class as before" are choice
  fields. GLiNER2 counts the instances of each structure and fills each one.

A prediction is turned back into our format (each request's role spans by text, its attributes as the gold
values, requests ordered by where their spans are) and scored by ``correct`` / ``score``: exact match of
the whole reading, by tier, with the coverage / accuracy curve over ``conf`` (the least confident decision).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

try:  # one kernel file on Kaggle: these are defined above this code
    from nts_encoder import ACTIONS, COMPOSITES, FIELD_TEXT, build_schema, correct, load, role_name, score, usable, value_text
except ImportError:
    pass

CHOICE_LABELS = ["any of", "ranked", "fallback", "none"]
SAME = "same class as before"


def kind_text(k: str) -> str:
    return k.replace("_", " ")


def field_names(schema: dict, kind: str) -> dict:
    """Our role / attribute / ref -> GLiNER2 field name, unique within the structure."""
    sk = schema["per_kind"][kind]
    out = {("role", r): role_name(r) for r in sk["roles"]}
    out.update({("class", f): FIELD_TEXT.get(f, f.replace(".", " ").replace("_", " ")) for f in sk["classes"]})
    out.update({("ref", f): f"{role_name(f + '.x').split(' x')[0]} {SAME}" if f in ("which", "other") else f"{f} {SAME}"
                for f in sk["refs"]})
    names = list(out.values())
    assert len(names) == len(set(names)), f"field names collide in {kind}: {names}"
    return out


def sender(row: dict) -> str:
    return f"Sender: {row['context'].get('role') or 'faculty'}.\n"


def to_examples(row: dict, schema: dict) -> list:
    """One span row -> two GLiNER2 examples, shaped like the two inference stages: (A) the classifications
    alone, (B) the structures alone. Trained together, GLiNER2 prefixes the text of every example with
    structures by their choice fields ("( move: firmness ( wish | must ) ... )"), so its classifiers learned
    "no prefix -> not a change" and read every request as out of scope at stage 1, which has no structures
    (Kaggle run 1: action accuracy 0.29, non-compile rows 110/112, compile rows 0)."""
    from gliner2.training.data import InputExample

    ex = to_example(row, schema)
    out = [InputExample(text=ex.text, classifications=ex.classifications)]
    if ex.structures:
        out.append(InputExample(text=ex.text, structures=ex.structures))
    return out


def to_example(row: dict, schema: dict):
    """One span row -> a GLiNER2 InputExample (every field of a request's kind given, empty when unsaid)."""
    from gliner2.training.data import ChoiceField, Classification, InputExample, Structure

    comps = sorted({c["kind"].replace("_", " ") for c in row.get("composites", [])}) or ["none"]
    structures = []
    for it in sorted(row["intents"], key=lambda i: (i["extent"] or [10 ** 6])[0]):
        if it["kind"] not in schema["per_kind"]:
            continue
        sk, names = schema["per_kind"][it["kind"]], field_names(schema, it["kind"])
        fields = {}
        for r in sk["roles"]:
            vals = []
            for s in it["spans"]:
                if s["role"] == r and s["text"] not in vals:
                    vals.append(s["text"])
            fields[names[("role", r)]] = vals
        for f, vals in sk["classes"].items():
            v = json.dumps(it["classes"][f]) if f in it["classes"] else "none"
            fields[names[("class", f)]] = ChoiceField(value_text(f, v), [value_text(f, x) for x in vals])
        for f in sk["refs"]:
            fields[names[("ref", f)]] = ChoiceField("yes" if f in it["refs"] else "no", ["no", "yes"])
        structures.append(Structure(kind_text(it["kind"]), mode=None, _field_values=fields))
    return InputExample(
        text=sender(row) + row["text"],
        classifications=[Classification("action", [a.replace("_", " ") for a in ACTIONS], row["action"].replace("_", " ")),
                         Classification("choice", CHOICE_LABELS, comps, multi_label=True),
                         Classification("requests", kind_labels(schema),
                                        sorted({kind_text(s.struct_name.replace(" ", "_")) for s in structures})
                                        or ["none"], multi_label=True)],
        structures=structures)


def kind_labels(schema: dict) -> list[str]:
    return [kind_text(k) for k in schema["kinds"]] + ["none"]


def stage1_schema(model, schema: dict):
    """What the message is: the action, the choices and which kinds of request it makes."""
    s = model.create_schema()
    s = s.classification("action", [a.replace("_", " ") for a in ACTIONS])
    s = s.classification("choice", CHOICE_LABELS, multi_label=True)
    return s.classification("requests", kind_labels(schema), multi_label=True)


def make_schema(model, schema: dict, kinds: list[str]):
    """The requests of ``kinds`` (stage 2). GLiNER2 is trained only on the structures a message has, so a
    schema of every kind would ask its count head about kinds it never saw absent."""
    s = model.create_schema()
    for kind in kinds:
        sk, names = schema["per_kind"][kind], field_names(schema, kind)
        b = s.structure(kind_text(kind))
        for r in sk["roles"]:
            b = b.field(names[("role", r)], dtype="list")
        for f, vals in sk["classes"].items():
            b = b.field(names[("class", f)], dtype="str", choices=[value_text(f, x) for x in vals])
        for f in sk["refs"]:
            b = b.field(names[("ref", f)], dtype="str", choices=["no", "yes"])
    return s


def _val(x):
    """A formatted value: plain, or {"text"/"label", "confidence", "start", ...} when asked for."""
    if isinstance(x, dict):
        v = x.get("text", x.get("label", x.get("value")))
        return v, float(x.get("confidence", 1.0)), x.get("start")
    return x, 1.0, None


def from_result(row: dict, res: dict, schema: dict) -> dict:
    """GLiNER2's result for one message -> our prediction format, with conf (the least sure decision)."""
    off = len(sender(row))
    confs = []
    act, c, _ = _val(res.get("action"))
    confs.append(c)
    action = (act or "compile").replace(" ", "_")
    chosen = res.get("choice") or []
    comps = []
    for x in chosen if isinstance(chosen, list) else [chosen]:
        v, c, _ = _val(x)
        confs.append(c)
        if v and v != "none":
            comps.append(v.replace(" ", "_"))
    intents = []
    for kind in schema["kinds"]:
        sk, names = schema["per_kind"][kind], field_names(schema, kind)
        for inst in res.get(kind_text(kind)) or []:
            spans, classes, refs, starts = [], {}, {}, []
            for r in sk["roles"]:
                vals = inst.get(names[("role", r)]) or []
                for x in vals if isinstance(vals, list) else [vals]:
                    v, c, st = _val(x)
                    if not v:
                        continue
                    confs.append(c)
                    a = (st - off) if isinstance(st, int) else row["text"].find(v)
                    starts.append(a if a >= 0 else 10 ** 6)
                    spans.append({"role": r, "start": a, "end": a + len(v), "text": v})
            for f, vals in sk["classes"].items():
                v, c, _ = _val(inst.get(names[("class", f)]))
                confs.append(c)
                back = {value_text(f, x): x for x in vals}
                if v in back and back[v] != "none":
                    classes[f] = json.loads(back[v])
            for f in sk["refs"]:
                v, c, _ = _val(inst.get(names[("ref", f)]))
                confs.append(c)
                if v == "yes":
                    refs[f] = True
            intents.append({"kind": kind, "spans": spans, "classes": classes, "refs": refs,
                            "extent": [min(starts, default=10 ** 6), 0]})
    intents.sort(key=lambda i: i["extent"][0])
    return {"id": row["id"], "action": action, "composites": sorted(comps), "intents": intents,
            "conf": round(min(confs, default=0.0), 4)}


def evaluate(model, schema: dict, rows: list[dict], bs: int, threshold: float) -> tuple[dict, list[dict]]:
    t1 = time.time()
    texts = [sender(r) + r["text"] for r in rows]
    first = model.batch_extract(texts, stage1_schema(model, schema), batch_size=bs, threshold=threshold,
                                include_confidence=True)
    kinds_of, res = [], []
    for x in first:
        got = x.get("requests") or []
        got = [_val(v)[0] for v in (got if isinstance(got, list) else [got])]
        kinds_of.append([k for k in schema["kinds"] if kind_text(k) in got])
    todo = [i for i, ks in enumerate(kinds_of) if ks]
    second = model.batch_extract([texts[i] for i in todo], [make_schema(model, schema, kinds_of[i]) for i in todo],
                                 batch_size=bs, threshold=threshold, include_confidence=True,
                                 include_spans=True) if todo else []
    by = dict(zip(todo, second))
    for i, x in enumerate(first):
        res.append({**x, **by.get(i, {})})
    ms = 1000 * (time.time() - t1) / max(len(rows), 1)
    preds = [from_result(r, x, schema) for r, x in zip(rows, res, strict=True)]
    sc = score(rows, preds)
    sc["ms_per_row"] = round(ms, 1)
    for p, g, x in zip(preds, rows, res):
        p["ok"] = correct(g, p)
        p["raw"] = x
    return sc, preds


def main2() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--scratch", default=None, help="smoke test: a randomly initialised GLiNER2 on this encoder")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--task_lr", type=float, default=5e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--zero_shot", type=int, default=0)  # run 1: 0.5% val exact, 2.6% test
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import torch
    from gliner2 import GLiNER2
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

    if args.data is None:
        hits = [h for h in sorted(Path("/kaggle/input").rglob("train.jsonl")) if (h.parent / "composed.jsonl").exists()]
        args.data = str(hits[0].parent)
    data, out = Path(args.data), Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    t0 = time.time()

    def say(**kw):
        kw["t"] = round(time.time() - t0, 1)
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    lim = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    train, val = lim(load(data / "train.jsonl")), lim(load(data / "val.jsonl"))
    schema = build_schema(train)
    (out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.scratch:
        from gliner2 import ExtractorConfig
        model = GLiNER2(ExtractorConfig(model_name=args.scratch, architecture="span"))
    else:
        model = GLiNER2.from_pretrained(args.model)
    model = model.to(device)
    for p in model.parameters():  # safetensors will not save a non-contiguous weight (the best checkpoint)
        p.data = p.data.contiguous()
    splits = ["test", "composed"] + sorted(p.stem for p in data.glob("suite-*.jsonl"))
    metrics = {"args": vars(args)}

    if args.zero_shot and not args.scratch:  # the video's claim: labels in, no training
        model.eval()
        for name in ["val", "test", "composed"]:
            rows = lim(load(data / f"{name}.jsonl"))
            sc, _ = evaluate(model, schema, rows, args.bs * 4, args.threshold)
            metrics[f"zero_shot/{name}"] = sc
            say(event="zero_shot", split=name, exact=sc["exact"], action=sc["action"], tiers=sc["by_tier"])

    exs = [e for r in train if usable(r) for e in to_examples(r, schema)]
    bad = [e for e in exs if e.validate()]
    say(event="data", device=device, model=args.scratch or args.model, examples=len(exs), invalid=len(bad),
        first_error=(bad[0].validate()[:3] if bad else None))
    exs = [e for e in exs if not e.validate()]
    vex = [e for r in val if usable(r) for e in to_examples(r, schema) if not e.validate()]
    cfg = TrainingConfig(output_dir=str(out / "gliner2-ckpt"), num_epochs=args.epochs, batch_size=args.bs,
                         eval_batch_size=args.bs * 2, encoder_lr=args.encoder_lr, task_lr=args.task_lr,
                         warmup_ratio=0.06, scheduler_type="linear", eval_strategy="epoch", save_best=True,
                         metric_for_best="eval_loss", save_total_limit=1, logging_steps=100, num_workers=args.workers,
                         fp16=device == "cuda", bf16=False, seed=13)
    trainer = ExtractorTrainer(model=model, config=cfg, train_data=exs, eval_data=vex)
    summary = trainer.train()
    say(event="trained", summary={k: v for k, v in (summary or {}).items() if not isinstance(v, (list, dict))})
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model.eval()
    for name in ["val"] + splits:
        path = data / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = lim(load(path))
        sc, preds = evaluate(model, schema, rows, args.bs * 4, args.threshold)
        metrics[name] = sc
        with (out / "predictions" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
        say(event="eval", split=name, exact=sc["exact"], action=sc["action"], tiers=sc["by_tier"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    say(event="done")


if __name__ == "__main__":
    main2()
