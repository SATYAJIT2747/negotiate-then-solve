"""Messages written and labelled by people: the data no generator can make.

Real requests are the measure that matters (typos, wordings nobody templated, several asks at once), and
the most valuable rows to train on once there are enough. They live in ``data/human/*.jsonl``, one labelled
message per line:

    {"id": "H-0001", "text": "pls move my CA lec to fri am", "sender_id": "F-121", "role": "faculty",
     "current_week": 7, "today": "2026-09-14", "timetable": {"C-054-L1": {"day": "Wed", "slot": 6, "room": "R1"}},
     "gold": {"action": "compile", "intents": [...]}, "schema": 3, "reviewed": true, "annotator": "AB",
     "split": "human"}

* ``gold`` is an ``IntentOutput`` labelled by ``docs/labelling.md``; ``schema`` the vocabulary version it
  was labelled in (``language.migrations`` brings older labels up to date);
* ``timetable`` (optional) places the sender's classes, so "my Wednesday 3 pm lecture" resolves; without it
  the published timetable is not known and only references by course resolve;
* ``split``: ``human`` (the default) keeps the row for evaluation only; ``train`` or ``val`` lets a
  fine-tuning run use it. Keep most for evaluation until there are a few hundred;
* only ``reviewed`` rows are used. ``prefill`` writes unreviewed rows with the model's reading as the
  gold, for a person to correct and mark reviewed: correcting is faster than writing JSON.

    uv run python -m training.human prefill --in inbox.jsonl --out data/human/batch-01.jsonl --local URL
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from core.instance import Instance
from core.schemas import Channel, Placement, Request, Role
from language.intent_compiler import _chat, _meta, intent_message
from language.intents import SCHEMA_VERSION, IntentOutput
from language.migrations import migrate


def request_of(h: dict) -> Request:
    return Request(id=h["id"], channel=Channel(h.get("channel", "portal")), sender_id=h["sender_id"],
                   role=Role(h.get("role", "faculty")), raw_text=h["text"], received_at=datetime(2026, 9, 1))


def to_row(instance: Instance, h: dict) -> dict:
    """A labelled message as a training row (``intent_compiler`` format), its gold in the current schema."""
    request = request_of(h)
    timetable = {sid: Placement.model_validate(p) for sid, p in (h.get("timetable") or {}).items()}
    today = date.fromisoformat(h["today"]) if h.get("today") else None
    gold = IntentOutput.model_validate(migrate(h["gold"], h.get("schema")))
    mine = {sid: p for sid, p in timetable.items()
            if sid in instance.session_by_id and instance.session_by_id[sid].faculty == request.sender_id}
    return {"id": h["id"], "source": "human", "kind": h.get("kind") or (gold.intents[0].kind if gold.intents
                                                                          else gold.action),
            "schema": SCHEMA_VERSION, "annotator": h.get("annotator"), "paraphrased": False,
            "messages": _chat(instance, intent_message(instance, request, mine, h.get("current_week"), today), gold),
            **_meta(request, timetable, h.get("current_week"), today)}


def load(instance: Instance, paths: list[Path]) -> list[tuple[str, dict]]:
    """``(split, row)`` for every reviewed message in ``paths``."""
    out = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and (h := json.loads(line)).get("reviewed"):
                out.append((h.get("split", "human"), to_row(instance, h)))
    return out


def prefill(instance: Instance, client, rows: list[dict]) -> list[dict]:
    """Each message with the model's reading as a gold to correct (``reviewed`` false)."""
    from language.intent_compiler import read_answer

    out = []
    for h in rows:
        request = request_of(h)
        timetable = {sid: Placement.model_validate(p) for sid, p in (h.get("timetable") or {}).items()}
        try:
            schema = getattr(client, "schema", None) or SCHEMA_VERSION
            reading, notes = read_answer(client, intent_message(instance, request, timetable, h.get("current_week"),
                                                                schema=schema))
            gold = json.loads(reading.model_dump_json(exclude_defaults=True))
        except Exception as e:  # noqa: BLE001 - a message the model cannot read is labelled from scratch
            gold, notes = None, [f"{type(e).__name__}: {e}"[:200]]
        out.append({**h, "gold": gold, "schema": SCHEMA_VERSION, "reviewed": False, "prefill_notes": notes})
    return out


def main() -> None:
    from language.adapters import add_args, local_client, registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prefill"])
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="JSONL: id, text, sender_id, role, ...")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--local", metavar="URL", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None)
    add_args(ap)
    args = ap.parse_args()
    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    client = local_client("compile", base_url=args.local, schema=args.schema, server_model=args.local_model, adapter=args.adapter,
                          reg=registry(args.adapters))
    rows = [json.loads(x) for x in args.inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = prefill(instance, client, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done), encoding="utf-8")
    print(f"{len(done)} messages prefilled to {args.out}: correct each gold and set reviewed to true")


if __name__ == "__main__":
    main()
