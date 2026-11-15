import json
from pathlib import Path

from core.instance import Instance
from language.intent_corpus import TEST_TEMPLATE
from training import balance

ROOT = Path(__file__).resolve().parents[1]


def row(kind: str, *intents: str) -> dict:
    out = {"action": "compile", "intents": [{"kind": k} for k in intents]}
    return {"kind": kind, "messages": [{}, {}, {"role": "assistant", "content": json.dumps(out)}]}


def test_a_kind_taught_mostly_by_messages_of_several_requests_is_short():
    rows = ([row("cancel", "cancel")] * 2 + [row("compound", "cancel", "avoid")] * 3
            + [row("avoid_between", "avoid")] * 9)
    assert balance.short(rows, 0.5) == {"cancel": 4}
    r = balance.report(rows)
    assert (r["cancel"]["alone"], r["cancel"]["part"], r["cancel"]["top"]) == (2, 3, {"compound": 0.6, "cancel": 0.4})


def test_the_top_up_is_the_kind_alone_from_its_own_training_wordings():
    instance = Instance.model_validate_json((ROOT / "data" / "synthetic-cse-s0.json").read_text(encoding="utf-8"))
    rows = [row("compound", "cancel", "only")] * 6  # "only" has no generator of its own: never topped up
    pairs, need = balance.top_up(instance, rows, {"max_parts": 0.5, "seed": "t"})
    assert need == {"cancel": 12}
    train = [r for s, r in pairs if s == "train"]
    assert len(train) >= 12 and {s for s, _ in pairs} <= {"train", "val"}
    assert all(r["kind"] == "cancel" and r["template"] != TEST_TEMPLATE for _, r in pairs)
    assert balance.short(rows + train, 0.5) == {}
