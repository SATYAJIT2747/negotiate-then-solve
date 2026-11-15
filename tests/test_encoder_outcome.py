"""The encoder normaliser (``evaluation.encoder_outcome``): gold spans decode back to the gold reading."""

import json
from pathlib import Path

import pytest

from core.instance import Instance
from evaluation.encoder_outcome import Normaliser, Preds, gold_as_pred, key, load
from evaluation.intents import score

SPANS = Path("data/encoder-intents/spans")
INSTANCE = Path("data/synthetic-cse-s0.json")
pytestmark = pytest.mark.skipif(not SPANS.exists(), reason="python -m training.encoder_spans first")


@pytest.fixture(scope="module")
def norm() -> Normaliser:
    return Normaliser(Instance.model_validate_json(INSTANCE.read_text(encoding="utf-8")))


def test_values(norm):
    assert norm.value("which.day", " Thursdy") == "Thu"
    assert norm.value("to.at", " 3 pm.") == "15:00"
    assert norm.value("time.after", "2") == "14:00"  # a bare hour, in the teaching day
    assert norm.value("n", " two") == 2
    assert norm.value("which.kind", "lab") == "practical"
    assert norm.value("equipment", "GPU") == "gpu"
    assert norm.value("which.course", "Discrete Mathematics") == "Discrete Mathematics"


def test_range(norm):
    text = "I'm at a conference in week 13, Monday to Wednesday."
    a, b = text.index("Monday"), text.index("Wednesday")
    it = {"kind": "unavailable", "classes": {}, "refs": {},
          "spans": [{"role": "time.days", "start": a, "end": a + 6, "text": "Monday"},
                    {"role": "time.days", "start": b, "end": b + 9, "text": "Wednesday"},
                    {"role": "period.weeks", "start": 28, "end": 30, "text": "13"}]}
    assert norm.intent(text, it)["time"]["days"] == ["Mon", "Tue", "Wed"]


@pytest.mark.parametrize("split,data", [("test", Path("data/compiler-intents/test.jsonl")),
                                        ("composed", Path("data/encoder-intents/composed.jsonl"))])
def test_roundtrip(norm, split, data):
    """Gold spans as a prediction: the same outcome as the gold on every row of the test split and the
    complex suite (refs and composites rebuilt from the spans' order)."""
    rows = load(data)
    got = [gold_as_pred(r) for r in load(SPANS / f"{split}.jsonl")]
    preds = {key(r): p for r, p in zip(rows, got, strict=True)}
    res = score(rows, norm.instance, Preds(norm.instance, preds, norm))
    bad = [(r["id"], r["text"][:80]) for r in res["rows"] if not r["match"]]
    assert not bad, json.dumps(bad[:5])
