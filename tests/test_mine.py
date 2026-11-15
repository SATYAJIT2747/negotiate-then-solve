"""Mining training rows from the local model's mistakes. No server: a stub model that is always wrong."""

import json
from pathlib import Path

import pytest

from agents.policy import PolicyOutput
from core.instance import Instance
from core.schemas import Request
from language.intents import IntentOutput, compile_intents
from training import mine
from training.distill import DIR, _read, canonical_answers

DATA = Path("data")


@pytest.fixture(scope="module")
def instance():
    return Instance.model_validate_json((DATA / "synthetic-cse-s0.json").read_text(encoding="utf-8"))


class Stub:
    model = "stub"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, system, user, schema):
        return self.answer


def test_wrong_fields_name_where_the_answer_differs():
    gold = {"action": "compile", "intents": [{"kind": "avoid", "hard": True, "time": {"days": ["Tue"]}}]}
    pred = {"action": "compile", "intents": [{"kind": "avoid", "time": {"days": ["Wed"]}}]}
    assert mine.wrong_fields(pred, gold) == ["intents.0.hard", "intents.0.time.days.0"]
    assert mine.wrong_fields(None, gold) == ["*"]
    assert mine.field(gold, "intents.0.hard") is True and mine.field(gold, "intents.3.hard") is None


def test_the_intent_pool_never_holds_a_test_wording(instance):
    pool = mine.intent_pool(instance, 6, seed=7)
    assert {ex.template for k, ex in pool if k in mine.GENERATORS} <= set(mine.TRAIN_TEMPLATES)
    assert all(ex.split == "train" and ex.id.startswith("M7-") for _, ex in pool)


def test_mined_intent_rows_keep_the_gold_with_twins_and_contrasts(instance):
    rows, report = mine.mine_intents(instance, Stub(IntentOutput(action="out_of_scope")), pool=4, bank=10, twins=1,
                                     contrast=1, seed=7, held=set(), workers=1)
    s = report["summary"]
    assert s["failures"] == s["pool"] > 0 and s["as"]["failure"] == s["failures"]
    assert s["as"].get("variant", 0) > 0 and s["as"].get("contrast", 0) > 0
    for r in rows:  # every target is the generator's gold, and it still compiles
        gold = IntentOutput.model_validate_json(r["messages"][2]["content"])
        assert gold.action != "out_of_scope"
        got = compile_intents(instance, Request.model_validate(r["request"]), gold, current_week=r["current_week"])
        assert got.result.action.value != "out_of_scope"
    failures = {r["mined"]["from"] for r in rows if r["mined"]["as"] == "failure"}
    assert all(r["mined"]["from"] in failures for r in rows)


def test_mined_policy_rows_take_the_canonical_answer_and_the_other_verdict_beside(instance):
    answers = canonical_answers(_read(DIR / "policy.jsonl"))
    lax = Stub(PolicyOutput(verdict="allowed", explanation="Fine."))  # misses every denial
    rows, report = mine.mine_policy(instance, lax, DATA / "handbook.md", answers, pool=12, twins=1, contrast=1,
                                    seed=7, held=[], workers=1)
    s = report["summary"]
    assert s["failures"] > 0 and all(f["gold"] == "deny" for f in report["failures"])
    verdicts = {r["mined"]["as"]: [] for r in rows}
    for r in rows:
        verdicts[r["mined"]["as"]].append(json.loads(r["messages"][2]["content"])["verdict"])
    assert set(verdicts["failure"]) == {"forbidden"}
    assert set(verdicts.get("contrast", [])) == {"allowed"}  # the legal look-alike of the same rule
