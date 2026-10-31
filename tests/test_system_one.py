"""System One's classifiers (language.system_one) and their benchmark (evaluation.system_one).

The gates at the end run the benchmark on the real corpus and intent sets: they are the bar a change to
System One must keep. Gates for the embedding models are skipped when the bge-small model cannot be loaded.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from core.generator import generate_department
from core.schemas import Channel, Request, Role
from evaluation.system_one import INTENT_SETS, benchmark, score
from language.corpus import ExpectedAction, RequestType, generate_corpus, load_jsonl
from language.system_one import (DEFAULT_MODEL, FAST_PATH_TYPES, INTENT_TYPES, MODELS, Decision, Labelled, SystemOne,
                                 intent_examples, read_intents)

DATA = Path(__file__).resolve().parents[1] / "data"
TEXT_MODELS = ("logreg", "svm")  # TF-IDF only: nothing to download


def _embeddings_available() -> bool:
    try:
        from agents.policy import embedder

        list(embedder().embed(["probe"]))
        return True
    except Exception:
        return False


needs_embeddings = pytest.mark.skipif(not _embeddings_available(), reason="bge-small embeddings not available")


def _models():
    return [m if m in TEXT_MODELS else pytest.param(m, marks=needs_embeddings) for m in MODELS]


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(generate_department(seed=0), n=600, seed=0)


@pytest.fixture(scope="module")
def app_training():
    """What the web app trains System One on (web.world): both corpus files and the intent kinds they lack."""
    corpus = load_jsonl(DATA / "requests.jsonl") + load_jsonl(DATA / "requests-para.jsonl")
    return [e for e in corpus if e.split == "train"] + intent_examples("train")


def _request(text: str, role: Role) -> Request:
    return Request(id="R-T", channel=Channel.PORTAL, sender_id="X-1", role=role, raw_text=text,
                   received_at=datetime(2026, 9, 1))


def test_the_default_is_the_linear_svm():
    assert DEFAULT_MODEL == "svm" and SystemOne().model == "svm"


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown System One model"):
        SystemOne(model="forest")


@pytest.mark.parametrize("model", _models())
def test_every_model_learns_the_corpus(corpus, model):
    s1 = SystemOne(model=model).fit([e for e in corpus if e.split == "train"])
    val = [e for e in corpus if e.split == "val"]
    decisions = [s1.decide(e.request) for e in val]
    assert sum(d.request_type == e.request_type for d, e in zip(decisions, val)) / len(val) > 0.9
    assert sum(d.action == e.expected_action for d, e in zip(decisions, val)) / len(val) > 0.9
    for d in decisions:
        assert 0.0 <= d.type_p <= 1.0 and 0.0 <= d.action_p <= 1.0
    # the median: one embedding call can stall on a busy machine (it failed a full-suite run once)
    assert sorted(d.latency_ms for d in decisions)[len(decisions) // 2] < 100


STUDENT_ASKS = ["Please move the Tuesday 10 am lecture to Thursday.", "Can you cancel the Friday lab this week?",
                "I want no classes on Monday mornings.", "Please shift my class to the afternoon."]


@pytest.mark.parametrize("model", [
    pytest.param("logreg", marks=pytest.mark.xfail(strict=True, reason="lets \"no classes on Monday mornings\" "
                                                   "from a student through as compile")),
    *[m for m in _models() if m != "logreg"]])
def test_a_student_asking_for_a_change_is_refused(app_training, model):
    """The same words: a change for the teacher who owns the class, a refusal from a student."""
    s1 = SystemOne(model=model).fit(app_training)
    for text in STUDENT_ASKS:
        assert s1.decide(_request(text, Role.STUDENT)).action == ExpectedAction.REFUSE, text
    assert s1.decide(_request(STUDENT_ASKS[0], Role.FACULTY)).action == ExpectedAction.COMPILE


# -- the kinds the request corpus has no example of (improvement A1) -------------------------------------------

def test_intent_examples_cover_every_missing_kind_and_teach_the_type_only():
    train = intent_examples("train")
    assert {e.kind for e in train} == set(INTENT_TYPES)
    assert all(e.request_type == INTENT_TYPES[e.kind] and e.expected_action is None for e in train)
    assert not any(t in FAST_PATH_TYPES for t in INTENT_TYPES.values())
    assert intent_examples("val") and intent_examples("no-such-split") == []


def test_intent_test_sets_are_read_with_their_senders_and_labels():
    cases = read_intents(INTENT_SETS["intents"], "test")
    assert len(cases) == 417
    assert all(c.request.raw_text and "<message>" not in c.request.raw_text for c in cases)
    assert {c.kind for c in cases if c.kind in INTENT_TYPES} >= {"swap", "withdraw", "club_booking", "cancel"}
    for c in cases:
        assert c.request_type is not None and c.expected_action is not None, c.kind
    assert {c.expected_action for c in cases if c.kind == "withdraw"} == {ExpectedAction.WITHDRAW}


def test_an_example_without_an_action_trains_only_the_type_model(corpus):
    train = [e for e in corpus if e.split == "train"]
    swaps = [Labelled(_request(f"Could I swap my {d} class with Dr. Rao's?", Role.FACULTY), RequestType.SWAP,
                      None, "swap") for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")] * 4
    s1 = SystemOne(model="logreg").fit(train + swaps)
    assert RequestType.SWAP.value in s1.type_model.classes_
    assert set(s1.action_model.classes_) == {e.expected_action.value for e in train}


@pytest.mark.parametrize("model", TEXT_MODELS)
def test_a_swap_stays_off_the_fast_path(app_training, model):
    s1 = SystemOne(model=model).fit(app_training)
    d = s1.decide(_request("Can I swap my Wednesday 11 am lecture with Dr. Iyer's Friday one?", Role.FACULTY))
    assert d.request_type == RequestType.SWAP
    assert not d.fast_path(0.3)


# -- the benchmark's scoring ------------------------------------------------------------------------------------

class _Says:
    """A System One stand-in that always gives one decision."""

    def __init__(self, action: ExpectedAction, rtype: RequestType = RequestType.PREFERENCE):
        self.decision = Decision(rtype, 0.9, action, 0.9, 1.0)

    def decide(self, request):
        return self.decision


def test_deny_counts_as_compile_only_on_the_intent_sets():
    """The intent data labels a rule-breaking request compile (the policy agent denies it)."""
    case = Labelled(_request("Put my class in the lunch hour.", Role.FACULTY), RequestType.PREFERENCE,
                    ExpectedAction.COMPILE, "preference")
    assert score(_Says(ExpectedAction.DENY), 0.5, [case], intents=True)["action_accuracy"] == 1.0
    assert score(_Says(ExpectedAction.DENY), 0.5, [case], intents=False)["action_accuracy"] == 0.0


def test_new_kinds_sent_to_the_fast_path_count_as_a_leak():
    known = Labelled(_request("No classes on Friday please.", Role.FACULTY), RequestType.PREFERENCE,
                     ExpectedAction.COMPILE, "preference")
    swap = Labelled(_request("Swap my Monday class with Dr. Rao's.", Role.FACULTY), RequestType.SWAP,
                    ExpectedAction.COMPILE, "swap")
    m = score(_Says(ExpectedAction.COMPILE), 0.5, [known, swap], intents=True)
    assert m["fast_path_leak"] == 1.0 and m["leaked_kinds"] == {"swap": 1} and m["new_kinds"] == 1
    assert m["new_kind_type_accuracy"] == 0.0
    assert m["fast_path_precision"] == 1.0  # the swap is not one of the scored kinds
    m = score(_Says(ExpectedAction.COMPILE, RequestType.SWAP), 0.5, [swap], intents=True)
    assert m["fast_path_leak"] == 0.0 and m["new_kind_type_accuracy"] == 1.0


def test_benchmark_reports_every_metric(corpus):
    results = benchmark(["logreg"], ["corpus", "para", "unseen-wording"], templates=corpus, paraphrases=corpus,
                        intent_labels=False)
    for setting in ("corpus", "para", "unseen-wording"):
        m = results["logreg"][setting]
        for key in ("type_accuracy", "action_accuracy", "action_macro_f1", "action_ece", "fast_path_share",
                    "fast_path_precision", "fast_path_leak", "new_kind_type_accuracy", "tau", "latency_ms_p50",
                    "latency_ms_p95", "fit_seconds"):
            assert key in m
        assert m["type_accuracy"] > 0.9  # the generated test split's actions are harder than data/requests.jsonl's


# -- gates on the real data: what a change to System One must keep --------------------------------------------

GATE_SETTINGS = ["para", "unseen-wording", "intents", "typo", "noisy"]


@pytest.fixture(scope="module")
def measured():
    models = [m for m in MODELS if m in TEXT_MODELS or _embeddings_available()]
    return benchmark(models, GATE_SETTINGS)


# the measured action accuracy less a small margin (runs/system_one-a1.json)
GATES = {
    "logreg": {"unseen-wording": 0.90, "intents": 0.95, "typo": 0.93, "noisy": 0.95},
    "svm": {"unseen-wording": 0.94, "intents": 0.98, "typo": 0.98, "noisy": 0.98},
    "embed": {"unseen-wording": 0.89, "intents": 0.97, "typo": 0.95, "noisy": 0.96},
    "hybrid": {"unseen-wording": 0.92, "intents": 0.98, "typo": 0.98, "noisy": 0.98},
}


@pytest.mark.slow
@pytest.mark.parametrize("model", _models())
def test_action_accuracy_holds_on_unseen_and_noisy_wording(measured, model):
    for setting, floor in GATES[model].items():
        assert measured[model][setting]["action_accuracy"] >= floor, (model, setting)


@pytest.mark.slow
def test_the_default_is_at_least_as_good_as_logistic_regression(measured):
    for setting in GATE_SETTINGS:
        assert measured["svm"][setting]["action_accuracy"] >= measured["logreg"][setting]["action_accuracy"], setting
        # calibration: tied on unseen wording (0.091 and 0.090), better everywhere else
        assert measured["svm"][setting]["action_ece"] <= measured["logreg"][setting]["action_ece"] + 0.01, setting


@pytest.mark.slow
@pytest.mark.parametrize("model", _models())
def test_confidence_is_calibrated(measured, model):
    for setting in GATE_SETTINGS:
        assert measured[model][setting]["action_ece"] <= 0.18, (model, setting)


@pytest.mark.slow
@pytest.mark.parametrize("model", _models())
def test_every_message_is_classified_quickly(measured, model):
    for setting in GATE_SETTINGS:
        assert measured[model][setting]["latency_ms_p95"] < 50, (model, setting)


@pytest.mark.slow
def test_most_new_kinds_stay_off_the_fast_path(measured):
    """Before A1 the default sent 87% of them there; swaps, cancellations, extra classes, club bookings,
    withdrawals and avoid-between requests now never go."""
    for setting in ("intents", "typo", "noisy"):
        m = measured["svm"][setting]
        assert m["fast_path_leak"] <= 0.25, setting
        assert set(m["leaked_kinds"]) <= {"load_limit", "unavailable_relative", "move"}, setting


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="the test set words load limits and relative absences in ways no training "
                                       "row does (\"next week I won't...\" vs \"from week 9...\")")
def test_no_new_kind_takes_the_fast_path(measured):
    assert measured["svm"]["intents"]["fast_path_leak"] <= 0.05
