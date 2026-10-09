from collections import Counter, defaultdict

import pytest

from core.generator import generate_department
from core.instance import policy_constraints
from core.schemas import Role
from core.solver import TimetableSolver
from core.validator import validate_constraint
from language.corpus import (
    CorpusExample,
    ExpectedAction,
    RequestType,
    Variant,
    core_text,
    generate_corpus,
    load_jsonl,
    multi_constraint_examples,
    save_jsonl,
    split_group,
    tag_slices,
)


@pytest.fixture(scope="module")
def instance():
    return generate_department(seed=0)


@pytest.fixture(scope="module")
def corpus(instance):
    return generate_corpus(instance, n=600, seed=0)


def test_size_ids_and_splits(corpus):
    assert len(corpus) == 600
    assert [ex.id for ex in corpus] == [f"R-{i:04d}" for i in range(1, 601)]
    splits = Counter(ex.split for ex in corpus)
    assert abs(splits["train"] - 400) <= 10 and abs(splits["val"] - 50) <= 10 and abs(splits["test"] - 150) <= 10
    assert len({ex.request.raw_text for ex in corpus}) == 600


def test_splits_never_divide_a_template_or_a_request(corpus):
    """Leakage (proposal Section 13.1): a wording template, and any request
    that differs only in greeting or sign-off, lives in exactly one split."""
    by_group, by_text = defaultdict(set), defaultdict(set)
    for ex in corpus:
        by_group[split_group(ex)].add(ex.split)
        by_text[core_text(ex.request.raw_text)].add(ex.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert all(len(s) == 1 for s in by_text.values())
    test = [ex for ex in corpus if ex.split == "test"]
    assert {ex.request_type for ex in test} >= {RequestType.PREFERENCE, RequestType.UNAVAILABILITY,
                                                RequestType.ROOM_ISSUE}
    assert all(ex.split != "train" or not ex.slices for ex in corpus)
    tagged = Counter(s for ex in test for s in ex.slices)
    assert tagged["unseen_wording"] > 50 and tagged["ambiguous"] and tagged["adversarial"]


def test_multi_constraint_requests_are_test_only_and_unseen(instance, corpus):
    multi = multi_constraint_examples(instance, 10, 0, corpus)
    assert len(multi) == 10
    train_groups = {split_group(ex) for ex in corpus if ex.split == "train"}
    for ex in multi:
        assert ex.split == "test" and len(ex.targets) == 2 and " Also, " in ex.request.raw_text
        assert all(t.source.request == ex.id and not validate_constraint(t, instance) for t in ex.targets)
        assert all(part not in train_groups for part in ex.template.split("+"))
    both = [*corpus, *multi]
    tag_slices(both)
    assert all({"multi_constraint", "unseen_combination"} <= set(ex.slices) for ex in both[len(corpus):])


def test_deterministic(instance, corpus):
    again = generate_corpus(instance, n=600, seed=0)
    assert [e.model_dump() for e in again] == [e.model_dump() for e in corpus]
    other = generate_corpus(instance, n=600, seed=1)
    assert [e.request.raw_text for e in other] != [e.request.raw_text for e in corpus]


def test_every_kind_is_present(corpus):
    variants = Counter(ex.variant for ex in corpus)
    actions = Counter(ex.expected_action for ex in corpus)
    assert set(variants) == set(Variant)
    assert set(actions) == set(ExpectedAction) - {ExpectedAction.WITHDRAW}  # an intent parser's action only
    assert variants[Variant.CLEAN] > 300


def test_labels_are_consistent(corpus):
    for ex in corpus:
        compiles = ex.expected_action in (ExpectedAction.COMPILE, ExpectedAction.CLARIFY)
        assert bool(ex.targets) == compiles, ex.id
        assert (ex.expected_action == ExpectedAction.DENY) == (ex.violates_rule is not None), ex.id
        assert ex.authorised == (ex.expected_action != ExpectedAction.REFUSE), ex.id
        if ex.variant == Variant.AMBIGUOUS:
            assert ex.missing
        for t in ex.targets:
            assert t.source.request == ex.id and t.id.startswith(f"T-{ex.id[2:]}-")
            if t.owner is not None:
                assert t.owner == ex.request.sender_id
        if ex.request.role == Role.STUDENT:
            assert not ex.targets


def test_private_reasons_stay_out_of_targets(corpus):
    for ex in corpus:
        if "medical" in ex.request.raw_text or "hospital" in ex.request.raw_text:
            dumped = " ".join(t.model_dump_json() for t in ex.targets)
            assert "medical" not in dumped and "hospital" not in dumped


def test_injected_legit_requests_still_compile(corpus):
    injected = [ex for ex in corpus if ex.injection]
    assert injected
    assert any(ex.expected_action == ExpectedAction.COMPILE for ex in injected)
    assert any(ex.expected_action == ExpectedAction.REFUSE for ex in injected)


@pytest.mark.slow
def test_clean_targets_are_solvable_one_at_a_time(instance, corpus):
    """Each compiled request on its own is satisfiable on the department."""
    base = policy_constraints(instance)
    sample = [ex for ex in corpus if ex.variant == Variant.CLEAN and ex.targets][:4]
    for ex in sample:
        week = next((t.valid.from_week for t in ex.targets if t.valid.from_week), None)
        solver = TimetableSolver(instance, base + ex.targets, week=week, time_limit=60)
        assert solver.solve().ok, ex.id


def test_jsonl_round_trip(tmp_path, instance, corpus):
    path = tmp_path / "requests.jsonl"
    save_jsonl(corpus, path)
    loaded = load_jsonl(path, instance)
    assert [e.model_dump() for e in loaded] == [e.model_dump() for e in corpus]


def test_handwritten_example_with_bad_reference_is_rejected(tmp_path, instance, corpus):
    bad = corpus[next(i for i, ex in enumerate(corpus) if ex.targets)].model_copy(deep=True)
    bad.targets[0].scope.faculty, bad.targets[0].owner = "F-999", None
    bad.handwritten = True
    path = tmp_path / "handwritten.jsonl"
    save_jsonl([bad], path)
    with pytest.raises(ValueError, match="F-999"):
        load_jsonl(path, instance)
    assert isinstance(load_jsonl(path)[0], CorpusExample)  # without an instance, no reference check
