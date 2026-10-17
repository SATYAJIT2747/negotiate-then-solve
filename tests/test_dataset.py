"""The training data: schema versions, the labelling rule, generators with held-out wordings, gates and the
frozen test split."""

import json
from pathlib import Path

import pytest

from core.instance import Instance
from language.intent_generators import REGISTRY, generated
from language.intents import SCHEMA_VERSION, IntentOutput, Period, Unresolved, weeks
from language.lexicon import placement_is_hard
from language.migrations import migrate, migrate_row
from training import quality
from training.dataset import build, fingerprint, load_recipe

ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "data" / "recipes" / "intents-v5.toml"  # the newest recipe: its frozen test split is the current one


@pytest.fixture(scope="module")
def instance():
    return Instance.model_validate_json((ROOT / "data" / "synthetic-cse-s0.json").read_text(encoding="utf-8"))


def test_a_version_one_move_was_a_requirement_and_stays_one():
    old = {"action": "compile", "intents": [{"kind": "move", "which": {"course": "C-1"}, "to": {"days": ["Fri"]}}]}
    assert migrate(old, 1)["intents"][0]["hard"] is True
    assert "hard" not in migrate(old, 2)["intents"][0]  # written in 2: a wish, as it says
    with pytest.raises(ValueError):
        migrate(old, SCHEMA_VERSION + 1)
    row = {"id": "x", "messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""},
                                   {"role": "assistant", "content": json.dumps(old)}]}
    got = migrate_row(row)
    assert got["schema"] == SCHEMA_VERSION and got["migrated_from"] == 1
    assert IntentOutput.model_validate_json(got["messages"][-1]["content"]).intents[0].hard is True


def test_weeks_counted_from_now(instance):
    n = instance.calendar.weeks
    assert weeks(instance, Period(relative="next_week", count=3), current_week=5) == [6, 7, 8]
    assert weeks(instance, Period(relative="this_week", count=2), current_week=5) == [5, 6]
    assert weeks(instance, Period(relative="rest_of_semester"), current_week=n - 2) == [n - 2, n - 1, n]
    assert weeks(instance, Period(relative="next_week", count=3), current_week=n - 1) == [n]  # what is left
    with pytest.raises(Unresolved):
        weeks(instance, Period(relative="next_week"), current_week=n)
    with pytest.raises(ValueError):
        Period(weeks=[3], count=2)


def test_the_labelling_rule_for_moves():
    assert not placement_is_hard("Please shift my lab on Monday at 2 pm to Friday.")
    assert not placement_is_hard("If possible, I'd like my 9 am class moved to Friday.")
    assert placement_is_hard("My lecture has to be at 1 pm on Tuesday, no other time works.")
    assert placement_is_hard("My Thursday lecture clashes with a committee meeting; could it go to Friday?")


def test_held_out_wordings_never_reach_training(instance):
    rows = generated(instance, per_generator=12, held_per_generator=6)
    for g in REGISTRY:
        mine = [r for r in rows if r.generator == g.name]
        assert {r.template for r in mine if r.split in ("train", "val")} <= set(g.train_templates), g.name
        assert {r.template for r in mine if r.split == g.suite} <= set(g.held_out), g.name
        assert any(r.split == g.suite for r in mine) and any(r.split == "train" for r in mine), g.name


def test_adding_a_generator_leaves_the_others_rows_alone(instance):
    one = generated(instance, per_generator=8, held_per_generator=4, names=["firmness"])
    both = generated(instance, per_generator=8, held_per_generator=4, names=["firmness", "multi"])
    assert [r.request.raw_text for r in one] == [r.request.raw_text for r in both if r.generator == "firmness"]


@pytest.fixture(scope="module")
def built():
    recipe = load_recipe(LATEST)
    recipe["generators"] |= {"per_generator": 10, "held_per_generator": 4}
    recipe["augment"]["fraction"] = 0.1
    return build(recipe)


def test_the_build_passes_every_gate_and_keeps_the_test_split(built):
    splits, result = built
    assert not any(result["failures"].values()), quality.dump_failures(result["failures"])
    m = result["manifest"]
    assert m["fingerprints"]["test"] == load_recipe(LATEST)["frozen"]["test"]
    assert {"firmness", "moves", "periods", "multi", "clubs", "test-noisy"} <= set(splits)
    assert all(r["schema"] == SCHEMA_VERSION for rows in splits.values() for r in rows)
    twins = [r for r in splits["train"] if r.get("origin") == "augment"]
    assert twins and all(r["parent_text"] != r["request"]["raw_text"] for r in twins)


def test_a_changed_test_split_stops_the_build():
    recipe = load_recipe(LATEST)
    recipe["frozen"]["test"] = "0" * 16
    recipe["generators"] |= {"per_generator": 2, "held_per_generator": 1, "names": ["firmness"]}
    recipe.pop("augment")
    with pytest.raises(SystemExit, match="test split changed"):
        build(recipe)


def test_the_gates_catch_a_bad_row(built, instance):
    splits, _ = built
    row = json.loads(json.dumps(next(r for r in splits["train"] if r["kind"] == "move" and r.get("origin") == "base")))
    out = IntentOutput.model_validate_json(row["messages"][-1]["content"])
    bad = out.model_copy(update={"intents": [out.intents[0].model_copy(update={"hard": not out.intents[0].hard})]})
    row["messages"][-1]["content"] = bad.model_dump_json()  # not canonical, and against the rule
    gates = {g for g, _ in quality.check_row(instance, row)}
    assert {"schema", "rubric"} <= gates
    test = splits["test"][0]
    leak = {**json.loads(json.dumps(test)), "id": "copy"}
    failures = quality.check(instance, {"train": [leak], "test": [test]}, held=["test"])
    assert failures["leak"]


def test_a_labelled_message_becomes_a_row(instance):
    from training.human import to_row

    s = next(x for x in instance.sessions if x.kind.value == "lecture")
    h = {"id": "H-1", "text": "no clases b4 11 on thrusday plz", "sender_id": s.faculty, "current_week": 4,
         "gold": {"action": "compile", "intents": [{"kind": "avoid", "time": {"days": ["Thu"], "before": "11:00"}}]},
         "reviewed": True}
    row = to_row(instance, h)
    assert row["schema"] == SCHEMA_VERSION and "thrusday" in row["messages"][1]["content"]
    assert not quality.check_row(instance, row)


def test_a_fingerprint_does_not_depend_on_order_or_the_system_prompt(built):
    splits, _ = built
    rows = splits["test"]
    shuffled = list(reversed(rows))
    other_prompt = [{**r, "messages": [{"role": "system", "content": "x"}, *r["messages"][1:]]} for r in rows]
    assert fingerprint(rows) == fingerprint(shuffled) == fingerprint(other_prompt)


def test_an_adapter_on_an_older_vocabulary_is_read_with_its_prompt_and_migrated(tmp_path):
    from language.adapters import Registry
    from language.intent_compiler import INTENT_PROMPT, INTENT_PROMPTS, read_answer

    class Fake:
        def __init__(self, schema):
            self.schema, self.system = schema, None

        def generate(self, system, prompt, schema):
            self.system = system
            return IntentOutput.model_validate({"action": "compile", "intents": [
                {"kind": "move", "which": {"course": "C-1"}, "to": {"days": ["Fri"]}}]})

    old, new = Fake(1), Fake(None)
    assert read_answer(old, "...")[0].intents[0].hard is True  # a version-1 move without hard was a must
    assert old.system == INTENT_PROMPTS[1] != INTENT_PROMPT
    assert read_answer(new, "...")[0].intents[0].hard is False and new.system == INTENT_PROMPT
    (tmp_path / "a.toml").write_text('base = "B"\n[adapters]\nnts-intents = ["compile"]\n[schemas]\nnts-intents = 2\n')
    assert Registry.load(tmp_path / "a.toml").schemas == {"nts-intents": 2}
