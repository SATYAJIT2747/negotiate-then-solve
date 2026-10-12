"""Training data on intents: gold converted from the corpus, the new requests, the prompt and the targets."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from core.instance import Instance
from core.schemas import Channel, Request, Role
from language.corpus import ExpectedAction, load_jsonl
from language.intent_compiler import (
    completion,
    gold_intents,
    intent_directory,
    random_timetable,
    spec_from,
)
from language.intent_corpus import GENERATORS, TEST_TEMPLATE, _resolves, intent_examples, train_only_examples
from language.intents import IntentOutput, TimeSpec
from language.local import grammar_schema
from language.parsing import postprocess, to_constraint

DATA = Path("data")


@pytest.fixture(scope="module")
def instance():
    return Instance.model_validate_json((DATA / "synthetic-cse-s0.json").read_text(encoding="utf-8"))


def test_corpus_gold_converts_and_compiles_back(instance):
    from language.compiler import gold_output

    examples = [ex for ex in load_jsonl(DATA / "requests.jsonl", instance)
                if ex.expected_action in (ExpectedAction.COMPILE, ExpectedAction.CLARIFY)][:120]
    for ex in examples:
        old = gold_output(ex, instance)
        targets = [to_constraint(d, ex.request, i) for i, d in enumerate(old.constraints, 1)]
        out = gold_intents(instance, ex.request, ex.expected_action, ex.request_type, targets, ex.missing,
                           postprocess(instance, ex.request, old).constraints, ex.request.raw_text)
        assert out is not None, ex.request.raw_text
        assert out.action == ex.expected_action.value


def test_times_are_worded_as_the_message_words_them(instance):
    assert spec_from(instance, ["Thu"], [0], "nothing before 10 am") == TimeSpec(days=["Thu"], before="10:00")
    assert spec_from(instance, ["Thu"], [0], "at 9 am please") == TimeSpec(days=["Thu"], at=["09:00"])
    assert spec_from(instance, None, [5, 6, 7]) == TimeSpec(part="afternoon")
    assert spec_from(instance, None, [2, 3]) == TimeSpec(after="11:00", before="13:00")


def test_targets_write_kind_first_and_parse_back():
    from language.intents import Avoid, LoadLimit

    out = IntentOutput(action="compile", intents=[Avoid(time=TimeSpec(days=["Fri"])), LoadLimit(max_consecutive=2)])
    text = completion(out)
    d = json.loads(text)
    assert [next(iter(i)) for i in d["intents"]] == ["kind", "kind"]
    assert "hard" not in d["intents"][0] and IntentOutput.model_validate_json(text) == out
    schema = grammar_schema(IntentOutput.model_json_schema())
    assert all(node["required"][0] == "kind" for name, node in schema["$defs"].items()
               if "kind" in node.get("properties", {}) and name != "ClassRef")


def test_the_prompt_says_where_the_senders_classes_are(instance):
    import random

    f = next(x for x in instance.faculty if any(s.faculty == x.id for s in instance.sessions))
    mine = [s.id for s in instance.sessions if s.faculty == f.id]
    tt = random_timetable(instance, random.Random(0), mine)
    teacher = Request(id="R-1", channel=Channel.PORTAL, sender_id=f.id, role=Role.FACULTY, raw_text="x",
                      received_at=datetime(2026, 9, 1))
    text = intent_directory(instance, teacher, tt, 7)
    assert "Current week: 7." in text and "Courses:" not in text
    p = tt[mine[0]]
    assert f"{mine[0]}: " in text and f" {p.day} {9 + p.slot:02d}:00" in text
    office = teacher.model_copy(update={"sender_id": "C-TT", "role": Role.COORDINATOR})
    assert "Courses:" in intent_directory(instance, office)  # someone else's classes, by course
    assert len({(p.day, p.slot) for p in tt.values()}) == len(tt)  # one class at a time
    assert all(p.slot != instance.calendar.lunch_slot for p in tt.values())


def test_new_requests_cover_every_generator_and_resolve(instance):
    examples = intent_examples(instance, per_generator=12, seed=3)
    assert {e.generator for e in examples} == set(GENERATORS)
    assert all(_resolves(e, instance) for e in examples)
    assert all((e.split == "test") == (e.template == TEST_TEMPLATE) for e in examples)
    withdraw = [e for e in examples if e.generator == "withdraw"]
    assert withdraw and all(e.gold.action == "withdraw" for e in withdraw)


def test_train_only_wordings_stay_out_of_test_and_say_what_the_test_wording_says(instance):
    examples = train_only_examples(instance, per_generator=10)
    assert {e.generator for e in examples} == {"move", "club_booking"}
    assert all(_resolves(e, instance) and e.split != "test" for e in examples)
    intents = [e.gold.intents[0] for e in examples]
    assert all(not i.hard for i in intents if i.kind == "move")  # a wish
    assert all(i.room is None for i in intents if i.kind == "club_booking")  # a kind of room, not a room


def test_the_scorer_gives_gold_full_marks_and_catches_a_wrong_class(instance):
    from evaluation.intents import Reader, score

    rows = [json.loads(line) for line in (DATA / "compiler-intents" / "test.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    rows = [r for r in rows if r["kind"] in ("move", "cancel", "withdraw", "unavailability")][:30]
    assert score(rows, instance, Reader("gold", instance))["summary"]["outcome_match"] == 1.0

    class OffByAnHour(Reader):  # moves every class reference and time by an hour
        def read(self, row):
            text = row["messages"][2]["content"]
            for h in range(16, 8, -1):
                text = text.replace(f'"{h:02d}:00"', f'"{h + 1:02d}:00"')
            return IntentOutput.model_validate_json(text)

    got = score(rows, instance, OffByAnHour("gold", instance))
    timed = [r for r in got["rows"] if "00\"" in json.dumps(r["gold"])]
    assert timed and not any(r["match"] for r in timed)  # a different hour is a different answer
    assert got["by_kind"]["withdraw"]["outcome_match"] == 1.0  # nothing timed to get wrong
