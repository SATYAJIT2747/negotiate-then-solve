import json
from pathlib import Path

import pytest

from core.instance import Instance
from training.encoder_spans import Aligner, dates, days, kinds, numbers, times


@pytest.fixture(scope="module")
def aligner():
    return Aligner(Instance.model_validate_json(Path("data/synthetic-cse-s0.json").read_text(encoding="utf-8")))


def _row(text, intents, action="compile"):
    return {"id": "x", "kind": "t", "request": {"raw_text": text, "role": "faculty"}, "current_week": 5,
            "messages": [{"role": "assistant", "content": json.dumps({"action": action, "intents": intents})}]}


def test_times():
    got = {v for _, _, v in times("before 2 pm, 14:00-16:00, 2-4pm, at 3, between 10 and 4, noon, 11 a.m.")}
    assert got >= {"14:00", "16:00", "15:00", "10:00", "12:00", "11:00"}


def test_days_nearest_not_first():
    assert [d for *_, d in days("Thursday, thurs, wenesday, Mondays")] == ["Thu", "Thu", "Wed", "Mon"]


def test_numbers_skip_list_markers_and_articles():
    text = "1. No more than 3 classes.\n2. A committee meeting, an hour gap."
    assert sorted(v for *_, v in numbers(text, [])) == [1, 3]


def test_kinds_and_dates():
    assert [k for *_, k in kinds("lab, lec, practiical")] == ["practical", "lecture", "practical"]
    assert [d for *_, d in dates("on 23 November or Nov 3rd")] == ["2026-11-23", "2026-11-03"]


def test_each_value_where_its_intent_is(aligner):
    text = ("My class currently on Tuesday at 3 pm: could we do Thursday morning? Separately, Dr. Kumar and I would "
            "like to swap slots: my class on Tuesday at 3 pm for their Thursday 9 am lecture.")
    out = aligner.row(_row(text, [
        {"kind": "move", "which": {"day": "Tue", "time": "15:00"}, "to": {"days": ["Thu"], "part": "morning"}},
        {"kind": "swap", "mine": {"day": "Tue", "time": "15:00"},
         "theirs": {"owner": "Dr. Kumar", "day": "Thu", "time": "09:00", "kind": "lecture"}}]))
    move, swap = out["intents"]
    assert max(s["end"] for s in move["spans"]) < text.index("Separately")
    assert min(s["start"] for s in swap["spans"] if s["role"] != "theirs.owner") > text.index("Separately")
    assert move["classes"] == {"to.part": "morning"} and out["aligned"]


def test_range_covers_inner_day(aligner):
    text = "Week 16, Wednesday through Friday: I can't teach. Also no Thursday classes for me in odd weeks."
    out = aligner.row(_row(text, [
        {"kind": "unavailable", "time": {"days": ["Wed", "Thu", "Fri"]}, "period": {"weeks": [16]}},
        {"kind": "avoid", "time": {"days": ["Thu"]}, "period": {"parity": "odd"}}]))
    first = out["intents"][0]
    assert {s["text"] for s in first["spans"]} == {"16", "Wednesday", "Friday"} and out["aligned"]


def test_unsaid_kind_is_inferred(aligner):
    text = "Can Dr. Rao take my Friday 10 am slot and I take theirs on Thursday at 11 am? Cancel my Monday lecture."
    out = aligner.row(_row(text, [
        {"kind": "swap", "mine": {"day": "Fri", "time": "10:00", "kind": "lecture"},
         "theirs": {"owner": "Dr. Rao", "day": "Thu", "time": "11:00"}},
        {"kind": "cancel", "which": {"day": "Mon", "kind": "lecture"}}]))
    assert out["intents"][0]["inferred"] == ["mine.kind=lecture"] and out["aligned"]
