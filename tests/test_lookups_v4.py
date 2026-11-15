"""nts-lookup-v4's vocabulary (agents.lookups_v4): each tool only its fields, clock times and weeks worked out by
the code, request and rule parts with their words; v3's training rows carried over with the same lookups."""

import json
from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agents.lookups_v4 import LookupsV4
from agents.tool_agent import Other, ToolAgent
from web.app import World, create_web_app

WEDNESDAY = datetime(2026, 10, 21, 10, 30)  # week 13 (evaluation.lookup_set's "now")
FRIDAY = datetime(2026, 10, 23, 10, 30)


class FakeV4:
    """Stands in for nts-lookup-v4: answers with ``parts`` (dicts); counts its calls."""

    def __init__(self, *parts):
        self.parts, self.calls = list(parts), 0

    def generate(self, system, prompt, schema):
        self.calls += 1
        return LookupsV4.model_validate({"parts": self.parts})


@pytest.fixture(scope="module")
def world():
    return World(parser_mode="offline", seed_history=False, deadline=60)


def menon(world):
    return next(f.id for f in world.instance.faculty if f.name == "Dr. Menon")


def read(world, *parts, now=WEDNESDAY, text="a message"):
    from evaluation.lookup_set import SEMESTER_START

    agent = ToolAgent(world.instance, FakeV4(*parts), menon(world), 13, today=now.date(), now=now,
                      semester_start=SEMESTER_START, dsl=True)
    return agent.read(text), agent


# -- the schema ---------------------------------------------------------------------------------------------


def test_a_tool_takes_only_its_own_fields():
    with pytest.raises(ValidationError):
        LookupsV4.model_validate({"parts": [{"tool": "free_rooms", "change": "cancelled"}]})
    with pytest.raises(ValidationError):  # who is not free_rooms'
        LookupsV4.model_validate({"parts": [{"tool": "free_rooms", "who": {"teachers": ["me"]}}]})


def test_the_tool_is_always_written_and_at_most_eight_parts():
    one = LookupsV4.model_validate({"parts": [{"tool": "my_rules"}]})
    assert json.loads(one.model_dump_json(exclude_defaults=True)) == {"parts": [{"tool": "my_rules"}]}
    with pytest.raises(ValidationError):
        LookupsV4.model_validate({"parts": [{"tool": "my_rules"}] * 9})


# -- times and weeks: the code's, not the model's ---------------------------------------------------------------


def test_clock_times_become_slots(world):
    call, _ = read(world, {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Thu"], "at": ["14:00"]}})
    assert call.args["slots"] == [5] and call.args["days"] == ["Thu"]


def test_a_range_of_times_is_the_teaching_hours_in_it(world):
    lunch = world.instance.calendar.lunch_slot
    call, _ = read(world, {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"after": "11:00"}})
    assert call.args["slots"] == [s for s in range(2, world.instance.calendar.slots_per_day) if s != lunch]


def test_next_week_and_several_weeks(world):
    call, _ = read(world, {"tool": "changes", "when": {"period": {"relative": "next_week"}}})
    assert call.week == 14
    call, _ = read(world, {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Tue"],
                                                                                     "period": {"weeks": [14, 15]}}})
    assert call.week == 14 and call.args["weeks"] == [14, 15]


def test_tomorrow_on_a_friday_is_monday_of_next_week(world):
    call, _ = read(world, {"tool": "classes_of", "who": {"teachers": ["me"]}, "when": {"day": "tomorrow"}}, now=FRIDAY)
    assert (call.week, call.args["days"]) == (14, ["Mon"])


def test_a_time_or_week_that_does_not_exist_is_said(world):
    call, _ = read(world, {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"at": ["19:00"]}})
    assert not call.confident and "No class starts at 7 pm" in call.note
    call, _ = read(world, {"tool": "changes", "when": {"period": {"weeks": [20]}}})
    assert not call.confident and "week 20 isn't in it" in call.note


# -- request and rule parts ----------------------------------------------------------------------------------


def test_request_and_rule_parts_keep_their_kind_and_words(world):
    call, agent = read(world, {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Fri"]}},
                       {"tool": "rule", "text": "how many online classes are allowed?"},
                       {"tool": "request", "text": "move my DS lecture to Thursday"}, {"tool": "other"})
    assert call.tool == "free_hours"
    assert agent.others == [Other("rule", "how many online classes are allowed?"),
                            Other("request", "move my DS lecture to Thursday"), Other("other")]


def test_in_the_chat_each_part_is_read_in_its_own_words(monkeypatch):
    from web.fallback import ModelLink

    monkeypatch.delenv("NTS_LOOKUP_READER", raising=False)
    w = World(parser_mode="offline", seed_history=False, deadline=60)
    w.lookup_adapter = "nts-lookup-v4"
    w.model_link = ModelLink(lambda: FakeV4(
        {"tool": "free_hours", "who": {"teachers": ["me"]}, "when": {"days": ["Fri"]}},
        {"tool": "rule", "text": "how many online classes are allowed in a week?"},
        {"tool": "request", "text": "please move my Data Structures lecture to Thursday"}))
    c = TestClient(create_web_app(w))
    r = c.post("/api/chat", json={"text": "When am I free on Friday, how many online classes are allowed in a week, "
                                          "and please move my Data Structures lecture to Thursday."},
               headers={"X-User": menon(w)})
    _, answer, rule, card = r.json()
    assert answer["kind"] == "lookup" and rule["kind"] == "rule" and card["kind"] == "request"  # sent at once
    assert "P-ONLINE" in [x["id"] for x in rule["rules"]]  # the rule question alone, not the whole message
    assert card["request_text"] == "please move my Data Structures lecture to Thursday"


# -- v3's training rows in v4 -----------------------------------------------------------------------------------


def test_v3_rows_carry_over_with_the_same_lookups(world):
    from training.lookup_data import ROOT, TEMPLATES, departments
    from training.lookup_data_v3 import TEMPLATES as V3_TEMPLATES
    from training.lookup_data_v4 import Left, migrate_row

    groups = {r["id"]: r.get("group", "") for p in (TEMPLATES, V3_TEMPLATES)
              for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    rows = [json.loads(x) for x in (ROOT / "data" / "lookup-train-v3" / "val.jsonl").read_text(encoding="utf-8")
            .splitlines()[:120]]
    depts = dict(departments(world.instance, 0))
    kept = left = 0
    for row in rows:
        try:
            migrate_row(row, depts[row["department"]], groups)  # checks the lookups are the same
            kept += 1
        except Left as e:
            assert "its words" in str(e), f"{row['id']}: {e}"  # only mixed messages, whose request words v3 lacks
            left += 1
    assert kept > 100
