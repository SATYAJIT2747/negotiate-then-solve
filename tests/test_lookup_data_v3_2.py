"""v3.2 and v4's training data (training.lookup_data_v3_2): requests and rule questions keep their own words, and
every row is the same in both formats."""

import json

import pytest

from agents.lookups_v4 import LookupsV4
from agents.tool_agent import Lookups
from training.lookup_data_v3_2 import TEMPLATES, V2_TEMPLATES, V3_TEMPLATES, both, build
from web.app import World


@pytest.fixture(scope="module")
def made():
    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    train, val, dropped, _ = build(instance, seed=0, fills=1, mixed=25, long=10)
    groups = {r["id"]: r.get("group", "") for p in (V2_TEMPLATES, V3_TEMPLATES, TEMPLATES)
              for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    rows = train + val
    words = {r["id"]: [tuple(o) for o in r["others"]] for r in rows}
    v32, v4 = both(rows, instance, groups, why := __import__("collections").Counter())
    return v32, v4, words, why


def answer(row):
    return json.loads(row["messages"][2]["content"])


def test_a_request_keeps_its_own_words(made):
    v32, _, words, _ = made
    mixed = [r for r in v32 if r["kind"] == "mixed"]
    assert mixed
    for row in mixed:
        question = row["messages"][1]["content"].split("<question>")[1].split("</question>")[0]
        said = [x["request"] for x in answer(row)["lookups"] if x["tool"] == "none" and "request" in x]
        assert said and all(s in question and s != question for s in said)  # a part of the message, not all of it
        assert said == [w for k, w in words[row["id"]] if k != "other"]


def test_both_formats_have_the_same_rows(made):
    v32, v4, _, why = made
    assert [r["id"] for r in v32] == [r["id"] for r in v4] and not why
    for a, b in zip(v32, v4):
        Lookups.model_validate(answer(a))
        parts = LookupsV4.model_validate(answer(b)).parts
        assert len(parts) == len(answer(a)["lookups"])
        assert [p.tool for p in parts if p.tool in ("request", "rule", "other")] == \
               [k for k, _ in made[2][a["id"]]]  # each "none" in v3.2 is the part it is in v4
