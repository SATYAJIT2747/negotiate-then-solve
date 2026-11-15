"""The long-message test set (evaluation.multi_set): built the same every time, from test templates only, and
scored as the chat reads a message."""

import pytest

from agents.questions import ToolCall
from evaluation.lookup_set import with_first_names
from evaluation.multi_set import MAX_PARTS, build, score
from web.app import World


@pytest.fixture(scope="module")
def rows():
    instance = with_first_names(World(parser_mode="offline", seed_history=False, deadline=60).instance)
    return build(instance, n=30), build(instance, n=30)


def test_the_set_is_the_same_on_every_run(rows):
    first, again = rows
    assert first == again


def test_every_message_has_two_to_five_parts_and_a_question(rows):
    for row in rows[0]:
        kinds = [p["kind"] for p in row["parts"]]
        assert 2 <= len(kinds) <= MAX_PARTS and "question" in kinds
        assert kinds.count("request") == row["requests"] and kinds.count("rule") == row["rules"]


def perfect(row):
    """The reading a model that gets every part right gives."""
    calls = [ToolCall(p["tool"], dict(p["args"]), p["week"]) for p in row["parts"] if p["kind"] == "question"]
    others = [""] * (row["requests"] + row["rules"])
    return (calls[0] if len(calls) == 1 else ToolCall("multi", {"parts": calls}, calls[0].week)), others


def test_a_perfect_reading_is_right_and_a_lost_request_is_not(rows):
    for row in rows[0]:
        call, others = perfect(row)
        assert score(row, call, others)["message"], row["message"]
        if row["requests"]:
            assert not score(row, call, others[1:])["others"]  # one part fewer left for others
