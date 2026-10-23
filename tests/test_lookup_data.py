"""Training data for the lookup adapter (training.lookup_data): every row checked, no test wording in it."""

import json

import pytest

from agents.tool_agent import SYSTEM, Choice
from training.lookup_data import build
from web.app import World


@pytest.fixture(scope="module")
def data():
    return build(World(parser_mode="offline", seed_history=False, deadline=60).instance, fills=2, seed=0)


def test_every_row_passes_the_agents_checks(data):
    train, val, dropped, _ = data
    assert train and val and dropped == []  # a row whose answer fails ToolAgent's checks is a template mistake


def test_rows_are_chats_as_the_server_is_sent_them(data):
    train, _, _, _ = data
    for row in train[:50]:
        system, user, answer = row["messages"]
        assert system == {"role": "system", "content": SYSTEM} and user["role"] == "user"
        assert "<question>" in user["content"] and answer["role"] == "assistant"
        Choice.model_validate_json(answer["content"])  # the answer is the shape ToolAgent reads


def test_validation_templates_are_not_in_training(data):
    train, val, _, _ = data
    assert not {r["template"] for r in train} & {r["template"] for r in val}


def test_every_tool_is_taught(data):
    _, _, _, cover = data
    for tool in ("free_hours", "classes_of", "free_rooms", "who_is_free", "describe", "changes", "unanswerable", "none"):
        assert cover[f"tool {tool}"] > 0, tool
    assert cover["field ambiguous"] and cover["field unknown"] and cover["field dates"]
