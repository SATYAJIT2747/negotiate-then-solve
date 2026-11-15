"""memory v1.1's new kind (training.memory_data, ``plural``): a plural word is filled in with all of it."""

import re

import pytest

from training.memory_data import build
from web.app import World


@pytest.fixture(scope="module")
def rows():
    w = World(parser_mode="offline", seed_history=False, deadline=60)
    train, _, _ = build(w.instance, w.store, 40, test=False, shares={"plural": 1}, tag="t-")
    test, _, _ = build(w.instance, w.store, 12, test=True, shares={"plural": 1}, tag="t-")
    return train, test


def test_these_days_and_those_times_are_all_of_it(rows):
    for row in rows[0] + rows[1]:
        m = row["meta"]
        asked, reply, full = m["history"][-2][1], m["history"][-1][1], m["answer"]["message"]
        if re.search(r"\b(these|those|same) days\b", m["message"]):  # every day of the earlier question
            days = re.search(r" on (.+)\?$", asked).group(1)
            assert days in full
        if "those times" in m["message"] or "those slots" in m["message"] or "those hours" in m["message"]:
            spans = re.findall(r"\d+(?: am| pm)?–\d+ (?:am|pm)", reply)
            assert len(spans) > 1 and all(s in full for s in spans)  # every span, not one asked back


def test_new_conversations_are_not_v1s(rows):
    assert all(r["id"].startswith("t-plural-") for r in rows[0] + rows[1])
