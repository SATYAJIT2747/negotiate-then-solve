"""What an adapter was trained on must not drift: the agents' versions (prompt, tools, budgets) and the
notes the harness adds to turns. A change here means new training data, so it should be on purpose:
make a new agent version instead of editing one an adapter was trained on."""

import hashlib

from nts.agents import COORDINATOR, COORDINATOR_V1, COORDINATOR_V2, COORDINATOR_V3
from nts.harness.context import LAST_TURN, UNCHECKED, WRAP_UP
from nts.tools import ALL_TOOLS, Registry

TRAINED = {  # agent -> version: what each adapter's episodes were collected with
    "v1": (COORDINATOR_V1, "8266a3f614bd"),  # nts-agent-v1 (85c5b198b9b3 with a local model's budgets)
    "v2": (COORDINATOR_V2, "a008f93774e0"),  # nts-agent-v2
    "v3": (COORDINATOR_V3, "696ca57090f2"),  # nts-agent-v3
}


def test_trained_agents_are_frozen():
    r = Registry(ALL_TOOLS)
    got = {k: a.version(r.specs(a.scopes)) for k, (a, _) in TRAINED.items()}
    assert got == {k: v for k, (_, v) in TRAINED.items()}
    assert COORDINATOR is COORDINATOR_V3


def test_the_turn_notes_are_frozen():
    assert hashlib.sha256((LAST_TURN + UNCHECKED + WRAP_UP).encode()).hexdigest()[:12] == "c0abf24d6d4e"
