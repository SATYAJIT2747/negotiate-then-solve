import random

from training.encoder_data import _end, _low, fill, quotas, tier


def test_tier():
    assert tier({"action": "clarify"}) == "non_compile"
    assert tier({"action": "compile", "intents": [{"kind": "move"}]}) == "single"
    assert tier({"action": "compile", "intents": [{"kind": "move"}, {"kind": "avoid"}]}) == "multi"
    assert tier({"action": "compile", "intents": [{"kind": "move"}, {"kind": "move"}, {"kind": "any_of"}]}) == "nested"
    assert tier({"action": "compile", "intents": [{"kind": "relate"}]}) == "linked"
    assert tier({"action": "compile", "intents": [{"kind": "move"}, {"kind": "cancel", "which": {"same": "a"}}]}) == "linked"


def test_quotas_water_fill():
    q = quotas({"big": 1000, "mid": 100, "rare": 10}, 300, 2.0)
    assert q["rare"] == 20  # a rare group gives at most max_up times its size
    assert sum(q.values()) == 300 and q["big"] == q["mid"] == 140


def test_fill_repeats_marked():
    out = fill({"a": [{"x": 1}], "b": [{"x": i} for i in range(10)]}, 6, 2.0, random.Random(0))
    assert len(out) == 6 and sum(1 for r in out if r.get("repeat")) == 1


def test_join_casing():
    assert _low("Please move it.") == "please move it."
    assert _low("I can't teach.") == "I can't teach."
    assert _low("Monday is out.") == "Monday is out."
    assert _end("could you move it?") == "could you move it?" and _end("no classes") == "no classes."
