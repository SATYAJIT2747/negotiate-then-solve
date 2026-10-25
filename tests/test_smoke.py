"""The smoke set (evaluation.smoke) offline: the everyday requests the rule-based stand-ins already handle
must keep doing what they should. The rest are for the model (``python -m evaluation.smoke --parser local``)."""

import json

import pytest

from evaluation.smoke import SMOKE, run

OFFLINE = {"S-01", "S-02", "S-03", "S-04", "S-06", "S-08", "S-12"}


@pytest.mark.parametrize("row", [r for r in map(json.loads, SMOKE.read_text(encoding="utf-8").splitlines())
                                 if r["id"] in OFFLINE], ids=lambda r: r["id"])
def test_an_everyday_request_does_what_it_should(row):
    (result,) = run("offline", [row], timeout=120)
    assert result["ok"], result["wrong"]
