"""How the training rows teach each kind of intent, and rows that keep it balanced.

A kind of request is learned from the messages where it is the whole request ("please cancel my Monday 10 am
lecture") and from those where it is one part of several ("cancel my Monday lecture, and no classes before
lunch on Friday"). The parts come from generators made for the combining (``multi``, ``compound``,
``exception``) and carry their own habits: a cancellation in ``compound`` names its class by the day alone, a
limit in ``multi`` is always the sender's. When the parts outnumber the requests on their own, those habits
become the kind's reading (intents-v7 dropped the time of "(Monday, 10 am)" from cancellations and gave
limits a ``who`` they never had).

``report`` counts, per kind, the rows where it stands alone and where it is a part, and how much of it each
generator makes. ``top_up`` adds rows of a kind on its own from the kind's first generator
(``intent_corpus.GENERATORS``) until the parts are at most ``max_parts`` of them: its training wordings only
(never ``TEST_TEMPLATE``) and a seed of its own, so the test split and every other row stay as they were.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

from core.instance import Instance

# the first generator of each kind of intent (language.intent_corpus.GENERATORS), whose rows are the kind alone
OWN = {"cancel": "cancel", "load_limit": "load_limit", "move": "move", "swap": "swap", "extra_class": "extra_class",
       "avoid": "avoid_between", "unavailable": "unavailable_relative", "club_booking": "club_booking",
       "clash_report": "clash_report"}


def _intents(row: dict) -> list[dict]:
    return json.loads(row["messages"][-1]["content"]).get("intents") or []


def report(rows: list[dict]) -> dict:
    """Per kind: rows where it is the whole request, rows where it is one part of several, and the
    generators (row ``kind``) that make the most of it."""
    alone, part, by = Counter(), Counter(), defaultdict(Counter)
    for r in rows:
        its = _intents(r)
        for i in its:
            (alone if len(its) == 1 else part)[i["kind"]] += 1
            by[i["kind"]][r["kind"]] += 1
    out = {}
    for k in sorted(by, key=lambda k: -sum(by[k].values())):
        n = sum(by[k].values())
        out[k] = {"alone": alone[k], "part": part[k], "parts_per_alone": round(part[k] / max(alone[k], 1), 2),
                  "top": {g: round(v / n, 2) for g, v in by[k].most_common(3)}}
    return out


def short(rows: list[dict], max_parts: float) -> dict[str, int]:
    """How many more rows of each kind on its own keep its parts at most ``max_parts`` of them."""
    alone, part = Counter(), Counter()
    for r in rows:
        its = _intents(r)
        for i in its:
            (alone if len(its) == 1 else part)[i["kind"]] += 1
    return {k: need for k in sorted(part) if k in OWN and (need := math.ceil(part[k] / max_parts) - alone[k]) > 0}


def top_up(instance: Instance, rows: list[dict], spec: dict) -> tuple[list[tuple[str, dict]], dict[str, int]]:
    """``(split, row)`` of the kinds ``short`` of rows on their own, from their own generators' training
    wordings, about a tenth to val as everywhere else; and how many training rows each kind was short."""
    from language.intent_compiler import export_rows
    from language.intent_corpus import GENERATORS, TEST_TEMPLATE, _examples

    need = short(rows, spec["max_parts"])
    templates = tuple(t for t in range(5) if t != TEST_TEMPLATE)
    out: list[tuple[str, dict]] = []
    for kind, n in sorted(need.items()):
        g = OWN[kind]
        # a tenth of what is made goes to val: make more, in rounds of their own seed, until train has enough
        for round_ in range(5):
            have = sum(s == "train" for s, r in out if _intents(r)[0]["kind"] == kind)
            if have >= n:
                break
            made = _examples(instance, random.Random(f"{spec['seed']}:{g}:{round_}"),
                             {f"{g}-alone{round_ or ''}": (g, GENERATORS[g])}, math.ceil((n - have) / 0.9), None,
                             templates=templates)
            out += [(s, r) for s, r in export_rows(instance, [], made) if r is not None]
    return out, need
