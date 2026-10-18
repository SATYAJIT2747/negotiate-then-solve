"""Readings stored in an older version of the intent vocabulary, brought to the current one.

Training rows made by the generators are rebuilt from code, so they are always current. What cannot be
rebuilt is kept: rows mined from a model's mistakes, messages people labelled, readings saved by an
evaluation. Each carries the ``schema`` it was written in (``intents.SCHEMA_VERSION``); a row without one
was written before versions existed, in version 1. ``migrate`` takes a reading (the JSON an
``IntentOutput`` dumps to, as in a training target) through every step from its version to the current one,
so that a later fine-tuning run can use every earlier row with the meaning it was labelled with.

A step only rewrites what changed meaning. Version 2 made a move without ``hard`` a wish, so a version-1
move without it, a requirement when it was written, gets ``hard: true``. Version 3 only added fields.
Version 4 reads names where there were IDs, and an ID still reads as itself (``language.names``), so a
reading needs no change; the training rows are written with names when they are put in the current
prompt (``intent_compiler.format_row``).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .intents import SCHEMA_VERSION


def _v1_to_v2(out: dict) -> dict:
    for intent in out.get("intents") or []:
        if intent.get("kind") == "move" and "hard" not in intent:
            intent["hard"] = True
    return out


def _added_fields(out: dict) -> dict:
    return out


STEPS: dict[int, Callable[[dict], dict]] = {1: _v1_to_v2, 2: _added_fields, 3: _added_fields}
assert sorted(STEPS) == list(range(1, SCHEMA_VERSION)), "a migration for every version before the current one"


def migrate(out: dict, version: int | None) -> dict:
    """``out`` (a reading as JSON, written in ``version``; None: 1) in the current version."""
    version = version or 1
    if version > SCHEMA_VERSION:
        raise ValueError(f"a reading in schema {version} is newer than this code ({SCHEMA_VERSION})")
    out = json.loads(json.dumps(out))
    for v in range(version, SCHEMA_VERSION):
        out = STEPS[v](out)
    return out


def migrate_row(row: dict) -> dict:
    """A training row (its assistant message a reading) in the current version, marked with it."""
    version = row.get("schema")
    if version == SCHEMA_VERSION:
        return row
    from .intent_compiler import completion  # the canonical target: defaults left out, fields in order
    from .intents import IntentOutput

    messages = [dict(m) for m in row["messages"]]
    out = IntentOutput.model_validate(migrate(json.loads(messages[-1]["content"]), version))
    messages[-1]["content"] = completion(out)
    return {**row, "messages": messages, "schema": SCHEMA_VERSION, "migrated_from": version or 1}
