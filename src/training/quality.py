"""Gates every training row passes before a fine-tuning run may use it, and a report of what is in the data.

A row is one message with its gold reading (``language.intent_compiler`` row format: the chat messages and
the request, timetable and week the gold resolves against). The gates hold for every source alike, so a
row mined from a model's mistakes or labelled by a person meets the same bar as a generated one:

* ``schema``: the target is an ``IntentOutput`` of the current version, written canonically
  (``intent_compiler.completion``), so one meaning is one string to learn;
* ``resolves``: the gold resolves against the row's own timetable and week (a question only where the
  row's action asks one);
* ``rubric``: the checks a reading goes through at run time (``language.checks``) change nothing in the
  gold, and a lone move is as firm as the labelling rule says (``lexicon.placement_is_hard``): data and
  checks follow one rule;
* ``leak``: no training or validation message is a test or suite message reworded only in greeting,
  sign-off, case or spacing.

``check`` returns the failures by gate and row; ``report`` counts rows by split, source, kind, action and
firmness, which is what a run's manifest records.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from core.instance import Instance
from core.schemas import Placement, Request
from language.checks import check_reading
from language.intent_compiler import completion
from language.intents import SCHEMA_VERSION, IntentOutput, Move, compile_intents
from language.lexicon import placement_is_hard

GATES = ("schema", "resolves", "rubric", "leak")
_MESSAGE = re.compile(r"<message>\n(.*)\n</message>", re.DOTALL)


def message(row: dict) -> str:
    m = _MESSAGE.search(row["messages"][1]["content"])
    return m.group(1) if m else ""


_GREETING = re.compile(r"^(dear (timetable (office|team)|sir( or madam)?|madam|team|all|coordinator)|hello|hi|hey|"
                       r"good (morning|afternoon|evening))\b\s*")
_SIGN_OFF = re.compile(r"\s*\b(thanks|thank you|many thanks|regards|best regards|best|cheers)\b"
                       r"(\s+(dr|prof)?\s*[a-z]+){0,3}\s*$")


def core(text: str) -> str:
    """A message without its greeting, its closing line, case, punctuation or spacing: what tells two
    messages apart. Only a closing at the very end goes (a "P.S." after it stays: it may be the point)."""
    t = re.sub(r"[^\w]+", " ", text.lower()).strip()
    return _SIGN_OFF.sub("", _GREETING.sub("", t)).strip()


def _context(row: dict):
    request = Request.model_validate(row["request"])
    timetable = {sid: Placement.model_validate(p) for sid, p in row["timetable"].items()}
    return request, timetable, row.get("current_week")


def check_row(instance: Instance, row: dict) -> list[tuple[str, str]]:
    """``(gate, reason)`` for each gate the row fails (the leak gate needs every row: ``check``)."""
    fails: list[tuple[str, str]] = []
    if row.get("schema") != SCHEMA_VERSION:
        fails.append(("schema", f"schema {row.get('schema')}, not {SCHEMA_VERSION}"))
    target = row["messages"][-1]["content"]
    try:
        out = IntentOutput.model_validate_json(target)
    except ValueError as e:
        return [*fails, ("schema", f"not an IntentOutput: {str(e)[:80]}")]
    if completion(out) != target:
        fails.append(("schema", "target not canonical"))
    request, timetable, week = _context(row)
    try:
        got = compile_intents(instance, request, out, timetable=timetable, current_week=week)
    except Exception as e:  # noqa: BLE001 - any failure to resolve is a failed gate
        return [*fails, ("resolves", f"{type(e).__name__}: {str(e)[:80]}")]
    if got.question is not None and out.action == "compile" and not row.get("asks"):
        fails.append(("resolves", f"asks: {got.question[:80]}"))
    # a reworded twin keeps its parent's gold, so its firmness is judged on the words it was labelled from
    said = request.model_copy(update={"raw_text": row.get("parent_text") or request.raw_text})
    checked = check_reading(instance, said, out, timetable)
    if checked.notes or checked.question:
        fails.append(("rubric", (checked.notes or [checked.question])[0][:100]))
    moves = [i for i in out.intents if isinstance(i, Move)]
    if len(out.intents) == 1 and moves and moves[0].hard != placement_is_hard(said.raw_text):
        fails.append(("rubric", f"move hard={moves[0].hard}, the rule says {not moves[0].hard}"))
    return fails


def check(instance: Instance, splits: dict[str, list[dict]], held: Iterable[str] = ("test",)) -> dict:
    """Every row of ``splits`` through every gate; ``held`` names the splits no other split may repeat."""
    failures: dict[str, list[dict]] = defaultdict(list)
    held = set(held)
    held_text = {core(message(r)): (name, r["id"]) for name in held for r in splits.get(name, [])}
    for name, rows in splits.items():
        for row in rows:
            fails = check_row(instance, row)
            if name not in held and (hit := held_text.get(core(message(row)))):
                fails.append(("leak", f"repeats {hit[0]} row {hit[1]}"))
            for gate, why in fails:
                failures[gate].append({"split": name, "id": row["id"], "origin": row.get("origin"), "why": why})
    return {gate: failures.get(gate, []) for gate in GATES}


def firmness_of(out: IntentOutput) -> str:
    hard = [getattr(i, "hard", None) for i in out.intents if hasattr(i, "hard")]
    return "none" if not hard else "hard" if all(hard) else "soft" if not any(hard) else "mixed"


def report(splits: dict[str, list[dict]]) -> dict:
    """Counts for the manifest: rows by split and origin, intent kind, action and firmness."""
    out: dict = {}
    for name, rows in splits.items():
        c: dict[str, Counter] = {k: Counter() for k in ("origin", "kind", "action", "firmness", "augment")}
        for r in rows:
            o = IntentOutput.model_validate_json(r["messages"][-1]["content"])
            c["origin"][r.get("origin", r.get("source", "?"))] += 1
            c["kind"][r.get("kind", "?")] += 1
            c["action"][o.action] += 1
            c["firmness"][firmness_of(o)] += 1
            c["augment"][r.get("augment", "none")] += 1
        out[name] = {"rows": len(rows), **{k: dict(sorted(v.items())) for k, v in c.items()}}
    return out


def summary(failures: dict) -> str:
    return ", ".join(f"{g} {len(failures[g])}" for g in GATES)


def dump_failures(failures: dict, n: int = 5) -> str:
    lines = []
    for gate in GATES:
        for f in failures[gate][:n]:
            lines.append(f"  {gate:8} {f['split']:9} {f['id']:28} {f['why']}")
        if len(failures[gate]) > n:
            lines.append(f"  {gate:8} ... {len(failures[gate]) - n} more")
    return "\n".join(lines)


def to_json(failures: dict) -> str:
    return json.dumps(failures, indent=1)
