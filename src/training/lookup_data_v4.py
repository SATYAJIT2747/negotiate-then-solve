"""Training data for nts-lookup-v4 (``agents.lookups_v4``): v3's rows (``data/lookup-train-v3``: generated and
Gemini's rewordings) with the answer written in v4's vocabulary, and v4's prompt.

    uv run python -m training.lookup_data_v4          # writes data/lookup-train-v4/{train,val,distilled}.jsonl

The question stays as it is: only the answer changes, so Gemini's rewordings carry over with no new calls. A row
is kept only if its v4 answer, read by ``ToolAgent`` (``dsl=True``), gives exactly the lookups its v3 answer gives
(``multi=True``): the same tools, weeks, arguments and notes, and as many parts left for others. Each row says the
schema it is in (``agents.lookups_v4.LOOKUP_SCHEMA``) and the one it came from.

A v3 "none" says neither which kind it is nor its words; v4's request and rule parts say both. The kind comes from
the part's template (its group: request, rule or other). The words are known when the whole message is that part;
in a message of several parts they are not, and the row is left out (``training.lookup_data_v3`` writes those
messages again with their words, for v3.1 and v4 alike).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from agents.lookups_v4 import LOOKUP_SCHEMA, SYSTEM_V4, WEEKLESS, LookupsV4, from_choice
from agents.questions import ToolCall
from agents.tool_agent import Lookups, Other, ToolAgent
from evaluation.lookup_set import SEMESTER_START

from .lookup_data import ROOT, TEMPLATES as V2_TEMPLATES, departments
from .lookup_data_v3 import TEMPLATES as V3_TEMPLATES, FixedLookups

V3 = ROOT / "data" / "lookup-train-v3"
OUT = ROOT / "data" / "lookup-train-v4"
QUESTION = re.compile(r"<question>(.*)</question>", re.S)


class Left(Exception):
    """Why a row is not carried over."""


def _weekless(call: ToolCall | None) -> ToolCall | None:
    """``call`` with the week left out of the lookups that don't depend on it (describe, my_requests, ...), and
    any question back the same as another ("Which one do you mean: A, B?" / "... A or B?")."""
    if call is None:
        return None
    if not call.confident:  # the same question back, listed either way ("A, B?" / "A or B?")
        return ToolCall("unclear", confident=False, week=0, note=re.sub(r",? or |, ", " ", call.note))
    if call.tool == "multi":
        return replace(call, args={"parts": [_weekless(p) for p in call.args["parts"]]}, week=0)
    return replace(call, week=0) if call.tool in WEEKLESS else call


def migrate_row(row: dict, dept, groups: dict[str, str], others: list[Other] | None = None) -> dict:
    """``row`` (v3) in v4, checked; ``Left`` says why it can't be. ``others``: what each "none" part is, with its
    words, in order, when the row's maker knows (``training.lookup_data_v3_2``); else from the templates."""
    m = row["meta"]
    question = QUESTION.search(row["messages"][1]["content"]).group(1)
    v3 = Lookups.model_validate_json(row["messages"][-1]["content"])
    now = datetime.fromisoformat(m["now"])

    def agent(answer, dsl: bool) -> ToolAgent:
        return ToolAgent(dept, FixedLookups(answer), m["asker"], m["week"], today=now.date(), now=now,
                         semester_start=SEMESTER_START, multi=True, dsl=dsl)

    v4_agent = agent(None, True)
    parts, given = [], iter(others or [])
    for k, c in enumerate(v3.lookups):
        other = None
        if c.tool == "none" and others is not None:
            other = next(given)
        elif c.tool == "none":
            template = row["templates"][k] if len(row["templates"]) == len(v3.lookups) else None
            kind = {"request": "request", "rule": "rule"}.get(groups.get(template, ""), "other") if template else None
            if kind is None:
                raise Left("none: which part it is is not known")
            if kind != "other" and len(v3.lookups) > 1:
                raise Left("none: its words in a message of several parts are not known")
            other = Other(kind, question.strip())
        try:
            parts.append(from_choice(c, v4_agent, question, other))
        except ValueError as e:
            raise Left(f"{c.tool}: {str(e).splitlines()[0]}") from None
    v4 = LookupsV4(parts=parts)
    a3, a4 = agent(v3, False), agent(v4, True)
    want, got = a3.read(question), a4.read(question)
    if _weekless(got) != _weekless(want):
        raise Left(f"different lookups: v3 {want} / v4 {got}")
    if len(a3.others) != len(a4.others):
        raise Left("a different number of parts left for others")
    if any(t["problem"] for t in a4.trace):
        raise Left(f"v4's answer fails a check: {a4.trace[-1]['problem']}")
    messages = [{"role": "system", "content": SYSTEM_V4}, {"role": "user", "content": a4._prompt(question)},
                {"role": "assistant", "content": v4.model_dump_json(exclude_defaults=True)}]
    return {**row, "messages": messages, "schema": LOOKUP_SCHEMA, "migrated_from": 3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=V3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--files", nargs="+", default=["train", "val", "distilled"])
    ap.add_argument("--show", action="store_true", help="print every row whose v4 answer gives other lookups")
    a = ap.parse_args()

    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    groups = {r["id"]: r.get("group", "") for p in (V2_TEMPLATES, V3_TEMPLATES)
              for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    depts: dict[int, dict] = {}
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in a.files:
        rows = [json.loads(x) for x in (a.source / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        kept, why = [], Counter()
        for row in rows:
            seed = row["meta"].get("seed", 0)
            dept = depts.setdefault(seed, dict(departments(instance, seed)))[row["department"]]
            try:
                kept.append(migrate_row(row, dept, groups))
            except Left as e:
                why[re.sub(r" v3 .*", "", str(e))] += 1
                if a.show and str(e).startswith("different"):
                    print(f"  left out {row['id']}: {str(e)[:400]}")
        with (a.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        report[name] = {"v3": len(rows), "v4": len(kept), "left out": dict(why.most_common())}
        print(f"{name}: {len(kept)}/{len(rows)} rows; left out: {dict(why.most_common())}")
    (a.out / "manifest.json").write_text(json.dumps({"task": "lookup", "version": 4, "schema": LOOKUP_SCHEMA,
                                                     "from": a.source.relative_to(ROOT).as_posix(), "files": report},
                                                    indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
