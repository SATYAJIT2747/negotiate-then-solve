"""The test set for long messages: 1 to 5 timetable questions, 0 to 2 requests and now and then a rule question in
one message, scored for a model that reads several lookups a message (nts-lookup-v3 and after).

    uv run python -m evaluation.multi_set --build                                   # writes the set (no model)
    uv run python -m evaluation.multi_set --local http://10.1.25.191:8080/v1 --adapter nts-lookup-v3 \
        --out before_after/phase_1/multi/before.jsonl

Every part comes from the TEST templates (``lookup_set``'s and v3's), never the training ones, joined the ways
the training data joins them and a few it does not ("\\n", "Also,", "Then"). A message is right when its lookups
are all right, in order, and the model found as many other parts (requests and rule questions: v3's "none") as it
has. Also scored: each question part on its own (by position), and whether every request was found, as the chat
finds one (a part the model left for others, or request words beside the lookups).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from agents.questions import REQUEST, ToolCall, from_now
from core.instance import Instance

from .lookup_set import (
    NOW,
    SEMESTER_START,
    TEMPLATES,
    TEMPLATES_V2,
    WEEK,
    Filled,
    TemplateFiller,
    Unreachable,
    _show,
    _with_retries,
    right,
    with_first_names,
)

ROOT = Path(__file__).parents[2]
SET = ROOT / "data" / "lookup-multi-test.jsonl"
TEMPLATES_V3 = ROOT / "data" / "lookup-v3-test-templates.jsonl"
QUESTIONS = [1, 2, 2, 3, 3, 4, 5]  # question parts in a message, drawn in these proportions
REQUESTS = [0, 0, 1, 1, 2]
MAX_PARTS = 5  # v3 writes at most five lookups
# a part's placeholders get other names, so "{teacher}" in part 2 can be another teacher (as lookup_data_v3)
RENAME = {i: {f"{base}{j}": f"{base}{i}{j or 1}" for base in ("teacher", "section", "room", "lab", "course", "day", "time")
              for j in ("", "2", "3")} for i in range(2, 7)}
PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)(?:\.([a-z_]+))?\}")


def _rename(obj, names: dict[str, str]):
    if isinstance(obj, str):
        return PLACEHOLDER.sub(lambda m: "{" + names.get(m.group(1), m.group(1)) + (f".{m.group(2)}" if m.group(2) else "") + "}", obj)
    if isinstance(obj, list):
        return [_rename(x, names) for x in obj]
    if isinstance(obj, dict):
        return {k: _rename(v, names) for k, v in obj.items()}
    return obj


def _joined(heads: list[str], rng: random.Random) -> tuple[str, str]:
    """The parts as one message, and how they were joined (``train``: as the training data joins them)."""
    if len(heads) == 2:
        way = rng.choice(["{a} and {b}", "{a}? and {b}?", "{a}. Also, {b}", "{a}; {b}", "{a}? {b}?",  # as trained
                          "{a}.\n{b}.", "{a}? Then {b}."])  # not
        return way.format(a=heads[0], b=heads[1]), "train" if "\n" not in way and "Then" not in way else "new"
    way = rng.choice(["semicolons", "questions", "numbered", "lines", "also"])
    text = {"semicolons": "; ".join(heads[:-1]) + "; also " + heads[-1],
            "questions": "? ".join(heads) + "?",
            "numbered": "\n".join(f"{n}. {h}" for n, h in enumerate(heads, 1)),
            "lines": "\n".join(heads),
            "also": ". ".join(heads[:-1]) + ". Also, " + heads[-1] + "."}[way]
    return text, "train" if way in ("semicolons", "questions", "numbered") else "new"


def build(instance: Instance, n: int = 200, seed: int = 0) -> list[dict]:
    """``n`` messages, the same on every run (``seed``). Teachers' messages only (a class rep's "we" and a
    teacher's "my" don't mix); no "each one on its own" parts (they become several lookups, scored apart)."""
    rows = [json.loads(x) for p in (TEMPLATES, TEMPLATES_V2, TEMPLATES_V3)
            for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    teacher = [r for r in rows if r.get("asker", "teacher") != "rep" and not r.get("split")]
    pools = {"question": [r for r in teacher if r["tool"] not in (None, "unclear")],
             "request": [r for r in teacher if r["tool"] is None and r.get("group") == "request"],
             "rule": [r for r in teacher if r["tool"] is None and r.get("group") == "rule"]}
    partial = {r["id"] for r in rows if r.get("choice")}  # v3's tool templates: their labels give some arguments
    filler, out, k = TemplateFiller(instance), [], 0
    while len(out) < n:
        rng = random.Random(f"multi-{seed}-{k}")
        k += 1
        counts = {"question": rng.choice(QUESTIONS), "request": rng.choice(REQUESTS),
                  "rule": int(rng.random() < 0.15)}
        if sum(counts.values()) < 2:
            continue
        counts["question"] = max(1, min(counts["question"], MAX_PARTS - counts["request"] - counts["rule"]))
        kinds = [kind for kind, c in counts.items() for _ in range(c)]
        rng.shuffle(kinds)
        picked = {kind: rng.sample(pools[kind], c) for kind, c in counts.items()}
        when = rng.choice([("this week", WEEK), ("next week", WEEK + 1), ("", WEEK)])  # every part's the same
        parts = []
        for i, kind in enumerate(kinds, 1):
            p = picked[kind].pop()
            p = _rename(p, RENAME[i]) if i > 1 else p
            had_when = "{when}" in p["template"]
            p = {**p, "kind": kind,
                 "template": re.sub(r"\s+", " ", p["template"].replace("{when}", when[0])).replace(" ?", "?").strip()}
            if had_when:
                p["week"] = when[1]
            parts.append(p)
        text, joiner = _joined([p["template"].rstrip(" ?.") for p in parts], rng)
        filled = filler.fill({"id": "multi", "template": text, "tool": "free_hours", "args": {}}, rng)
        if filled is None:
            continue
        try:
            wants = [_want(filler, p, p["id"] in partial) for p in parts]
        except (KeyError, ValueError, LookupError):
            continue
        out.append({"id": f"M-{len(out) + 1:03d}", "message": filled.question, "asker": filled.asker,
                    "questions": counts["question"], "requests": counts["request"], "rules": counts["rule"],
                    "joiner": joiner, "parts": wants})
    return out


def _want(filler: TemplateFiller, p: dict, partial: bool) -> dict:
    """A part's label, filled with what the message was filled with (as ``TemplateFiller.fill``)."""
    week = filler._week(p)
    args = {k: filler._label(k, v) for k, v in p.get("args", {}).items()}
    if args.pop("next", False):
        week, after = from_now(filler.instance, filler.current_week, filler.now)
        args["after"] = list(after)
    filler._resolve(p["tool"], args)
    also = [{"tool": a["tool"], "args": {k: filler._label(k, v) for k, v in a.get("args", {}).items()}}
            for a in p.get("also", [])]
    return {"template": p["id"], "kind": p["kind"], "group": p.get("group", p["tool"]), "tool": p["tool"],
            "args": {k: v for k, v in args.items() if v not in (None, [], False)}, "week": week, "also": also,
            "partial": partial}


def score(row: dict, call: ToolCall | None, others: list) -> dict:
    """What a reading of ``row`` got right: every lookup (``lookups``), each question part by position
    (``parts``), the other parts found (``others``: as many requests and rule questions as the message has; v4's
    "other" parts don't count), every request found as the chat finds one (``requests``), and all of it
    (``message``)."""
    others = [o for o in others if getattr(o, "kind", None) != "other"]
    got = call.args["parts"] if call is not None and call.tool == "multi" else ([call] if call else [])
    wants = [p for p in row["parts"] if p["kind"] == "question"]
    marks = [i < len(got) and right(Filled(p["template"], p["group"], row["message"], row["asker"], p["tool"],
                                           p["args"], p["week"], p["also"]), got[i], p["partial"])
             for i, p in enumerate(wants)]
    lookups = len(got) == len(wants) and all(marks)
    found = bool(others) or (call is not None and REQUEST.search(row["message"]) is not None)
    return {"message": lookups and len(others) == row["requests"] + row["rules"], "lookups": lookups,
            "parts": marks, "others": len(others) == row["requests"] + row["rules"],
            "requests": found if row["requests"] else True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help=f"write the set to --set ({SET.relative_to(ROOT)})")
    ap.add_argument("--set", type=Path, default=SET)
    ap.add_argument("--n", type=int, default=200, help="messages, with --build")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local", metavar="URL", nargs="?", const="http://localhost:8080/v1", default=None,
                    help="the local model on llama-server, e.g. http://10.1.25.191:8080/v1")
    ap.add_argument("--adapter", default="nts-lookup-v3", help="with --local: the LoRA adapter")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "multi-set.jsonl", help="every message and reading")
    a = ap.parse_args()

    from web.app import World

    instance = with_first_names(World(parser_mode="offline", seed_history=False, deadline=60).instance)
    if a.build:
        rows = build(instance, a.n, a.seed)
        a.set.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{len(rows)} messages in {a.set}")
        return
    if not a.local:
        raise SystemExit("--build, or --local URL to read the set")
    from agents.tool_agent import ToolAgent
    from language.local import LocalClient

    client = LocalClient(base_url=a.local, adapter=a.adapter, timeout=300)
    print("local model:", client.model)
    rows = [json.loads(x) for x in a.set.read_text(encoding="utf-8").splitlines() if x.strip()]
    by = defaultdict(lambda: defaultdict(list))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for row in rows:
            agent = ToolAgent(instance, client, row["asker"], WEEK, today=NOW.date(), now=NOW, multi=True,
                              semester_start=SEMESTER_START, dsl="v4" in a.adapter, words="v3_2" in a.adapter)
            try:
                call = _with_retries(agent.read, row["message"])
            except Unreachable as e:
                raise SystemExit(f"stopped: {e}") from None
            marks = score(row, call, agent.others)
            for key in ("ALL", f"{row['questions']} questions", f"{row['requests']} requests",
                        f"rule question {'yes' if row['rules'] else 'no'}", f"joined {row['joiner']}"):
                for m in ("message", "lookups", "others", "requests"):
                    by[key][m].append(marks[m])
                by[key]["parts"] += marks["parts"]
            got = call.args["parts"] if call is not None and call.tool == "multi" else ([call] if call else [])
            out.write(json.dumps({**row, "got": [_show(g) for g in got], "others": agent.others, "marks": marks,
                                  "tries": agent.trace}, ensure_ascii=False) + "\n")
            out.flush()
    print(f"\n{len(rows)} messages\n")
    print(f"{'':22}{'message':>10}{'lookups':>10}{'parts':>12}{'others':>10}{'requests':>10}")
    for key in sorted(by, key=lambda k: (k != "ALL", k)):
        cells = [f"{sum(v)}/{len(v)}" for v in (by[key][m] for m in ("message", "lookups", "parts", "others", "requests"))]
        print(f"{key:22}{cells[0]:>10}{cells[1]:>10}{cells[2]:>12}{cells[3]:>10}{cells[4]:>10}")
    print(f"\nevery message and reading: {a.out}")


if __name__ == "__main__":
    main()
