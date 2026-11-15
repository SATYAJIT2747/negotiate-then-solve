"""Training data for nts-lookup-v3_2 and nts-lookup-v4: the same rows in two answer formats, so the two runs
differ only in the format (v3's flat lookups, or v4's vocabulary, ``agents.lookups_v4``).

    uv run python -m training.lookup_data_v3_2      # writes data/lookup-train-v3_2/ and data/lookup-train-v4/

What is new against v3's data (``training.lookup_data_v3``):

1. **a request or a rule question keeps its words**: v3.2's "none" carries them in ``request`` (``SYSTEM_V3_2``),
   v4's ``request`` and ``rule`` parts in ``text``, so the chat sends on only the request and asks the rule reader
   only the rule question. A part is wrapped in markers before the message is filled, so its words are exactly
   what the message says;
2. **mixed messages**: 1 to 4 questions with 1 to 3 requests, now and then a rule question or a "thanks";
3. **long messages**: 4 to 8 questions (v3 had at most 5), each part with its own names, sometimes a request;
4. **the slips v3 makes** (``data/lookup-v3_2-train-templates.jsonl``): "a 2 hr slot" (length), "is Rao away"
   (changes), "the exam hall" (unknown), "any section free for a guest session" (who_is_free sections), "list the
   teachers" (describe), "when are we and Dr X both free" (together), rule questions on hours in a row;
5. **replay**: v3's rows (generated and Gemini's rewordings) again, so nothing v3 knows is forgotten. v3's mixed
   messages are left out: their requests' words were not kept (2. writes such messages again).

Every row is checked by ``ToolAgent``'s own checks in both formats (v3.2: the lookups its labels expect; v4: the
same lookups as v3.2, ``training.lookup_data_v4.migrate_row``), and kept only if both pass, so the two data sets
have the same rows. Templates that are test templates are refused; one template in ten is kept for validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from agents.lookups_v4 import LOOKUP_SCHEMA
from agents.questions import from_now
from agents.tool_agent import SYSTEM_V3_2, ChoiceV3, Lookups, Other, ToolAgent
from core.instance import Instance
from evaluation.lookup_set import SEMESTER_START, Filled, TemplateFiller, apart

from .lookup_data import ROOT, TEMPLATES as V2_TEMPLATES, _shape, a_moment, departments
from .lookup_data_v3 import JOINERS, TEMPLATES as V3_TEMPLATES, FixedLookups, _rename, as_written, choice_v3, matches
from .lookup_data_v4 import QUESTION, Left, migrate_row

TEMPLATES = ROOT / "data" / "lookup-v3_2-train-templates.jsonl"
TESTS = [ROOT / "data" / p for p in ("lookup-test-templates.jsonl", "lookup-test-templates-v2.jsonl",
                                     "lookup-v3-test-templates.jsonl")]
V3_DATA = ROOT / "data" / "lookup-train-v3"
OUT_V3_2, OUT_V4 = ROOT / "data" / "lookup-train-v3_2", ROOT / "data" / "lookup-train-v4"
MAX_PARTS = 8
LONG = [4] * 35 + [5] * 30 + [6] * 20 + [7] * 10 + [8] * 5  # questions in a long message, in percent
RENAME = {i: {f"{base}{j}": f"{base}{i}{j or 1}" for base in ("teacher", "section", "room", "lab", "course", "day", "time")
              for j in ("", "2", "3")} for i in range(2, MAX_PARTS + 1)}
OPEN, CLOSE = "\x02", "\x03"  # around each part while the message is filled: its words are found after


def kind_of(row: dict) -> str:
    """A template that is not a timetable question: request, rule or other (its group)."""
    return row.get("group") if row.get("group") in ("request", "rule") else "other"


def build(instance: Instance, seed: int, fills: int, mixed: int, long: int) -> tuple[list[dict], list[dict], list[str], Counter]:
    """The new rows (train, validation), each with its ``others`` (what each "none" part is, with its words); the
    rows left out, with why; what the rows cover."""
    load = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]  # noqa: E731
    v2, v3, v32 = load(V2_TEMPLATES), load(V3_TEMPLATES), load(TEMPLATES)
    tests = {_shape(r["template"]) for p in TESTS for r in load(p)}
    if clash := [r["id"] for r in v32 if _shape(r["template"]) in tests]:
        raise SystemExit(f"training templates that are test templates (change them): {clash}")
    depts = departments(instance, seed)
    rng = random.Random(f"v3_2-{seed}")
    train, val, dropped, cover = [], [], [], Counter()

    def held_out(ids: list[str]) -> bool:
        return any(int(hashlib.sha1(i.encode()).hexdigest(), 16) % 10 == 0 for i in ids)

    def one(parts: list[dict], k: int, kind: str) -> None:
        r = random.Random(f"{seed}-{kind}-{'+'.join(p['id'] for p in parts)}-{k}")
        dept_name, dept = depts[k % len(depts)]
        now, week = a_moment(r, dept.calendar.weeks)
        filler = TemplateFiller(dept, current_week=week, now=now, semester_start=SEMESTER_START)
        when = r.choice([("this week", week), ("next week", week + 1), ("", week)])
        if when[1] > dept.calendar.weeks:
            when = ("", week)
        rows = []
        for i, p in enumerate(parts, 1):
            p = _rename(p, RENAME[i]) if i > 1 else p
            had_when = "{when}" in p["template"]
            p = {**p, "template": re.sub(r"\s+", " ", p["template"].replace("{when}", when[0])).replace(" ?", "?").strip()}
            if had_when:
                p["week"] = when[1]
            rows.append(p)
        if len({p.get("asker", "teacher") for p in rows}) > 1:
            return
        heads = [OPEN + p["template"].rstrip(" ?.") + CLOSE for p in rows]
        if len(rows) == 1:
            template = heads[0]
        elif len(rows) == 2:
            template = r.choice(JOINERS).format(a=heads[0], b=heads[1])
        else:
            template = r.choice(["; ".join(heads[:-1]) + r.choice([", and ", "; also "]) + heads[-1],
                                 "? ".join(heads) + "?", "\n".join(f"{n}. {h}?" for n, h in enumerate(heads, 1))])
        joined = {k2: v for k2, v in rows[0].items() if k2 not in ("choice", "week")}
        filled = filler.fill(joined | {"template": template, "args": {}, "tool": "free_hours"}, r)
        if filled is None:
            return
        words = [w.strip() for w in re.findall(f"{OPEN}(.*?){CLOSE}", filled.question, re.S)]
        question = re.sub(f"[{OPEN}{CLOSE}]", "", filled.question)
        wants, choices, others = [], [], []
        try:
            for p, said in zip(rows, words, strict=True):
                p_week = filler._week(p)
                f = Filled(p["id"], p.get("group", ""), question, filled.asker, p["tool"],
                           {k2: filler._label(k2, v) for k2, v in p.get("args", {}).items()}, p_week)
                if f.args.pop("next", False):
                    p_week, after = from_now(dept, week, now)
                    f.week, f.args["after"] = p_week, list(after)
                filler._resolve(p["tool"], f.args)
                args = {k2: v for k2, v in f.args.items() if v not in (None, [], False)}
                wants += [{"tool": p["tool"], "args": x, "week": p_week}
                          for x in (apart(p["tool"], args, p["split"]) if p.get("split") else [args])]
                if p["tool"] is None:
                    other = Other(kind_of(p), said if kind_of(p) != "other" else "")
                    others.append(other)
                    choices.append(ChoiceV3(tool="none", week=week, request=other.text or None))
                else:
                    choices.append(as_written(choice_v3(p, f, filler, week), filler))
        except (KeyError, ValueError, LookupError) as e:
            dropped.append(f"{kind} {[p['id'] for p in parts]} {question!r}: {e}")
            return
        answer = Lookups(lookups=choices)
        agent = ToolAgent(dept, FixedLookups(answer), filled.asker, week, today=now.date(), now=now,
                          semester_start=SEMESTER_START, multi=True, words=True)
        call = agent.read(question)
        got = call.args["parts"] if call is not None and call.tool == "multi" else ([call] if call else [])
        expect = [w for w in wants if w["tool"] is not None]
        ok = (not expect and call is None) or (len(got) == len(expect) and all(matches(w, g) for w, g in zip(expect, got)))
        if not ok or any(t["problem"] for t in agent.trace) or len(agent.others) != len(others):
            dropped.append(f"{kind} {[p['id'] for p in parts]} {question!r}: gives {call}; want {wants}")
            return
        row = {"id": f"{kind}-{'+'.join(p['id'] for p in parts)}-{k}", "task": "lookup", "kind": kind,
               "templates": [p["id"] for p in parts], "department": dept_name,
               "meta": {"asker": filled.asker, "now": now.isoformat(), "week": week, "seed": seed},
               "messages": [{"role": "system", "content": SYSTEM_V3_2}, {"role": "user", "content": agent._prompt(question)},
                            {"role": "assistant", "content": answer.model_dump_json(exclude_defaults=True)}],
               "others": others}
        (val if held_out([p["id"] for p in parts]) else train).append(row)
        cover[f"kind {kind}"] += 1
        cover[f"parts {len(parts)}"] += 1
        for o in others:
            cover[f"part {o.kind}"] += 1

    for row in v32:  # 4. the slips, and the new requests, rule questions and "thanks"
        for k in range(fills):
            one([row], k, "slip")
    questions = [r for r in v2 + v3 + v32 if r["tool"] not in (None, "unclear") and r.get("asker") != "rep"]
    requests = [r for r in v2 + v32 if r["tool"] is None and r.get("group") == "request" and r.get("asker") != "rep"]
    rules = [r for r in v2 + v32 if r["tool"] is None and r.get("group") == "rule"]
    chatter = [r for r in v2 + v32 if r["tool"] is None and r.get("group") == "other"]
    for k in range(mixed):  # 2. questions with 1 to 3 requests, a rule question or a "thanks" now and then
        parts = rng.sample(questions, rng.choice([1, 1, 2, 2, 3, 4])) + rng.sample(requests, rng.choice([1, 1, 1, 2, 2, 3]))
        parts += rng.sample(rules, 1) if rng.random() < 0.25 else []
        parts += rng.sample(chatter, 1) if rng.random() < 0.1 else []
        rng.shuffle(parts)
        one(parts[:MAX_PARTS], k, "mixed")
    for k in range(long):  # 3. long messages
        parts = rng.sample(questions, rng.choice(LONG))
        if rng.random() < 0.3 and len(parts) < MAX_PARTS:
            parts.insert(rng.randrange(len(parts) + 1), rng.choice(requests))
        one(parts, k, "long")
    return train, val, dropped, cover


def replay(name: str, groups: dict[str, str]) -> list[dict]:
    """v3's rows in v3.2's prompt, with ``others``: a message that is wholly a request or rule question keeps its
    words (the message). v3's mixed messages are left out: their requests' words were not kept."""
    out = []
    for line in (V3_DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lookups = Lookups.model_validate_json(row["messages"][-1]["content"])
        if any(c.tool == "none" for c in lookups.lookups) and len(lookups.lookups) > 1:
            continue
        question = QUESTION.search(row["messages"][1]["content"]).group(1)
        others = []
        if lookups.lookups[0].tool == "none":
            kind = kind_of({"group": groups.get(row["templates"][0], "")})
            others = [Other(kind, question.strip() if kind != "other" else "")]
            lookups = Lookups(lookups=[lookups.lookups[0].model_copy(update={"request": others[0].text or None})])
        out.append({**row, "messages": [{"role": "system", "content": SYSTEM_V3_2}, row["messages"][1],
                                        {"role": "assistant", "content": lookups.model_dump_json(exclude_defaults=True)}],
                    "others": others, "kind": f"replay {row.get('kind')}"})
    return out


def both(rows: list[dict], instance: Instance, groups: dict[str, str], why: Counter) -> tuple[list[dict], list[dict]]:
    """Each row in v3.2 and in v4, kept only if its v4 answer gives the same lookups (``migrate_row``)."""
    depts: dict[int, dict] = {}
    v32, v4 = [], []
    for row in rows:
        seed = row["meta"].get("seed", 0)
        dept = depts.setdefault(seed, dict(departments(instance, seed)))[row["department"]]
        others = [Other(*o) for o in row.pop("others")]
        try:
            v4.append({**migrate_row(row, dept, groups, others), "migrated_from": None})
        except Left as e:
            why[f"v4: {re.sub(r' v3 .*', '', str(e))}"] += 1
            continue
        v32.append({**row, "schema": 3})
    return v32, v4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fills", type=int, default=8, help="rows per new template")
    ap.add_argument("--mixed", type=int, default=420, help="messages with questions and requests")
    ap.add_argument("--long", type=int, default=320, help="messages of 4 to 8 questions")
    ap.add_argument("--replay", type=int, default=1700, help="v3's training rows again (a sample)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from web.app import World

    instance = World(parser_mode="offline", seed_history=False, deadline=60).instance
    groups = {r["id"]: r.get("group", "") for p in (V2_TEMPLATES, V3_TEMPLATES, TEMPLATES)
              for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    train, val, dropped, cover = build(instance, a.seed, a.fills, a.mixed, a.long)
    old_train = replay("train", groups)
    rows = {"train": train + random.Random(a.seed).sample(old_train, min(a.replay, len(old_train))),
            "val": val + replay("val", groups), "distilled": replay("distilled", groups)}
    why: Counter = Counter()
    report = {}
    for out in (OUT_V3_2, OUT_V4):
        out.mkdir(parents=True, exist_ok=True)
    for name, part in rows.items():
        v32, v4 = both(part, instance, groups, why)
        order = list(range(len(v32)))
        random.Random(a.seed).shuffle(order)
        for out, made in ((OUT_V3_2, v32), (OUT_V4, v4)):
            with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(made[i], ensure_ascii=False) + "\n" for i in order)
        report[name] = {"rows": len(v32), "kinds": dict(Counter(r["kind"].split("+")[0] for r in v32).most_common())}
        print(f"{name}: {len(v32)} rows in both formats")
    for out, version, schema in ((OUT_V3_2, 3.2, 3), (OUT_V4, 4, LOOKUP_SCHEMA)):
        (out / "manifest.json").write_text(json.dumps({
            "task": "lookup", "version": version, "schema": schema, "seed": a.seed, "fills": a.fills, "mixed": a.mixed,
            "long": a.long, "replay": a.replay, "files": report, "new rows left out": len(dropped),
            "left out of both formats": dict(why), "cover": dict(sorted(cover.items()))}, indent=1), encoding="utf-8")
    print(f"new rows: {len(train)} train, {len(val)} validation; {len(dropped)} left out; not in v4: {dict(why)}")
    for k, v in sorted(cover.items()):
        print(f"  {k:24} {v}")
    for d in dropped[:12]:
        print("LEFT OUT", d[:300])


if __name__ == "__main__":
    main()
