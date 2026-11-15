"""Everyday requests through the whole web app, each checked against what should happen: a smoke test of the
pipeline as it is served (reader, policy check, solver, replies), not of one component on its test split.

    uv run python -m evaluation.smoke                                   # offline: the rule-based stand-ins
    uv run python -m evaluation.smoke --parser local --local URL --adapters adapters-lora.toml

``data/smoke-requests.jsonl`` has one request per line, sent by Dr. Menon in a fresh demo world, with what is
expected: the status it ends in, a constraint it must (``constraint``) or must not (``no_constraint``) set,
a rule the reply rests on (``cites``), words the reply must not have (``reply_lacks``), and whether the
message is flagged as carrying an instruction to the system (``injection``). Wordings come from using the
app, so they find what a test split can miss: "I want Mondays off" read as a wish FOR Mondays went unseen
there. Exits 1 when a request does not do what it should.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SMOKE = Path(__file__).parents[2] / "data" / "smoke-requests.jsonl"
SENDER = "Dr. Menon"


def check(case, expect: dict) -> list[str]:
    """What is wrong with how ``case`` ended, against ``expect``; empty when nothing is."""
    wrong = []
    status = case.status.value
    if "status" in expect and status not in expect["status"]:
        wrong.append(f"status {status}, expected {' or '.join(expect['status'])}")
    made = [c for c in (case.parse.output.constraints if case.parse and case.parse.output else [])]

    def has(want: dict) -> bool:
        return any(c.type == want["type"] and set(want.get("days", [])) <= set(c.days or []) for c in made)

    if (want := expect.get("constraint")) and not has(want):
        wrong.append(f"no {want['type']} constraint on {', '.join(want.get('days', []))}: read as "
                     f"{[(c.type, c.days) for c in made] or case.parse.action.value if case.parse else None}")
    if (bad := expect.get("no_constraint")) and has(bad):
        wrong.append(f"set a {bad['type']} constraint on {', '.join(bad.get('days', []))}")
    if (rule := expect.get("cites")) and not (case.policy and rule in case.policy.cited):
        wrong.append(f"does not cite {rule} (cited {case.policy.cited if case.policy else None})")
    if (words := expect.get("reply_lacks")) and words.lower() in (case.reply or "").lower():
        wrong.append(f"reply says {words!r}: {case.reply[:120]!r}")
    if "injection" in expect and bool(case.routing.get("injection")) != expect["injection"]:
        wrong.append(f"injection flag {bool(case.routing.get('injection'))}, expected {expect['injection']}")
    return wrong


def run(parser_mode: str, rows: list[dict], timeout: float = 600) -> list[dict]:
    from web.world import World

    out = []
    for row in rows:
        w = World(parser_mode=parser_mode, seed_history=False, deadline=1)  # a time offered is chosen in 1 s
        me = next(f.id for f in w.instance.faculty if f.name == SENDER)
        start = time.time()
        r = w.intake.from_portal(me, row["text"])
        w.submit(r)
        while time.time() - start < timeout:  # a held request (model away) is read when the model is back
            t = w.threads.get(r.id)
            if r.id in w.orch.cases and t is not None and not t.is_alive():
                break
            time.sleep(0.2)
        case = w.orch.cases.get(r.id)
        wrong = ["not read in time"] if case is None else check(case, row["expect"])
        out.append({"id": row["id"], "text": row["text"], "ok": not wrong, "wrong": wrong,
                    "status": case.status.value if case else None, "reader": case.routing.get("reader") if case else None,
                    "reply": case.reply if case else None, "seconds": round(time.time() - start, 1)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", choices=["offline", "local", "gemini"], default="offline")
    ap.add_argument("--local", metavar="URL", help="llama-server URL (NTS_LOCAL_URL)")
    ap.add_argument("--adapters", help="adapter registry (NTS_ADAPTERS), e.g. adapters-lora.toml")
    ap.add_argument("--data", type=Path, default=SMOKE)
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.local:
        os.environ["NTS_LOCAL_URL"] = args.local
    if args.adapters:
        os.environ["NTS_ADAPTERS"] = args.adapters
        os.environ["NTS_INTENTS"] = "1"  # as scripts/host.ps1 serves it
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only.split(","))]
    results = run(args.parser, rows)
    for r in results:
        mark = "ok  " if r["ok"] else "FAIL"
        print(f"{mark} {r['id']} [{r['status']}, {r['reader']}, {r['seconds']}s] {r['text'][:70]}")
        for w in r["wrong"]:
            print(f"       - {w}")
    passed = sum(r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} as expected ({args.parser})")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
