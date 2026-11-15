"""Before and after a phase's changes: every message in ``<phase>/requests.jsonl`` sent to the chat
(``POST /api/chat``, as the side chat sends it) with the model on the GPU server, and what came back scored
against the row's ``want``:

- ``lookups``: the tools of the lookups answered, in order (a list of choices for a part where either will
  do; ``null``: not scored; ``[]``: no lookup answered);
- ``request``: whether a "send this as a request?" card came back;
- ``rule``: whether a rule answer came back;
- ``card_has`` / ``card_lacks`` (optional): words the request card must / must not hold, any case ("there" filled
  in from the answer: memory v2);
- ``asks`` (optional): a question back about the request instead of a card ("there" after two free times).

    uv run python before_after/run.py before_after/phase_1 --out before   # before the changes
    uv run python before_after/run.py before_after/phase_1 --out after    # after them

The chat is emptied before each message, so no message is read as a follow-up of the one before. Nothing is
sent as a request: a card is only offered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def tools(reply: dict) -> list[str]:
    call = reply["call"]
    return [p["tool"] for p in call["args"]["parts"]] if call["tool"] == "multi" else [call["tool"]]


def scored(want: dict, got: dict) -> dict[str, bool]:
    marks = {"request": got["request"] == want["request"], "rule": got["rule"] == want["rule"]}
    card = (got["card"] or "").lower()
    if "card_has" in want:
        marks["card has"] = all(w.lower() in card for w in want["card_has"])
    if "card_lacks" in want:
        marks["card lacks"] = bool(card) and not any(w.lower() in card for w in want["card_lacks"])
    if "asks" in want:
        marks["asks"] = (got["asked"] is not None) == want["asks"]
    if want["lookups"] is not None:
        wanted = [w if isinstance(w, list) else [w] for w in want["lookups"]]
        marks["lookups"] = len(wanted) == len(got["lookups"]) and all(g in w for g, w in zip(got["lookups"], wanted))
    return marks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", type=Path, help="the phase's folder, with requests.jsonl")
    ap.add_argument("--out", required=True, help="the results' name: before, after, or the adapter's (v3_2, v4)")
    ap.add_argument("--url", default="http://10.1.25.191:8080/v1", help="llama-server with the adapters")
    ap.add_argument("--lookup", default=None, help="the lookup adapter to read with (default: adapters-lora.toml's), "
                                                   "e.g. nts-lookup-v4")
    a = ap.parse_args()

    adapters = ROOT / "adapters-lora.toml"
    if a.lookup:  # a copy of the adapters file with "lookup" on the adapter asked for (the file itself unchanged)
        lines = [x for x in adapters.read_text(encoding="utf-8").splitlines()
                 if not re.match(r'^\S+\s*=\s*\["lookup"\]', x)]
        at = lines.index("[adapters]") + 1
        adapters = a.phase / f"adapters-{a.lookup}.toml"
        adapters.write_text("\n".join(lines[:at] + [f'{a.lookup} = ["lookup"]'] + lines[at:]) + "\n", encoding="utf-8")
    os.environ.update({"NTS_PARSER": "local", "NTS_LOCAL_URL": a.url, "NTS_PERSIST": "0",
                       "NTS_ADAPTERS": str(adapters)})
    from fastapi.testclient import TestClient

    from web.app import World, create_web_app

    world = World(parser_mode="local", seed_history=False, deadline=60)
    print("lookup adapter:", world.lookup_adapter, "| memory:", world.memory_on,
          "| model reachable:", world.model_link.usable())
    c = TestClient(create_web_app(world))
    rows = [json.loads(x) for x in (a.phase / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    passed, resent = 0, 0

    def ask(row: dict) -> list[dict]:
        """The replies to a row's message, read by the model: the network to the server drops for moments (WinError
        10065), and the app then answers with the regex until it sees the server again. Such a reply is not the
        model's: wait for the server and send the message again."""
        nonlocal resent
        me = {"X-User": row["as"]}
        for attempt in range(10):  # the drops come in bursts (a loose cable)
            c.delete("/api/chat", headers=me)
            r = c.post("/api/chat", json={"text": row["text"]}, headers=me)
            r.raise_for_status()
            replies = [m for m in r.json() if m["role"] == "assistant"]
            fallback = any(m.get("reader") in ("regex", "request words") for m in replies)
            if not fallback and world.model_link.up is not False:
                return replies
            resent += 1
            print(f"  {row['id']}: answered without the model; waiting for the server, then again", flush=True)
            for _ in range(30):  # up to 5 minutes
                time.sleep(10)
                if world.model_link.usable() and world.model_link.up is not False:
                    break
        return replies

    with (a.phase / f"{a.out}.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            replies = ask(row)
            looked = next((m for m in replies if m["kind"] == "lookup"), None)
            card = next((m for m in replies if m["kind"] == "confirm"), None)
            rule = next((m for m in replies if m["kind"] == "rule"), None)
            asked = next((m for m in replies if m["kind"] == "other" and m.get("about")), None)
            got = {"lookups": tools(looked) if looked else [], "request": card is not None, "rule": rule is not None,
                   "reader": (looked or card or {}).get("reader"),
                   "answer": looked["answer"] if looked else None,
                   "card": card["text"] if card else None,
                   "rule_answer": rule["answer"] if rule else None,
                   "asked": asked["text"] if asked else None,
                   "replies": [m["kind"] for m in replies]}
            marks = scored(row["want"], got)
            ok = all(marks.values())
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'} {row['id']} {row['kind']:20} got {got['replies']} {got['lookups']}"
                  + ("" if ok else f"  (wrong: {', '.join(k for k, v in marks.items() if not v)})"))
            out.write(json.dumps({**row, "got": got, "marks": marks, "pass": ok}, ensure_ascii=False) + "\n")
    print(f"\n{passed}/{len(rows)} pass; every reply in {a.phase / (a.out + '.jsonl')}"
          + (f" ({resent} sent again after the server dropped)" if resent else ""))
    if not world.model_link.usable() or world.model_link.up is False:
        print("WARNING: the model was marked down during the run (a server restart?): the regex and the request "
              "words answered instead, so these results are not the model's. Run again.")


if __name__ == "__main__":
    main()
