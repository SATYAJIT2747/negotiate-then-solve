"""Memory, then lookup, end to end: nts-memory's test conversations (training.memory_data) through nts-memory and
then nts-lookup-v3, as the chat runs them. Is the follow-up looked up as it should be?

    uv run python -m evaluation.end_to_end --local http://10.1.25.191:8080/v1 --out before_after/phase_1/end_to_end/before.jsonl

A conversation's test row says what the follow-up should be made into (the message in full, the same message,
or a question back), not which lookups that gives. So the lookups to expect are the lookup model's own reading
of the expected message: the score is whether memory's rewrite is looked up as the right rewrite would be. Two
readings are the same when every lookup has the same tool, week and arguments, and the same number of parts is
left for others (requests, rule questions). A conversation that expects a question back is right when memory
asks one.

Each miss is put on one stage: ``memory`` (memory_set's check finds the rewrite wrong) or ``lookup`` (the
rewrite is right, yet it is read differently from the expected wording). Also scored: the follow-up read as
typed, without memory (what memory adds).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from agents.memory import Memory, Turn
from agents.questions import ToolCall
from agents.tool_agent import ToolAgent
from evaluation.lookup_set import SEMESTER_START, Unreachable, _norm, _with_retries
from evaluation.memory_set import right as memory_right

ROOT = Path(__file__).parents[2]


def _key(call: ToolCall | None) -> list:
    """A reading in a form two equal readings share: each lookup's tool, week and arguments (lists in any order)."""
    parts = call.args["parts"] if call is not None and call.tool == "multi" else ([call] if call else [])
    return [(p.tool, p.week, p.confident, json.dumps(_norm(p.args), sort_keys=True, default=str)) for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "memory-train" / "test.jsonl")
    ap.add_argument("--local", metavar="URL", required=True, help="llama-server with the adapters")
    ap.add_argument("--memory", default=None, help="the memory adapter (default: adapters-lora.toml's)")
    ap.add_argument("--lookup", default="nts-lookup-v3", help="the lookup adapter")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "end-to-end.jsonl")
    a = ap.parse_args()

    from language.adapters import local_client, registry
    from language.local import LocalClient
    from training.memory_data import build_departments
    from web.app import World

    memory_client = local_client("memory", base_url=a.local, adapter=a.memory, reg=registry(ROOT / "adapters-lora.toml"))
    lookup_client = LocalClient(base_url=a.local, adapter=a.lookup, timeout=300)
    print("memory:", a.memory or "adapters-lora.toml's", "| lookup:", lookup_client.model)
    departments = dict(build_departments(World(parser_mode="offline", seed_history=False, deadline=60).instance, test=True))
    rows = [json.loads(x) for x in a.test.read_text(encoding="utf-8").splitlines() if x.strip()][: a.limit]
    score: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    blame: dict[str, int] = defaultdict(int)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for n, row in enumerate(rows, 1):
            m, dept = row["meta"], departments[row["department"]]
            now = datetime.fromisoformat(m["now"])
            history = [Turn(who, text, datetime.fromisoformat(at)) for who, text, at in m["history"]]

            def read(text: str) -> tuple[list, list[str]]:
                agent = ToolAgent(dept, lookup_client, m["asker"], m["week"], today=now.date(), now=now,
                                  semester_start=SEMESTER_START, multi=True, dsl="v4" in a.lookup,
                                  words="v3_2" in a.lookup)
                call = _with_retries(agent.read, text)
                return _key(call), [o for o in agent.others if o.kind != "other"]

            try:
                got = _with_retries(lambda text: Memory(dept, memory_client, m["asker"], m["week"], now=now)
                                    .rewrite(text, history), m["message"])
                want = m["answer"]
                if "ask" in want:
                    ok, typed_ok, reading = got.ask is not None, False, None
                else:
                    expected = read(want.get("message", m["message"]))
                    reading = None if got.ask else read(got.text)
                    ok = reading is not None and reading[0] == expected[0] and len(reading[1]) == len(expected[1])
                    typed = read(m["message"])
                    typed_ok = typed[0] == expected[0] and len(typed[1]) == len(expected[1])
            except Unreachable as e:
                raise SystemExit(f"\nstopped at {n}/{len(rows)}: {e}; run the same command again (cached)") from None
            memory_ok = memory_right(row, got, dept)
            stage = None if ok else "memory" if not memory_ok else "lookup"
            if stage:
                blame[stage] += 1
            for kind in (row["kind"], "ALL"):
                score[kind]["end to end"].append(ok)
                score[kind]["memory"].append(memory_ok)
                score[kind]["as typed"].append(typed_ok)
            said = {"ask": got.ask} if got.ask else {"message": got.text} if got.changed else {"same": True}
            out.write(json.dumps({"id": row["id"], "kind": row["kind"], "message": m["message"], "want": want,
                                  "memory": said, "memory_right": memory_ok, "lookups": reading[0] if reading else None,
                                  "right": ok, "as_typed_right": typed_ok, "miss": stage}, ensure_ascii=False,
                                 default=str) + "\n")
            out.flush()
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    print(f"\n{len(rows)} conversations\n")
    print(f"{'kind':12}{'end to end':>12}{'memory':>10}{'as typed':>10}")
    for kind in sorted(score, key=lambda k: (k == "ALL", k)):
        cells = [f"{sum(v)}/{len(v)}" for v in (score[kind][c] for c in ("end to end", "memory", "as typed"))]
        print(f"{kind:12}{cells[0]:>12}{cells[1]:>10}{cells[2]:>10}")
    print(f"\nmisses by stage: {dict(blame)}  (memory: the rewrite is wrong; lookup: right, read differently)")
    print(f"every conversation: {a.out}")


if __name__ == "__main__":
    main()
