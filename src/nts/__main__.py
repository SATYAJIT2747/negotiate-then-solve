"""Run the agent from the command line; every episode is recorded in runs/telemetry.

    python -m nts init --store runs/agentic/demo.jsonl                      # a generated department, solved
    python -m nts ask "When am I free on Tuesday?" --as "Dr. Iyer" --week 5
    python -m nts ask "Please keep my Friday afternoons free" --as Iyer --week 5 --approve
    python -m nts ask "..." --model llama --url http://10.1.25.191:8080/v1 --adapter nts-agent-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nts.agents import COORDINATOR, READER
from nts.harness import Harness
from nts.kernel.generator import generate_department
from nts.kernel.instance import policy_constraints
from nts.models import GeminiChat, OpenAIChat
from nts.state import InvalidChange, Store, bootstrap
from nts.telemetry import DEFAULT_ROOT, default_tracer
from nts.tools import ALL_TOOLS, Refusal, Registry
from nts.tools.draft import OFFICE
from nts.tools.names import Directory

DEFAULT_STORE = Path("runs/agentic/demo.jsonl")
AGENTS = {"coordinator": COORDINATOR, "reader": READER}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nts")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="create a store for a generated department")
    i.add_argument("--store", type=Path, default=DEFAULT_STORE)
    i.add_argument("--seed", type=int, default=1)
    i.add_argument("--faculty", type=int, default=8)
    i.add_argument("--sections", type=int, default=3)
    i.add_argument("--rooms", type=int, default=4)
    i.add_argument("--labs", type=int, default=3)
    a = sub.add_parser("ask", help="run one episode")
    a.add_argument("text")
    a.add_argument("--store", type=Path, default=DEFAULT_STORE)
    a.add_argument("--as", dest="actor", default="office", help="a teacher's name, or 'office'")
    a.add_argument("--week", type=int)
    a.add_argument("--agent", choices=sorted(AGENTS), default="coordinator")
    a.add_argument("--model", choices=["gemini", "llama"], default="gemini")
    a.add_argument("--gemini-model")
    a.add_argument("--url", default="http://10.1.25.191:8080/v1")
    a.add_argument("--adapter")
    a.add_argument("--env", type=Path, help="a .env file with GEMINI_API_KEY")
    a.add_argument("--telemetry", type=Path, default=DEFAULT_ROOT)
    a.add_argument("--approve", action="store_true", help="commit a submitted draft")
    a.add_argument("--show", action="store_true", help="print every tool call and result")
    args = p.parse_args(argv)
    return init(args) if args.cmd == "init" else ask(args)


def init(args: argparse.Namespace) -> int:
    inst = generate_department(seed=args.seed, n_faculty=args.faculty, n_groups=args.sections,
                               n_lecture_rooms=args.rooms, n_labs=args.labs)
    store = bootstrap(inst, policy_constraints(inst), path=args.store)
    print(f"{args.store}: {len(inst.faculty)} teachers, {len(inst.groups)} sections, {len(inst.sessions)} classes, "
          f"version {store.version}")
    return 0


def ask(args: argparse.Namespace) -> int:
    if args.env:
        from dotenv import load_dotenv

        load_dotenv(args.env)
    store = Store.open(args.store)
    actor = OFFICE
    if args.actor.lower() not in ("office", OFFICE):
        try:
            actor = Directory(store.world.instance).teacher(args.actor).id
        except Refusal as r:
            print(f"--as: {r.error.message} {r.error.hint}", file=sys.stderr)
            return 2
    model = (GeminiChat(args.gemini_model) if args.model == "gemini"
             else OpenAIChat(args.url, adapter=args.adapter))
    harness = Harness(Registry(ALL_TOOLS), model, default_tracer(args.telemetry))
    out = harness.run(AGENTS[args.agent], args.text, store, actor, args.week)
    if args.show:
        for m in out.transcript[2:]:
            for c in m.tool_calls:
                print(f"  -> {c.name} {json.dumps(c.arguments, ensure_ascii=False)}")
            if m.role == "tool":
                print(f"  <- {m.content[:400]}")
    print(f"[{out.status}] {out.text or out.detail}")
    print(f"episode {out.episode_id}: {out.turns} turns, {out.tool_calls} calls ({out.refused} refused), "
          f"{out.usage.prompt_tokens}+{out.usage.completion_tokens} tokens, {out.seconds:.1f} s")
    if out.status == "submitted" and args.approve:
        try:
            store.commit(out.draft)
        except InvalidChange as e:
            print(f"not committed: {e}", file=sys.stderr)
            return 1
        print(f"committed: version {store.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
