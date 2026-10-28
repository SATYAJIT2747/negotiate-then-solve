"""Run the agent from the command line; every episode is recorded in runs/telemetry.

    python -m nts init --store runs/agentic/demo.jsonl                      # a generated department, solved
    python -m nts ask "When am I free on Tuesday?" --as "Dr. Iyer" --week 5
    python -m nts ask "Please keep my Friday afternoons free" --as Iyer --week 5 --approve
    python -m nts ask "Please move my class to Friday" --as Iyer --week 5 --talk   # you answer the agent's questions
    python -m nts ask "..." --model llama --url http://10.1.25.191:8080/v1 --adapter nts-agent-v1-read-100 --agent coordinator-v1
    python -m nts eval --model gemini --per-family 3 --seed 0      # a scenario suite, judged; report in runs/eval
    python -m nts export --out data/agent-v1 --models gemini        # SFT / DPO rows from judged episodes
    python -m nts serve --model llama --adapter <lora> --port 8100   # the API and its page, http://127.0.0.1:8100/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from nts.agents import (
    COORDINATOR,
    COORDINATOR_V1,
    COORDINATOR_V2,
    COORDINATOR_V3,
    COORDINATOR_V4,
    READER,
)
from nts.harness import AgentConfig, Harness, approve, best_of
from nts.kernel.generator import generate_department
from nts.kernel.instance import policy_constraints
from nts.models import GeminiChat, Model, OpenAIChat
from nts.protocol import tools_hash
from nts.state import Store, bootstrap
from nts.telemetry import DEFAULT_ROOT, default_tracer
from nts.tools import ALL_TOOLS, Refusal, Registry, Scope
from nts.tools.draft import OFFICE
from nts.tools.names import Directory

DEFAULT_STORE = Path("runs/agentic/demo.jsonl")
DEFAULT_HANDBOOK = Path("data/handbook.md")
AGENTS = {"coordinator": COORDINATOR, "coordinator-v1": COORDINATOR_V1, "coordinator-v2": COORDINATOR_V2,
          "coordinator-v3": COORDINATOR_V3, "coordinator-v4": COORDINATOR_V4, "reader": READER}


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
    a.add_argument("--approve", action="store_true", help="commit a submitted draft")
    a.add_argument("--show", action="store_true", help="print every tool call and result")
    a.add_argument("--talk", action="store_true", help="answer the agent's questions and offers at the terminal")
    e = sub.add_parser("eval", help="run a scenario suite and judge it")
    e.add_argument("--store", type=Path, default=DEFAULT_STORE)
    e.add_argument("--week", type=int, default=5)
    e.add_argument("--per-family", type=int, default=2)
    e.add_argument("--families", nargs="*", help="default: the families the agent has the tools for")
    e.add_argument("--suite", choices=["families", "compositional"], default="families",
                   help="compositional: the fourteen hand-written requests of nts.eval.compositional")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", type=Path, help="report path (default runs/eval/<model>-s<seed>.json)")
    e.add_argument("--rewrite", choices=["train", "heldout"],
                   help="requests rewritten by Gemini in this style set (cached; needs GEMINI_API_KEY the first time)")
    x = sub.add_parser("export", help="training rows from judged episodes")
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--models", nargs="*", help="only episodes of these models (substring match)")
    x.add_argument("--include-unjudged", action="store_true")
    x.add_argument("--keep-errors", action="store_true")
    x.add_argument("--stores", nargs="*", type=Path, help="only episodes run on these stores")
    x.add_argument("--agent", choices=sorted(AGENTS), help="only episodes of this agent's tools")
    v = sub.add_parser("serve", help="the API and its web page")
    v.add_argument("--store", type=Path, nargs="+", default=[DEFAULT_STORE], help="one store per department")
    v.add_argument("--users", type=Path, help="passwords (python -m nts passwd); none: sign in by name alone")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8100)
    v.add_argument("--simulate", action="store_true", help="stand-ins answer the agent's asks (no one waits)")
    w = sub.add_parser("passwd", help="set a sign-in password for the API")
    w.add_argument("name", help="a teacher's name, or 'office'")
    w.add_argument("--dept", default=DEFAULT_STORE.stem, help="the department: its store's file name without .jsonl")
    w.add_argument("--users", type=Path, default=Path("runs/agentic/users.json"))
    for q in (a, e, x, v):
        q.add_argument("--telemetry", type=Path, default=DEFAULT_ROOT)
    for q in (a, e, v):
        q.add_argument("--agent", choices=sorted(AGENTS), default="coordinator")
        q.add_argument("--model", choices=["gemini", "llama"], default="gemini")
        q.add_argument("--gemini-model")
        q.add_argument("--url", default="http://10.1.25.191:8080/v1")
        q.add_argument("--adapter")
        q.add_argument("--env", type=Path, help="a .env file with GEMINI_API_KEY")
        q.add_argument("--max-tokens", type=int, help="token budget of an episode (default: the agent's for Gemini, "
                                                       "none for a local model)")
        q.add_argument("--handbook", type=Path, default=DEFAULT_HANDBOOK, help="the regulations, for the rules tool")
        q.add_argument("--temperature", type=float, help="sampling (default 0, or 0.7 with --best-of)")
        q.add_argument("--max-seconds", type=float, help="time budget of an episode (default: the agent's); the model "
                                                          "never sees it, so a slow server need not cut episodes short")
    for q in (a, e):
        q.add_argument("--best-of", type=int, default=1, help="run each request n times and keep the run most agree on")
    args = p.parse_args(argv)
    return {"init": init, "ask": ask, "eval": evaluate, "export": export_rows, "serve": serve,
            "passwd": passwd}[args.cmd](args)


def init(args: argparse.Namespace) -> int:
    inst = generate_department(seed=args.seed, n_faculty=args.faculty, n_groups=args.sections,
                               n_lecture_rooms=args.rooms, n_labs=args.labs)
    store = bootstrap(inst, policy_constraints(inst), path=args.store)
    print(f"{args.store}: {len(inst.faculty)} teachers, {len(inst.groups)} sections, {len(inst.sessions)} classes, "
          f"version {store.version}")
    return 0


def handbook(args: argparse.Namespace):
    from nts.policy import Handbook

    return Handbook.load(args.handbook) if args.handbook and args.handbook.exists() else None


def budgeted(agent: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    """Local calls cost nothing: a local model gets no token budget (the turn budget still holds)."""
    limit = args.max_tokens if args.max_tokens is not None else (agent.budgets.max_tokens if args.model == "gemini" else None)
    seconds = args.max_seconds if args.max_seconds is not None else agent.budgets.max_seconds
    return replace(agent, budgets=replace(agent.budgets, max_tokens=limit, max_seconds=seconds))


def make_model(args: argparse.Namespace) -> Model:
    if args.env:
        from dotenv import load_dotenv

        load_dotenv(args.env)
    t = args.temperature if args.temperature is not None else (0.7 if getattr(args, "best_of", 1) > 1 else 0.0)
    if args.model == "gemini":
        return GeminiChat(args.gemini_model, temperature=t)
    return OpenAIChat(args.url, adapter=args.adapter, temperature=t)


def evaluate(args: argparse.Namespace) -> int:
    from nts.eval import (
        DEFAULT_FAMILIES,
        V1_FAMILIES,
        V2_FAMILIES,
        build_suite,
        run_suite,
        save,
    )

    model = make_model(args)
    store = Store.open(args.store)
    agent = AGENTS[args.agent]
    families = args.families or (V1_FAMILIES if Scope.TALK not in agent.scopes else
                                 V2_FAMILIES if Scope.POLICY not in agent.scopes else DEFAULT_FAMILIES)
    if args.suite == "compositional":
        from nts.eval.compositional import compositional_suite

        suite = compositional_suite(store.world)
    else:
        suite = build_suite(store.world, per_family=args.per_family, seed=args.seed, week=args.week, families=families)
    if args.rewrite:
        from nts.eval.rewrite import rewrite

        if args.env:
            from dotenv import load_dotenv

            load_dotenv(args.env)
        suite = rewrite(suite, args.rewrite, GeminiChat(args.gemini_model, temperature=1.0), store.world, log=True,
                        strict=True)
    verdicts = run_suite(suite, model, store, budgeted(agent, args), default_tracer(args.telemetry),
                         progress=True, handbook=handbook(args), best=args.best_of)
    path = args.out or Path("runs/eval") / f"{model.name.replace(':', '_').replace('+', '_')}-s{args.seed}{f'-best{args.best_of}' * (args.best_of > 1)}{f'-{args.rewrite}' * bool(args.rewrite)}.json"
    report = save(path, model.name, verdicts)
    for fam, st in report["summary"]["families"].items():
        print(f"  {fam:12} {st['passed']}/{st['n']}  turns {st['turns']}  refused {st['refused']}")
    a = report["summary"]["all"]
    print(f"{model.name}: {a['passed']}/{a['n']} passed ({a['rate']:.0%}), {a['tokens']} tokens, {a['seconds']} s -> {path}")
    return 0


def export_rows(args: argparse.Namespace) -> int:
    from nts.data import episodes, export

    eps = episodes(args.telemetry / "spans")
    if args.stores:  # only episodes that started on a version of these stores (keeps a held-out department out)
        versions = {e.hash for s in args.stores for e in Store.open(s).events}
        eps = (e for e in eps if e.episode.attrs.get("state_version") in versions
               or any(v.attrs.get("store_version") in versions for v in e.evaluations))  # a scenario's setup came first
    if args.agent:  # only episodes offered exactly this agent's tools (its prompt and tools are what is trained)
        want = tools_hash(Registry(ALL_TOOLS).specs(AGENTS[args.agent].scopes))
        eps = (e for e in eps if tools_hash(e.tools) == want)
    counts = export(eps, args.out, models=args.models, include_unjudged=args.include_unjudged,
                    keep_errors=args.keep_errors)
    print(json.dumps(counts))
    return 0


def ask(args: argparse.Namespace) -> int:
    model = make_model(args)
    store = Store.open(args.store)
    actor = OFFICE
    if args.actor.lower() not in ("office", OFFICE):
        try:
            actor = Directory(store.world.instance).teacher(args.actor).id
        except Refusal as r:
            print(f"--as: {r.error.message} {r.error.hint}", file=sys.stderr)
            return 2
    tracer = default_tracer(args.telemetry)
    harness = Harness(Registry(ALL_TOOLS), model, tracer, handbook=handbook(args))
    people = None
    if args.talk:
        from nts.people import Terminal

        people = Terminal()
        if args.best_of > 1:
            print("--best-of with --talk would ask you once per run; use one or the other", file=sys.stderr)
            return 2
    out = best_of(harness, args.best_of, budgeted(AGENTS[args.agent], args), args.text, store, actor, args.week,
                  people=people)
    if args.show:
        for m in out.transcript[2:]:
            for c in m.tool_calls:
                print(f"  -> {c.name} {json.dumps(c.arguments, ensure_ascii=False)}")
            if m.role == "tool":
                print(f"  <- {m.content[:400]}")
    print(f"[{out.status}] {out.text or out.detail}")
    if out.pending:
        print(f"  {out.pending.kind} to {out.pending.to_name}: {out.pending.text}  (--talk to answer it)")
    print(f"episode {out.episode_id}: {out.turns} turns, {out.tool_calls} calls ({out.refused} refused), "
          f"{out.usage.prompt_tokens}+{out.usage.completion_tokens} tokens, {out.seconds:.1f} s")
    if out.status == "submitted" and args.approve:
        if approve(store, out, tracer) is None:
            print("not committed: the kernel refused it (see the approval span)", file=sys.stderr)
            return 1
        print(f"committed: version {store.version}")
    return 0


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from nts.api import Service, Users, create_app

    model, tracer, rules = make_model(args), default_tracer(args.telemetry), handbook(args)
    depts = {path.stem: Service(Store.open(path), model, tracer, agent=budgeted(AGENTS[args.agent], args),
                                simulate=args.simulate, handbook=rules) for path in args.store}
    for d, svc in depts.items():
        print(f"department {d}: {len(svc.store.world.instance.faculty)} teachers, version {svc.store.version}")
    users = Users.load(args.users) if args.users else Users.open()
    if not users.needs_password:
        print("sign-in by name alone (no --users): fine on this machine, not on a network", file=sys.stderr)
    print(f"{model.name}, agent {args.agent}: http://{args.host}:{args.port}/")
    uvicorn.run(create_app(depts, users, telemetry=args.telemetry), host=args.host, port=args.port, log_level="warning")
    return 0


def passwd(args: argparse.Namespace) -> int:
    import getpass

    from nts.api import Users

    users = Users.load(args.users)
    first = getpass.getpass(f"password for {args.name} in {args.dept}: ")
    if len(first) < 8 or first != getpass.getpass("again: "):
        print("not set: the two differ, or it is shorter than 8 characters", file=sys.stderr)
        return 1
    users.set(args.dept, "office" if args.name.lower() == "office" else args.name, first)
    users.save()
    print(f"set in {args.users}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
