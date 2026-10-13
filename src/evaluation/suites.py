"""Score an intent reader on every evaluation suite and keep the result, so each fine-tuning run is compared
with the ones before it on the same requests.

    uv run python -m evaluation.suites --parser rule
    uv run python -m evaluation.suites --parser local --local URL --local-model Qwen3.5-4B-...-Intents-v3

The suites are the frozen test split (``data/compiler-intents/test.jsonl``) and every file in
``data/suites`` that ``training.dataset`` wrote: held-out wordings of each generator (``firmness``,
``moves``, ``periods``, ``multi``, ``clubs``), the test split rewritten with typos and noise
(``test-typo``, ``test-noisy``) and, when people have labelled messages, ``human``. Each is scored as
``evaluation.intents`` scores (outcome match, silently wrong, asked instead). One line per run is appended
to ``runs/leaderboard.jsonl`` with the model, the data manifest's fingerprints (which suites it was scored
on, exactly) and every suite's summary; ``--show`` prints the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.instance import Instance
from language.adapters import add_args, local_client, registry

from .intents import Reader, score

LEADERBOARD = Path("runs/leaderboard.jsonl")
COLUMNS = ("outcome_match", "silent_wrong", "asked_instead")


def suites(data: Path, extra: Path) -> dict[str, Path]:
    found = {"test": data / "test.jsonl"}
    found.update({p.stem: p for p in sorted(extra.glob("*.jsonl"))})
    return {k: v for k, v in found.items() if v.exists()}


def show(path: Path = LEADERBOARD) -> None:
    if not path.exists():
        print(f"no runs yet ({path})")
        return
    runs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    names = sorted({s for r in runs for s in r["suites"]}, key=lambda s: (s != "test", s))
    print(f"{'parser':48} " + " ".join(f"{n[:11]:>11}" for n in names) + "   (outcome match / silently wrong)")
    for r in runs:
        cells = [f"{r['suites'][n]['outcome_match']:>5.1%}/{r['suites'][n]['silent_wrong']:<5.1%}"
                 if n in r["suites"] else f"{'':>11}" for n in names]
        print(f"{(r['parser'] + (' ' + r['note'] if r.get('note') else ''))[:48]:48} " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", choices=["gold", "rule", "local", "gemini"], default="rule")
    ap.add_argument("--data", type=Path, default=Path("data/compiler-intents"))
    ap.add_argument("--suites", type=Path, default=Path("data/suites"))
    ap.add_argument("--only", default=None, help="comma-separated suite names")
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--local", metavar="URL", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--note", default="", help="a word on this run for the leaderboard")
    ap.add_argument("--show", action="store_true", help="print the leaderboard and stop")
    add_args(ap)
    args = ap.parse_args()
    if args.show:
        show()
        return

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    client = None
    if args.parser == "local":
        client = local_client("compile", base_url=args.local, schema=args.schema, server_model=args.local_model, adapter=args.adapter,
                              reg=registry(args.adapters))
    elif args.parser == "gemini":
        from language.llm import GeminiClient

        client = GeminiClient(args.model)
    reader = Reader(args.parser, instance, client)
    name = args.parser if client is None else f"{args.parser}-{client.model.split(':')[-1].removesuffix('.gguf')}"
    name += "" if not args.no_checks else " (no checks)"
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    wanted = set(args.only.split(",")) if args.only else None
    result = {"parser": name, "note": args.note, "at": time.strftime("%Y-%m-%d %H:%M"),
              "data": {"recipe": manifest.get("recipe"), "code": manifest.get("code")}, "suites": {}}
    print(f"{name}")
    print(f"  {'suite':12} {'n':>5} {'match':>7} {'silent':>7} {'asked':>7}")
    for suite, path in suites(args.data, args.suites).items():
        if wanted and suite not in wanted:
            continue
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        s = score(rows, instance, reader, args.workers, checks=not args.no_checks)["summary"]
        result["suites"][suite] = {k: s[k] for k in ("n", *COLUMNS, "action_accuracy", "invalid")} | {
            "fingerprint": manifest.get("fingerprints", {}).get(suite)}
        print(f"  {suite:12} {s['n']:>5} {s['outcome_match']:>7.1%} {s['silent_wrong']:>7.1%} "
              f"{s['asked_instead']:>7.1%}", flush=True)
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    with LEADERBOARD.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"appended to {LEADERBOARD}")


if __name__ == "__main__":
    main()
