"""How a parser holds up when the test requests are written the way people write: typos, chat shorthand and
casual messages.

Each test request is rewritten deterministically (seeded by its ID) at one or more levels, keeping what it
asks for, and scored against the same gold as ``evaluation.intents``:

* ``typo``: one to three slips in ordinary words and day names (a letter dropped, doubled or swapped);
  names, course titles and numbers are left alone;
* ``shorthand``: "b4", "pls", "u", "thurs", "weds", "thx", "cant", "lec";
* ``casual``: lower case, no greeting or sign-off, no full stop at the end;
* ``noisy``: all three.

Besides the usual scores it reports ``silently wrong`` (acted on a wrong reading) apart from ``asked``
(a question instead of the answer), and ``consistent``: how often the reading of a rewritten request resolves
to what the reading of the original does (needs no gold).

    uv run python -m evaluation.robustness --parser rule
    uv run python -m evaluation.robustness --parser local --local-model Qwen3.5-4B-Finetune-NTS-Intents-Q8_0
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from core.instance import Instance
from language.adapters import add_args, local_client, registry

from .intents import Reader, score

LEVELS = ("typo", "shorthand", "casual", "noisy")
_DAY_NAMES = re.compile(r"^(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?$")
SHORTHAND = [(r"\bbefore\b", "b4"), (r"\bplease\b", "pls"), (r"\byou\b", "u"), (r"\bThursdays?\b", "thurs"),
             (r"\bWednesdays?\b", "weds"), (r"\bTuesdays?\b", "tues"), (r"\bthank you\b", "thx"), (r"\bthanks\b", "thx"),
             (r"\bcan't\b", "cant"), (r"\bI am\b", "im"), (r"\blecture\b", "lec"), (r"\bdon't\b", "dont"),
             (r"\btomorrow\b", "tmrw")]
_GREETING = re.compile(r"^\s*(dear [^,\n]*,|hello,?|hi,?|hey,?)\s*", re.IGNORECASE)
_SIGN_OFF = re.compile(r"\s*(regards|best|thanks|thank you|cheers)[,!.]?\s*(\n.*)?$", re.IGNORECASE | re.DOTALL)


def typo(text: str, rng: random.Random, protect: set[str]) -> str:
    """One to three slips in ordinary words (four letters or more, lower case, or a day name)."""
    tokens = re.split(r"(\W+)", text)
    spots = [i for i, t in enumerate(tokens) if len(t) >= 4 and t.isalpha() and t.lower() not in protect
             and (t.islower() or _DAY_NAMES.match(t))]
    for i in rng.sample(spots, min(len(spots), rng.randint(1, 3))):
        w = tokens[i]
        j = rng.randrange(1, len(w) - 1)
        op = rng.choice(("drop", "double", "swap"))
        tokens[i] = (w[:j] + w[j + 1:] if op == "drop" else w[:j] + w[j] + w[j:] if op == "double"
                     else w[:j] + w[j + 1] + w[j] + w[j + 2:])
    return "".join(tokens)


def shorthand(text: str, rng: random.Random) -> str:
    for pat, short in SHORTHAND:
        if rng.random() < 0.8:
            text = re.sub(pat, short, text, flags=re.IGNORECASE)
    return text


def casual(text: str) -> str:
    text = _SIGN_OFF.sub("", _GREETING.sub("", text.strip()))
    return re.sub(r"\s+", " ", text).strip().rstrip(".").lower()


def rewrite(text: str, level: str, seed: str, protect: set[str]) -> str:
    rng = random.Random(f"{seed}:{level}")
    if level in ("typo", "noisy"):
        text = typo(text, rng, protect)
    if level in ("shorthand", "noisy"):
        text = shorthand(text, rng)
    if level in ("casual", "noisy"):
        text = casual(text)
    return text


def protected(instance: Instance) -> set[str]:
    """Words a slip would turn into a different name: course titles, people, rooms, clubs."""
    names = [*instance.course_titles.values(), *(f.name for f in instance.faculty), *(r.name for r in instance.rooms),
             *(g.name for g in instance.groups)]
    return {w.lower() for n in names for w in re.findall(r"[A-Za-z]+", n)}


def rewrite_row(row: dict, level: str, seed: str, protect: set[str]) -> dict:
    """``row`` (the intent data's row format) with its message rewritten at ``level``: in the request and in
    the prompt the model reads. The gold is left as it is."""
    r = copy.deepcopy(row)
    old = r["request"]["raw_text"]
    new = rewrite(old, level, seed, protect)
    r["request"]["raw_text"] = new
    user = r["messages"][1]["content"]
    r["messages"][1]["content"] = user.replace(f"<message>\n{old}\n</message>", f"<message>\n{new}\n</message>")
    assert r["messages"][1]["content"] != user or new == old, r["id"]
    return r


def rewritten(rows: list[dict], level: str, instance: Instance) -> list[dict]:
    keep = protected(instance)
    return [rewrite_row(row, level, row["id"], keep) for row in rows]


def _outcome_key(row: dict) -> str:
    return json.dumps(row.get("pred_outcome"), sort_keys=True, default=str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", choices=["rule", "local", "gemini"], default="rule")
    ap.add_argument("--data", type=Path, default=Path("data/compiler-intents"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--local", metavar="URL", default="http://localhost:8080/v1")
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--legacy-rules", action="store_true", help="the rule parser as the corpus labels were made")
    add_args(ap)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.data / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()][: args.limit]
    client = None
    if args.parser == "local":
        client = local_client("compile", base_url=args.local, schema=args.schema, server_model=args.local_model, adapter=args.adapter,
                              reg=registry(args.adapters))
    elif args.parser == "gemini":
        from language.llm import GeminiClient

        client = GeminiClient(args.model)
    reader = Reader(args.parser, instance, client)
    if args.legacy_rules:
        from language.rule_parser import RuleParser

        reader.rules = RuleParser(instance, robust=False)
    checks = not args.no_checks
    name = args.parser if client is None else f"{args.parser}-{client.model.split(':')[-1].removesuffix('.gguf')}"
    name += "-legacy" if args.legacy_rules else ""
    name += "" if checks else "-nochecks"

    report = {"parser": name, "levels": {}}
    clean = score(rows, instance, reader, args.workers, checks=checks)
    report["levels"]["clean"] = clean["summary"]
    base = {r["id"]: (r["match"], r["pred_action"], json.dumps(r["pred"], sort_keys=True)) for r in clean["rows"]}
    print(f"{name} on {len(rows)} test requests")
    print(f"  {'level':10} {'match':>7} {'action':>7} {'silent':>7} {'asked':>7} {'consistent':>11}")

    def show(level: str, s: dict, consistent: float | None) -> None:
        c = f"{consistent:>10.1%}" if consistent is not None else f"{'':>10}"
        print(f"  {level:10} {s['outcome_match']:>7.1%} {s['action_accuracy']:>7.1%} {s['silent_wrong']:>7.1%} "
              f"{s['asked_instead']:>7.1%} {c}")

    show("clean", clean["summary"], None)
    for level in [x for x in args.levels.split(",") if x]:
        got = score(rewritten(rows, level, instance), instance, reader, args.workers, checks=checks)
        # consistent: the same result as on the original wording (both right, or the same reading)
        same = sum(r["match"] == base[r["id"]][0] and (r["match"] or r["pred_action"] == base[r["id"]][1])
                   for r in got["rows"]) / len(got["rows"])
        report["levels"][level] = {**got["summary"], "consistent": round(same, 3)}
        report.setdefault("examples", {})[level] = [
            {"id": r["id"], "text": r["text"], "pred": r["pred"], "notes": r["notes"]}
            for r in got["rows"] if not r["match"] and base[r["id"]][0]][:15]
        show(level, got["summary"], same)
    out = args.out or Path("runs") / f"robustness-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
