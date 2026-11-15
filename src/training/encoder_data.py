"""Training data for the encoder intent reader (docs/encoder-intents.md), balanced over how complex a message is.

    python -m training.encoder_data --recipe data/recipes/encoder-intents-v1.toml

Every row is in the intents format (``intent_compiler.export_rows``) with a ``tier``:

* ``non_compile``: clarify, investigate, answer, out_of_scope, withdraw;
* ``single``: one intent;
* ``multi``: two or three independent intents;
* ``nested``: a choice between items (any_of, ranked, fallback), alone or beside other requests;
* ``linked``: an item placed against another class (relate) or naming another item's class (``same``).

The intents-v13 train split is 79% ``single`` and 2.5% ``nested``. Here ``single`` rows come from that split,
water-filled over intent kinds and their generators (the commonest kinds trimmed, the rarest repeated at most
``max_up`` times), and the complex tiers are made: one message joins the wordings of the single-intent
generators (``PARTS``), so a kind reads the same inside a longer message as alone (``training.balance``: parts
made by a few combining generators teach their habits as the kind's reading). Each part must resolve on its
own (``intent_corpus._resolves`` with its own ``want``) and the whole message must compile without a question
or an error. Training rows use only the generators' training wordings and ``CONNECT``'s training connectors;
the held-out ones make the ``composed`` suite, and the frozen test split and the suites are left as they are.
"""

from __future__ import annotations

import argparse
import json
import random
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from core.instance import Instance
from core.schemas import Channel, Request, Role
from language import intent_corpus as ic
from language import intent_generators as ig
from language.intent_compiler import export_rows
from language.intents import Avoid, IntentOutput
from training import quality

TIERS = ("non_compile", "single", "multi", "nested", "linked")
CHOICES = ("any_of", "ranked", "fallback")


def tier(gold: dict) -> str:
    if gold.get("action") != "compile":
        return "non_compile"
    its = gold.get("intents") or []
    if any(i["kind"] in CHOICES for i in its):
        return "nested"
    if any(i["kind"] == "relate" or (isinstance(i.get("which"), dict) and i["which"].get("same")) for i in its):
        return "linked"
    return "single" if len(its) == 1 else "multi"


def gold_of(row: dict) -> dict:
    return json.loads(row["messages"][-1]["content"])


def label(row: dict) -> str:
    """What a row is balanced on: its one intent's kind, or its action."""
    g = gold_of(row)
    its = g.get("intents") or []
    return its[0]["kind"] if g["action"] == "compile" and len(its) == 1 else g["action"]


# --- water-filling ---------------------------------------------------------------------------------------

def quotas(sizes: dict[str, int], total: int, max_up: float) -> dict[str, int]:
    """An equal share of ``total`` per group; a group short of its share gives all it has (up to ``max_up``
    times its size) and the rest is shared among the others."""
    q, open_ = {}, dict(sizes)
    left = total
    while open_:
        share = left / len(open_)
        small = {k: n for k, n in open_.items() if n * max_up <= share}
        if not small:
            base = int(share)
            extra = left - base * len(open_)
            for i, k in enumerate(sorted(open_)):
                q[k] = base + (1 if i < extra else 0)
            break
        for k, n in small.items():
            q[k] = int(n * max_up)
            left -= q[k]
            del open_[k]
    return q


def fill(groups: dict[str, list], total: int, max_up: float, rng: random.Random,
         inner: Callable[[dict], str] | None = None) -> list:
    """``total`` items spread over ``groups`` by ``quotas``; within a group, spread again over ``inner``'s
    keys when given. Items beyond a group's size are repeats, marked ``repeat``."""
    out = []
    for k, n in quotas({k: len(v) for k, v in groups.items()}, total, max_up).items():
        items = groups[k]
        if inner is not None:
            sub = defaultdict(list)
            for r in items:
                sub[inner(r)].append(r)
            out += fill(sub, n, max_up, rng)
            continue
        rng.shuffle(items)
        take = items[:n]
        while len(take) < n:
            take.append({**rng.choice(items), "repeat": True})
        out += take
    return out


# --- composed messages -----------------------------------------------------------------------------------

# the single-intent generators a part comes from, by the kind of intent asked for (a generator that can make
# several kinds is listed under each; a part counts only if it made the kind it was asked for)
PARTS: dict[str, list[str]] = {
    "move": ["move", "which_to", "move_pairs", "same_day_part"],
    "cancel": ["cancel"],
    "extra_class": ["extra_class"],
    "swap": ["swap"],
    "avoid": ["avoid_between", "firmness", "difficulty", "relative_span", "term_span", "week_patterns", "out_for_me"],
    "unavailable": ["unavailable_relative", "dated", "relative_span", "term_span", "week_patterns", "out_for_me"],
    "only": ["only_firmness", "reason_firmness", "only_times"],
    "load_limit": ["load_limit", "limits", "limit_firmness"],
    "limit": ["limits", "limit_firmness"],
}
RELATE = ["relations", "relation_pairs"]
ONCE = {"only", "swap"}  # two of these in one message would contradict each other or not resolve
# kinds not added beside a shape's own part: a choice is already two moves (or a move and a cancellation)
SKIP = {"nested": {"move"}}

_CORPUS_HELD = (ic.TEST_TEMPLATE,)
_REG = {g.name: g for g in ig.REGISTRY}


def withdraw_more(ctx: ic._Ctx, t: int):
    """More ways of taking a request back (encoder-intents-v2): v1's reader learned withdraw from four
    wordings and read the frozen test's fifth, held-out one ("Never mind my message from yesterday ...") as a
    question. None of these is that wording; they are other ways people say it."""
    D = ic.FULL_DAY[ctx.rng.choice(ctx.instance.calendar.days)]
    texts = ["Please disregard my last email; nothing needs to change.",
             "Forget what I asked for earlier, my timetable is fine.",
             f"I'm taking back my request from {D}; no change is needed.",
             f"You can drop my earlier request about {D}. It's no longer needed.",
             "My previous message can be ignored, please keep my schedule unchanged.",
             "Ignore my request from last week, things have worked out.",
             "No need to act on my earlier email any more.",
             f"Please don't go ahead with the change I asked for on {D}.",
             "I no longer need the change I requested yesterday.",
             "Disregard my message from this morning; the current schedule works for me.",
             f"Withdraw my request about my {D} class, please; it has been sorted out.",
             "The change I asked for is not needed any more."]
    return texts[t], IntentOutput(action="withdraw"), {}, False


EXTRA = {"withdraw_more": (withdraw_more, 12)}  # encoder-only generators: name -> (function, wordings)


def _gen(name: str):
    if name in EXTRA:
        return EXTRA[name][0]
    return _REG[name].fn if name in _REG else ic.GENERATORS[name]


def _templates(name: str, held: bool) -> tuple[int, ...]:
    if name in EXTRA:  # training wordings only: nothing of these is held out
        return () if held else tuple(range(EXTRA[name][1]))
    if name in _REG:
        g = _REG[name]
        return g.held_out if held else g.train_templates
    return _CORPUS_HELD if held else tuple(t for t in range(5) if t not in _CORPUS_HELD)


# connectors: a function of the parts' sentences; the last ones of each list are held out for the suite
def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


LOWER = {"please", "could", "can", "would", "no", "my", "keep", "move", "cancel", "ideally", "nothing", "avoid",
         "if", "don't", "do", "the", "for", "from", "starting", "every", "on", "in", "before", "after", "there",
         "it", "this", "next", "any", "only", "make", "add", "schedule", "put", "swap", "let", "kindly", "try",
         "first", "either", "is", "what", "when", "none", "not", "at", "until", "for", "during", "most",
         "between", "never", "an", "a", "one", "two", "three", "four", "five", "i'm", "me"}


def _low(s: str) -> str:
    w = s.split(" ", 1)[0].strip(",.;:!?").lower()
    return s[:1].lower() + s[1:] if w in LOWER and w not in ("i", "i'm") else s


def _end(s: str) -> str:
    s = s.strip()
    return s if s[-1:] in ".?!" else s + "."


def _bare(s: str) -> str:
    return s.strip().rstrip(".!")


TWO: list[Callable[[str, str], str]] = [
    lambda a, b: f"{_end(a)} Also, {_low(_end(b))}",
    lambda a, b: f"Two things: {_low(_bare(a))}; {_low(_end(b))}",
    lambda a, b: f"{_end(a)} Separately, {_low(_end(b))}",
    lambda a, b: f"{_end(a)} {_cap(_end(b))}",
    lambda a, b: f"{_end(a)} One more thing: {_low(_end(b))}",
    lambda a, b: f"{_end(a)}\n{_cap(_end(b))}",
    # held out
    lambda a, b: f"{_end(a)} Besides that, {_low(_end(b))}",
    lambda a, b: f"A couple of requests. {_end(a)} And {_low(_end(b))}",
]
THREE: list[Callable[[str, str, str], str]] = [
    lambda a, b, c: f"A few requests. {_end(a)} {_cap(_end(b))} {_cap(_end(c))}",
    lambda a, b, c: f"{_end(a)} Also, {_low(_end(b))} And finally, {_low(_end(c))}",
    lambda a, b, c: f"Three things: {_low(_bare(a))}; {_low(_bare(b))}; {_low(_end(c))}",
    lambda a, b, c: f"1. {_end(a)}\n2. {_end(b)}\n3. {_end(c)}",
    # held out
    lambda a, b, c: f"{_end(a)} In addition, {_low(_end(b))} Lastly, {_low(_end(c))}",
]
HELD_TWO, HELD_THREE = 2, 1
# joins by semicolons: only for parts with no semicolon of their own that end in a full stop, or the join
# blurs which part a "; this is a must" belongs to
SEMI = {2: {1}, 3: {2}}


def _part(ctx: ic._Ctx, name: str, held: bool, want_kind: str | None):
    """One generator's message, if it is one intent (of ``want_kind``) that resolves on its own."""
    got = _gen(name)(ctx, ctx.rng.choice(_templates(name, held)))
    if got is None:
        return None
    text, gold, want, asks = got
    if asks or gold.action != "compile" or not gold.intents:
        return None
    if want_kind is not None and (len(gold.intents) != 1 or gold.intents[0].kind != want_kind):
        return None
    req = Request(id="part", channel=Channel.PORTAL, sender_id=ctx.sender, role=Role.FACULTY, raw_text=text,
                  received_at=datetime(2026, 9, 1))
    ex = ic.IntentExample("part", name, 0, "train", req, gold, ctx.timetable, ctx.week, False, want, ctx.today)
    return (text, gold, want) if ic._resolves(ex, ctx.instance) else None


def composed(shape: str, held: bool, three: float) -> ic.Gen:
    """A generator of messages of several requests. ``shape``: ``multi`` (independent intents), ``nested``
    (a ``choices`` message beside one or two others) or ``linked`` (a relation beside one or two others)."""

    def gen(ctx: ic._Ctx, t: int):
        rng = ctx.rng
        n = 3 if rng.random() < three else 2
        parts = []
        if shape == "nested":
            p = _part(ctx, "choices", held, None)
            # choices' move-and-relation wording has no choice in it: a linked message, not a nested one
            parts.append(p if p and any(i.kind in CHOICES for i in p[1].intents) else None)
        elif shape == "linked":
            parts.append(_part(ctx, rng.choice(RELATE), held, "relate"))
        if parts and parts[0] is None:
            return None
        kinds = [k for k in rng.sample(sorted(PARTS), len(PARTS)) if k not in SKIP.get(shape, ())]
        for k in kinds:
            if len(parts) == n:
                break
            p = _part(ctx, rng.choice(PARTS[k]), held, k)
            if p is not None:
                parts.append(p)
        if len(parts) < n:
            return None
        rng.shuffle(parts)
        intents = [i for _, g, _ in parts for i in g.intents]
        if sum(i.kind in ONCE for i in intents) > 1:
            return None
        # a wish beside a must in one message: whether the wish's hedge softens both is not settled (``multi``)
        if any(isinstance(i, Avoid) and not i.hard for i in intents) and any(getattr(i, "hard", False) for i in intents):
            return None
        targets = [json.dumps(v, sort_keys=True) for _, _, w in parts for v in w.values()]
        if len(targets) != len(set(targets)):  # two parts about the same class
            return None
        texts = [p[0] for p in parts]
        bank = TWO if n == 2 else THREE
        cut = len(bank) - (HELD_TWO if n == 2 else HELD_THREE)
        ok = range(cut, len(bank)) if held else range(cut)
        if any(";" in x or _end(x)[-1] != "." for x in texts[:-1]):
            ok = [i for i in ok if i not in SEMI[n]]
        join = bank[rng.choice(list(ok))]
        return join(*texts), IntentOutput(action="compile", intents=intents), {}, False

    return gen


def make(instance: Instance, shape: str, n: int, *, held: bool, three: float, seed: int,
         tag: str) -> list[ic.IntentExample]:
    name = f"{tag}-composed_{shape}"
    exs = ic._examples(instance, random.Random(seed), {name: (name, composed(shape, held, three))}, n, None,
                       templates=(1 if held else 0,))
    for e in exs:
        e.split = "composed" if held else "train"
    return exs


def fresh(instance: Instance, names: list[str], n: int, seed: int, tag: str) -> list[ic.IntentExample]:
    """Rows of ``names`` (training wordings only) with a seed of their own."""
    out = []
    for i, name in enumerate(names):
        exs = ic._examples(instance, random.Random(seed + i), {f"{tag}-{name}": (name, _gen(name))}, n, None,
                           templates=_templates(name, False))
        for e in exs:
            e.split = "train"
        out += exs
    return out


# --- the build -------------------------------------------------------------------------------------------

def rows_of(instance: Instance, exs: list[ic.IntentExample]) -> list[dict]:
    out = []
    for _, r in export_rows(instance, [], exs):
        r["tier"] = tier(gold_of(r))
        out.append(r)
    return out


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["tier"] = tier(gold_of(r))
    return rows


def split_rows(instance: Instance, pool: list[dict], size: int, mix: dict, spec: dict, seed: int,
               seen: set[str], tag: str) -> tuple[list[dict], dict]:
    """One split of ``size`` rows in ``mix``'s shares: singles and non-compile rows from ``pool`` (topped up
    with fresh generator rows), the complex tiers from ``pool`` plus composed messages."""
    rng = random.Random(seed)
    want = {t: round(size * mix[t]) for t in TIERS}
    by_tier = defaultdict(list)
    for r in pool:
        by_tier[r["tier"]].append(r)

    def new(rows: list[dict]) -> list[dict]:
        keep = [r for r in rows if r["request"]["raw_text"] not in seen]
        seen.update(r["request"]["raw_text"] for r in keep)
        return keep

    made = {}
    # non-compile: what the pool has, the generated kinds topped up so no action is repeated more than twice
    nc = by_tier["non_compile"]
    if len(nc) * 1.0 < want["non_compile"]:
        names = ["vague", "withdraw", "clash_report"] + list(spec.get("extra_non_compile", []))
        extra = new(rows_of(instance, fresh(instance, names, (want["non_compile"] - len(nc)) // 3 + 1,
                                           seed + 100, tag)))
        made["non_compile"] = len(extra)
        nc = nc + extra
    groups = defaultdict(list)
    for r in nc:
        groups[label(r)].append(r)
    out = fill(groups, want["non_compile"], spec["max_up"], rng)

    groups = defaultdict(list)
    for r in by_tier["single"]:
        groups[label(r)].append(r)
    out += fill(groups, want["single"], spec["max_up"], rng, inner=lambda r: r["kind"])

    for j, shape in enumerate(("multi", "nested", "linked")):
        have = by_tier[shape]
        if shape == "nested":  # choices on their own: as many again from a fresh seed
            have = have + new(rows_of(instance, fresh(instance, ["choices"], max(len(have), 1), seed + 200, tag)))
        need = want[shape] - min(len(have), want[shape] // 3)  # pool rows at most a third of the tier
        comp = new(rows_of(instance, make(instance, shape, need, held=False, three=spec["three"],
                                          seed=seed + 300 + j, tag=tag)))
        made[shape] = len(comp)
        rng.shuffle(have)
        out += have[:want[shape] - len(comp)] + comp
    rng.shuffle(out)
    return out, made


def report(rows: list[dict]) -> dict:
    n = len(rows)
    tiers = Counter(r["tier"] for r in rows)
    kinds = Counter(i["kind"] for r in rows for i in (gold_of(r).get("intents") or []))
    actions = Counter(gold_of(r)["action"] for r in rows)
    n_int = Counter(len(gold_of(r).get("intents") or []) for r in rows if gold_of(r)["action"] == "compile")
    return {"rows": n, "repeats": sum(1 for r in rows if r.get("repeat")),
            "tiers": {t: f"{tiers[t]} ({tiers[t] / max(n, 1):.0%})" for t in TIERS},
            "actions": dict(actions.most_common()), "intents_per_message": dict(sorted(n_int.items())),
            "intent_kinds": dict(kinds.most_common()),
            "sources": dict(Counter(r["kind"] for r in rows).most_common(12))}


def build(recipe: dict) -> dict:
    instance = Instance.model_validate_json(Path(recipe["instance"]).read_text(encoding="utf-8"))
    src, out = Path(recipe["source"]), Path(recipe["out"])
    mix, seed = recipe["mix"], recipe["seed"]
    assert abs(sum(mix.values()) - 1) < 1e-6, "mix must sum to 1"
    spec = {"max_up": recipe["single"]["max_up"], "three": recipe["multi"]["three_parts"],
            "extra_non_compile": recipe.get("non_compile", {}).get("extra", [])}
    size = round(recipe["single"]["total"] / mix["single"])

    train_pool, val_pool = load(src / "train.jsonl"), load(src / "val.jsonl")
    held = {json.loads(line)["request"]["raw_text"]
            for p in [src / "test.jsonl", *Path("data/suites").glob("*.jsonl")]
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}
    seen = held | {r["request"]["raw_text"] for r in train_pool + val_pool}

    train, made_t = split_rows(instance, train_pool, size, mix, spec, seed, seen, "enc")
    val, made_v = split_rows(instance, val_pool, round(size * recipe["val"]["fraction"]), mix, spec, seed + 1, seen, "encval")
    suite = []
    for shape in ("multi", "nested", "linked"):
        exs = make(instance, shape, recipe["suite"][shape], held=True, three=spec["three"], seed=seed + 500,
                   tag="suite")
        suite += [r for r in rows_of(instance, exs) if r["request"]["raw_text"] not in seen]
    # the repo's gates (training.quality): a joined message fails the firmness rubric when one part's hedge reads
    # as the whole message's ("..., if possible. Also, I can only teach afternoons" read as a wish); dropped
    splits = {"train": train, "val": val, "composed": suite}
    for r in train + val + suite:
        r.setdefault("origin", "encoder")
    fails = quality.check(instance, splits, held=["composed"])
    bad = {(f["split"], f["id"]) for fs in fails.values() for f in fs}
    dropped = Counter(f"{s}:{tier}" for s, rows in splits.items() for r in rows
                      if (s, r["id"]) in bad for tier in [r["tier"]])
    train, val, suite = ([r for r in rows if (s, r["id"]) not in bad] for s, rows in splits.items())
    leaks = {r["request"]["raw_text"] for r in train + val} & {r["request"]["raw_text"] for r in suite}

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("composed", suite)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"recipe": recipe["name"], "made": {"train": made_t, "val": made_v}, "leaks": len(leaks),
                "gates": {g: len(f) for g, f in fails.items()}, "dropped": dict(dropped),
                "before": {"train": report(train_pool)},
                "after": {"train": report(train), "val": report(val), "composed": report(suite)}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", type=Path, default=Path("data/recipes/encoder-intents-v1.toml"))
    args = ap.parse_args()
    m = build(tomllib.loads(args.recipe.read_text(encoding="utf-8")))
    print(json.dumps({k: m[k] for k in ("made", "leaks", "gates", "dropped")}))
    for name, rep in [("before train", m["before"]["train"])] + [(k, v) for k, v in m["after"].items()]:
        print(f"\n== {name}: {rep['rows']} rows, {rep['repeats']} repeats")
        print(" tiers  ", rep["tiers"])
        print(" per msg", rep["intents_per_message"])
        print(" kinds  ", rep["intent_kinds"])


if __name__ == "__main__":
    main()
