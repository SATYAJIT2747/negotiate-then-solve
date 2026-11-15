"""Score an encoder reader's predictions the way ``evaluation.intents`` scores the LLM: by what they do.

    uv run python -m evaluation.encoder_outcome --preds ../kaggle-results/nts-encoder-v1/predictions
    uv run python -m evaluation.encoder_outcome --roundtrip        # the gold spans decoded: the ceiling

An encoder prediction (``kaggle/encoder/nts_encoder.py``) is spans of the message, each with a role
(``which.day``, ``to.at``, ``period.weeks``), chosen-from-a-set values (``hard``, ``to.part``) and refs
("the same class as the request before"), plus the message's action and composites (``any_of`` ...). The
normaliser here (deterministic, no model) turns that back into an ``IntentOutput``:

* span text -> typed value with the extractors the spans were labelled with (``training.encoder_spans``):
  days (typos too), clock times (bare hours in the teaching day), numbers (words too), kinds, dates;
  course names -> the course's title (abbreviations, typos); faculty by surname; rooms by name;
* a range said by its ends ("Wednesday through Friday", "weeks 9-11") -> every day or week inside it;
* refs -> ``which: {same: <id>}`` of the request before; each composite -> the next chain of requests
  joined by refs (the only way the training data writes them).

A span that does not normalise is dropped, so the reading is scored as it would act. Rows are then compiled
and compared with ``evaluation.intents.score`` against the frozen test split, the suites and the encoder's
own val/composed splits. With ``--llm`` (a suites misses file of an LLM run) it also reports the hybrid: the
encoder's reading where its gate accepts it, the LLM's elsewhere.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

from core.instance import Instance
from language.intents import IntentOutput
from training.encoder_spans import (DAYS, RANGE, Strings, dates, days, kinds, numbers, times, words,
                                    _clock)

from .intents import Reader, score

ORDER = list(DAYS)
LIST_FIELDS = {"time.days", "to.days", "time.at", "to.at", "period.weeks", "period.except_weeks", "courses",
               "cohorts", "equipment"}
INT_ROLES = {"period.weeks", "period.from_week", "period.to_week", "period.except_weeks", "n", "max_consecutive",
             "attendees", "days"}
TIME_SUFFIX = ("at", "after", "before", "time")
COMPOSITES = ("any_of", "ranked", "fallback")
# "Wednesday through Friday", "weeks 9 to 11": a range by its ends ("and" only after "between")
SPAN_RANGE = re.compile(r"^\s*(?:-|–|to|through|thru|till|until)\s*(?:weeks?\s*)?$", re.IGNORECASE)


class Normaliser:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.strings = Strings(instance)
        self.titles = sorted(set(instance.course_titles.values()))
        self.faculty = [f.name for f in instance.faculty]
        self.groups = [g.name for g in instance.groups]
        self.rooms = [r.name for r in instance.rooms]
        self.equipment = sorted({e for r in instance.rooms for e in r.equipment})

    # --- one span -> one value --------------------------------------------------------------------------

    def day(self, t: str) -> str | None:
        got = days(t)
        return got[0][2] if got else None

    def clock(self, t: str) -> str | None:
        got = times(t)
        if got:
            return got[0][2]
        m = re.search(r"\d{1,2}", t)
        if m:
            return _clock(int(m[0]), 0, None)
        if re.search(r"noon|midday", t, re.IGNORECASE):
            return "12:00"
        return None

    def int(self, t: str) -> int | None:
        got = numbers(t, [])
        return got[0][2] if got else None

    def kind(self, t: str) -> str | None:
        got = kinds(t)
        return got[0][2] if got else None

    def date(self, t: str) -> str | None:
        got = dates(t)
        return got[0][2] if got else None

    def _best(self, t: str, names: list[str], cut: float) -> str | None:
        low = t.lower().strip()
        for n in names:
            if n.lower() == low:
                return n
        score, best = max(((difflib.SequenceMatcher(None, low, n.lower()).ratio(), n) for n in names),
                          default=(0, None))
        return best if score >= cut else None

    def course(self, t: str) -> str:
        t = t.strip()
        if hit := self._best(t, self.titles, 0.8):
            return hit
        hits = [title for title in self.titles if self.strings.find(t, title)]  # "DBMS", a typo, a word dropped
        return hits[0] if len(hits) == 1 else t

    def person(self, t: str) -> str:
        t = t.strip()
        if hit := self._best(t, self.faculty, 0.85):
            return hit
        last = t.split()[-1].lower() if t.split() else ""
        hits = [f for f in self.faculty if f.split()[-1].lower() == last]
        return hits[0] if len(hits) == 1 else t

    def who(self, t: str) -> dict:
        t = t.strip()
        for kind, names in (("group", self.groups), ("faculty", self.faculty), ("room", self.rooms)):
            if hit := self._best(t, names, 0.9):
                return {"kind": kind, "id": hit}
        return {"kind": "course", "id": self.course(t)}

    def value(self, role: str, text: str):
        t = text.strip().strip(".,;:!?\"'()")
        last = role.rsplit(".", 1)[-1]
        if role in INT_ROLES:
            return self.int(t)
        if last in ("day", "days") and role != "days":
            return self.day(t)
        if last in TIME_SUFFIX or role in ("start", "end"):
            return self.clock(t)
        if last == "kind" and role != "kind":
            return self.kind(t)
        if role == "date":
            return self.date(t)
        if last == "course" or role == "courses":
            return self.course(t)
        if last == "owner":
            return self.person(t)
        if role == "who":
            return self.who(t)
        if last == "group":
            return self._best(t, self.groups, 0.85) or t
        if role in ("room",):
            return self._best(t, self.rooms, 0.85) or t
        if role == "equipment":  # the directory's words: "GPU" -> "gpu", "eletcronics" -> "electronics"
            return self._best(t, self.equipment, 0.75) or t.lower()
        return t or None

    # --- one prediction -> IntentOutput ------------------------------------------------------------------

    def ranges(self, text: str, spans: list[dict]) -> dict[str, list]:
        """Extra values inside ranges said by their ends, per role."""
        extra: dict[str, list] = {}
        for x in spans:
            for y in spans:
                if x is y or x["role"] != y["role"] or x["end"] > y["start"]:
                    continue
                gap = text[x["end"]:y["start"]]
                between = re.search(r"between\s*$", text[:x["start"]], re.IGNORECASE) and \
                    re.match(r"^\s*and\s*$", gap, re.IGNORECASE)
                if not (SPAN_RANGE.match(gap) or between):
                    continue
                a, b = self.value(x["role"], x["text"]), self.value(y["role"], y["text"])
                last = x["role"].rsplit(".", 1)[-1]
                if last == "days" and a in ORDER and b in ORDER:
                    extra.setdefault(x["role"], []).extend(ORDER[ORDER.index(a) + 1:ORDER.index(b)])
                elif x["role"] in ("period.weeks", "period.except_weeks") and isinstance(a, int) and isinstance(b, int):
                    extra.setdefault(x["role"], []).extend(range(a + 1, b))
        return extra

    def intent(self, text: str, it: dict) -> dict:
        out: dict = {"kind": it["kind"]}
        spans = sorted(it["spans"], key=lambda s: s["start"])
        vals: dict[str, list] = {}
        for s in spans:
            if s["role"].endswith(".part"):  # the part of day is a class; its span is only where it is said
                continue
            v = self.value(s["role"], s["text"])
            if v is not None and v not in vals.setdefault(s["role"], []):
                vals[s["role"]].append(v)
        for role, more in self.ranges(text, [s for s in spans if not s["role"].endswith(".part")]).items():
            for v in more:
                if v not in vals.setdefault(role, []):
                    vals[role].append(v)
        if it["kind"] in ("clash_report",):  # course IDs in the directory: the titles name them
            pass
        for role, vs in vals.items():
            if not vs:
                continue
            many = role in LIST_FIELDS
            if role.endswith("days") and role != "days":
                vs = sorted(vs, key=ORDER.index)
            elif role in ("period.weeks", "period.except_weeks"):
                vs = sorted(vs)
            put(out, role, vs if many else vs[0])
        for k, v in it["classes"].items():
            put(out, k, v)
        if it["kind"] == "relate" and "relation" not in out:
            out["relation"] = "same_day"
        if it["kind"] == "limit" and "n" not in out:  # "at least an hour between": a number not said as one
            out["n"] = 1
        if it["kind"] == "swap":  # "Dr. Pillai and I would like to swap a slot": nothing said of either class
            out.setdefault("mine", {})
            out.setdefault("theirs", {})
        return out

    def decode(self, text: str, pred: dict) -> IntentOutput:
        if pred["action"] not in ("compile", "investigate") or not pred["intents"]:
            return IntentOutput(action=pred["action"])
        items = [self.intent(text, it) for it in pred["intents"]]
        names = iter("abcdefghijklmnopqrstuvwxyz")
        ids: dict[int, str] = {}

        def name(i: int) -> str:
            if i not in ids:
                ids[i] = next(names)
                items[i]["id"] = ids[i]
            return ids[i]

        chains: list[list[int]] = []
        for i, it in enumerate(pred["intents"]):
            refs = [f for f, on in (it.get("refs") or {}).items() if on]
            if refs and i > 0:
                for f in refs:
                    items[i].setdefault(f, {})
                    items[i][f] = {"same": name(i - 1), **{k: v for k, v in items[i][f].items()
                                                         if k not in ("course", "kind", "day", "time")}}
                if chains and chains[-1][-1] == i - 1:
                    chains[-1].append(i)
                else:
                    chains.append([i - 1, i])
        out = list(items)
        for kind, chain in zip(pred.get("composites") or [], chains):
            if kind == "fallback":
                chain = chain[:2]
            out.append({"kind": kind, "of": [name(i) for i in chain]})
        # an intent without the fields its kind needs is dropped (the gate sends such rows to the LLM)
        return IntentOutput.model_validate({"action": pred["action"], "intents": out})


def key(row: dict) -> tuple[str, str]:
    """The test split gives some ids to two messages (a request and its rewording): id and text together."""
    return row["id"], row["request"]["raw_text"][:200]


def reading(norm: Normaliser, row: dict, pred: dict) -> str | None:
    """The decoded reading as JSON (None if it does not decode): two encoders agree when these are equal."""
    try:
        return norm.decode(row["request"]["raw_text"], pred).model_dump_json(exclude_defaults=True)
    except Exception:  # noqa: BLE001
        return None


def put(d: dict, path: str, v) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = v


class Preds(Reader):
    """Reads a row from a file of encoder predictions (no model)."""

    def __init__(self, instance: Instance, preds: dict[str, dict], norm: Normaliser) -> None:
        super().__init__("encoder", instance)
        self.preds, self.norm = preds, norm

    def read_noted(self, row: dict):
        p = self.preds[key(row)]
        return self.norm.decode(row["request"]["raw_text"], p), []


def gold_as_pred(span_row: dict) -> dict:
    """A gold span row in the prediction format: the ceiling of the decoder."""
    comps = [c["kind"] for c in sorted(span_row.get("composites") or [], key=lambda c: c["of"][0])]
    return {"id": span_row["id"], "action": span_row["action"], "composites": comps, "conf": 1.0,
            "intents": [{"kind": it["kind"], "spans": it["spans"], "classes": it["classes"],
                         "refs": {k: True for k in it["refs"]}, "extent": it["extent"]}
                        for it in sorted(span_row["intents"], key=lambda it: (it["extent"] or [10 ** 6])[0])]}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sources(data: Path, suites: Path, enc: Path) -> dict[str, Path]:
    found = {"test": data / "test.jsonl", "val": enc / "val.jsonl", "composed": enc / "composed.jsonl"}
    found.update({f"suite-{p.stem}": p for p in sorted(suites.glob("*.jsonl"))})
    return {k: v for k, v in found.items() if v.exists()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", type=Path, help="a predictions/ folder of an encoder run")
    ap.add_argument("--roundtrip", action="store_true", help="decode the gold spans instead (the ceiling)")
    ap.add_argument("--data", type=Path, default=Path("data/compiler-intents"))
    ap.add_argument("--suites", type=Path, default=Path("data/suites"))
    ap.add_argument("--enc", type=Path, default=Path("data/encoder-intents"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--only", default=None, help="comma-separated split/suite names")
    ap.add_argument("--llm", type=Path, default=None, help="an LLM run's misses file (suites__<run>.jsonl)")
    ap.add_argument("--tau", type=float, default=0.98, help="the encoder's gate for the hybrid")
    ap.add_argument("--llm-choices", action="store_true",
                    help="the gate also sends every reading with a choice (any_of/ranked/fallback) to the LLM")
    ap.add_argument("--agree", type=Path, default=None,
                    help="a second encoder's predictions/: the gate also needs both to decode to the same reading")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    instance = Instance.model_validate_json(args.instance.read_text(encoding="utf-8"))
    norm = Normaliser(instance)
    names = sources(args.data, args.suites, args.enc)
    if args.only:
        names = {k: v for k, v in names.items() if k in args.only.split(",")}
    llm_miss: dict[str, set] = {}
    if args.llm:
        for m in load(args.llm):
            llm_miss.setdefault("test" if m["suite"] == "test" else f"suite-{m['suite']}", set()).add(
                (m["id"], m["text"][:200]))
    report = {}
    print(f"{'split':26} {'n':>4} {'outcome':>8} {'action':>7} {'silent':>7}"
          + (f" {'gate cov':>8} {'gate acc':>8} {'hybrid':>7} {'llm':>6}" if args.llm else ""))
    for split, path in names.items():
        rows = load(path)
        if args.roundtrip:
            spans_file = args.enc / "spans" / ("test.jsonl" if split == "test" else f"{split}.jsonl")
            if not spans_file.exists():
                continue
            got = [gold_as_pred(r) for r in load(spans_file)]
        else:
            pf = args.preds / f"{split}.jsonl"
            if not pf.exists():
                continue
            got = load(pf)
        if [p["id"] for p in got] != [r["id"] for r in rows]:
            raise SystemExit(f"{split}: predictions are not in the data's order")
        preds = {key(r): p for r, p in zip(rows, got, strict=True)}
        same_reading: dict | None = None
        if args.agree:
            other = load(args.agree / f"{split}.jsonl")
            same_reading = {key(r): reading(norm, r, p) == reading(norm, r, q)
                            for r, p, q in zip(rows, got, other, strict=True)}

        class Safe(Preds):
            def read_noted(self, row):
                try:
                    return super().read_noted(row)
                except Exception as e:  # noqa: BLE001 - an undecodable reading is an invalid answer
                    raise ValueError(f"decode: {e}") from e

        res = score(rows, instance, Safe(instance, preds, norm), checks=True)
        s = res["summary"]
        line = f"{split:26} {s['n']:>4} {s['outcome_match']:>8.1%} {s['action_accuracy']:>7.1%} {s['silent_wrong']:>7.1%}"
        entry = {"summary": s, "confusion": res["confusion"]}
        if args.llm and split in llm_miss or (args.llm and split.startswith(("test", "suite-"))):
            miss = llm_miss.get(split, set())
            ok = {(r["id"], r["text"][:200]): r["match"] for r in res["rows"]}
            # a reading that does not decode or validate is never used: the gate sends it to the LLM
            valid = {(r["id"], r["text"][:200]) for r in res["rows"] if r["pred_action"] != "invalid"}
            kept = [i for i in ok if preds[i].get("conf", 0) >= args.tau  # "ok" is the gold check: never gate on it
                    and i in valid and (same_reading is None or same_reading[i])
                    and not (args.llm_choices and preds[i].get("composites"))]
            cov = len(kept) / max(len(ok), 1)
            acc = sum(ok[i] for i in kept) / max(len(kept), 1)
            hybrid = sum(ok[i] if i in kept else i not in miss for i in ok) / max(len(ok), 1)
            llm = sum(i not in miss for i in ok) / max(len(ok), 1)
            line += f" {cov:>8.1%} {acc:>8.1%} {hybrid:>7.1%} {llm:>6.1%}"
            entry["hybrid"] = {"tau": args.tau, "agree": str(args.agree) if args.agree else None, "coverage": round(cov, 4), "accuracy_on_covered": round(acc, 4),
                               "hybrid": round(hybrid, 4), "llm": round(llm, 4)}
        print(line, flush=True)
        entry["misses"] = [r for r in res["rows"] if not r["match"]]
        report[split] = entry
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print("report:", args.out)


if __name__ == "__main__":
    main()
