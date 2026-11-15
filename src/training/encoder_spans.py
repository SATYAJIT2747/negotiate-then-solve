"""Span labels for the encoder intent reader (docs/encoder-intents.md): each gold value found in the message.

    python -m training.encoder_spans            # data/encoder-intents/{train,val,composed}.jsonl and test
                                                # -> data/encoder-intents/spans/<split>.jsonl + report.json

The intents rows hold values (``"day": "Tue"``, ``"time": "14:00"``, ``"weeks": [9, 10, 11]``), not where the
message says them. An encoder reads values as spans of the message, so each value is looked up in the text the
ways the generators, the paraphrases and the noisy twins write it: days ("Tuesday", "tues", "wenesday"),
times ("2 pm", "14:00", "2-4pm", "at 2"), week and other numbers ("weeks 4-5", "two", "an hour"), class kinds
("lab", "lec"), course titles (exactly, by their abbreviations as ``language.shorthand`` reads them, or
nearly), and every other string (names, clubs, rooms, equipment) exactly or nearly.

A value said more than once is taken where its intent's other values are (the smallest stretch of text, in
the intents' order), so "move my Monday 9 am lecture to Tuesday at 9 am" gives each 9 am its own role. A day or
week inside a range ("Monday to Wednesday", "weeks 9-11") is covered by the range's two ends. A value found
nowhere is ``missing``: the reader cannot point at it, and a row with one is not usable for span training
(``aligned`` false); ``report.json`` counts them by role, tier and generator.

Out per row: the message, its context, the action, and per intent its kind, ``spans`` (role, start, end,
text), ``classes`` (the values chosen from a fixed set: hard, part, relative, parity, relation, measure,
justification, negate), ``refs`` (an item naming another's class: ``same``), ``extent`` (the stretch its spans
cover) and ``missing``; choices (any_of, ranked, fallback) as ``{"kind", "of": [intent indices]}``.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from core.instance import Instance
from language.shorthand import COMMON, SPEAK, _abbreviations

from .encoder_data import CHOICES, gold_of, tier

DAYS = {"Mon": "monday", "Tue": "tuesday", "Wed": "wednesday", "Thu": "thursday", "Fri": "friday",
        "Sat": "saturday", "Sun": "sunday"}
DAY_ABBR = {"mon": "Mon", "tue": "Tue", "tues": "Tue", "wed": "Wed", "weds": "Wed", "thu": "Thu", "thur": "Thu",
            "thurs": "Thu", "fri": "Fri"}
WEEKEND_ABBR = {"sat": "Sat", "sun": "Sun"}  # only capitalised: "sat" and "sun" are words too
KINDS = {"lecture": "lecture", "lectures": "lecture", "lec": "lecture", "lecs": "lecture", "lect": "lecture",
         "lects": "lecture", "lab": "practical", "labs": "practical", "practical": "practical",
         "practicals": "practical", "prac": "practical", "pract": "practical", "tutorial": "tutorial",
         "tutorials": "tutorial", "tut": "tutorial", "tuts": "tutorial"}
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2}
CLASS_FIELDS = {"hard", "negate", "justification", "relation", "measure", "part", "relative", "parity", "any"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
REF_FIELDS = {"which", "other", "mine", "theirs"}
TIME_FIELDS = {"time", "to"}
RANGE = re.compile(r"^\s*(?:-|–|to|through|thru|till|until|and)\s*(?:weeks?\s*)?$", re.IGNORECASE)
MER = r"(a\.?\s?m\.?|p\.?\s?m\.?)"


# --- mentions: where each kind of value is said ----------------------------------------------------------

def _clock(h: int, m: int, mer: str | None) -> str | None:
    if mer:
        pm = mer.lower().startswith("p")
        h = h % 12 + (12 if pm else 0)
    elif h < 8:  # "at 2": the teaching day is 09:00-17:00
        h += 12
    return f"{h:02d}:{m:02d}" if 0 <= h < 24 and 0 <= m < 60 else None


def times(text: str) -> list[tuple[int, int, str]]:
    out: dict[tuple[int, int], str] = {}
    # "2-4pm", "2 to 4 pm", "between 2 and 4 pm": the first takes the second's am/pm unless it would be later
    for m in re.finditer(rf"(?<![\d:])(\d{{1,2}})(?::(\d{{2}}))?\s*(?:-|–|to|and|till|until)\s*(\d{{1,2}})(?::(\d{{2}}))?"
                         rf"\s*{MER}", text, re.IGNORECASE):
        b = _clock(int(m[3]), int(m[4] or 0), m[5])
        a = _clock(int(m[1]), int(m[2] or 0), m[5])
        if a and b and a > b:
            a = _clock(int(m[1]), int(m[2] or 0), "am")
        if a and b:
            out[m.span(1)[0], m.span(2)[1] if m[2] else m.span(1)[1]] = a
            out[m.span(3)[0], m.end()] = b
    for m in re.finditer(rf"(?<![\d:])(\d{{1,2}}):(\d{{2}})(?:\s*{MER})?", text, re.IGNORECASE):
        if (c := _clock(int(m[1]), int(m[2]), m[3])) and not _inside(m.span(), out):
            out[m.span()] = c
    for m in re.finditer(rf"(?<![\d:])(\d{{1,2}})\s*{MER}(?![a-z])", text, re.IGNORECASE):
        if (c := _clock(int(m[1]), 0, m[2])) and not _inside(m.span(), out):
            out[m.span()] = c
    for m in re.finditer(r"\b(?:at|from|after|before|by|till|until|past)\s+(\d{1,2})(?:\s*o'?clock)?\b(?![:\d])",
                         text, re.IGNORECASE):
        if (c := _clock(int(m[1]), 0, None)) and not _inside(m.span(1), out):
            out[m.span(1) if "clock" not in m[0] else (m.start(1), m.end())] = c
    # "between 10 and 4", "from 9 to 1": bare hours, read inside the teaching day
    for m in re.finditer(r"\b(?:between|from)\s+(\d{1,2})\s*(?:-|–|and|to|till|until)\s*(\d{1,2})\b(?![:\d])",
                         text, re.IGNORECASE):
        for g in (1, 2):
            if (c := _clock(int(m[g]), 0, None)) and not _inside(m.span(g), out):
                out[m.span(g)] = c
    for m in re.finditer(r"\b(?:noon|midday|12 noon)\b", text, re.IGNORECASE):
        if not _inside(m.span(), out):
            out[m.span()] = "12:00"
    return sorted((a, b, c) for (a, b), c in out.items())


def _inside(span: tuple[int, int], taken) -> bool:
    return any(a < span[1] and span[0] < b for a, b in taken)


def words(text: str):
    return [(m.start(), m.end(), m[0]) for m in re.finditer(r"[A-Za-z]+(?:'[a-z]+)?|\d+", text)]


def days(text: str) -> list[tuple[int, int, str]]:
    out = []
    for a, b, w in words(text):
        low = w.lower()
        if low in DAY_ABBR:
            out.append((a, b, DAY_ABBR[low]))
        elif low in WEEKEND_ABBR and w[0].isupper():
            out.append((a, b, WEEKEND_ABBR[low]))
        elif len(low) >= 5:
            stem = low[:-1] if low.endswith("s") and not low.endswith("ss") else low
            # the nearest day, not the first near one: "thursday" is 0.8 like "tuesday"
            score, d = max((difflib.SequenceMatcher(None, stem, full).ratio(), d) for d, full in DAYS.items())
            if score == 1 or (score >= 0.8 and stem[0] == DAYS[d][0]):
                out.append((a, b, d))
    return out


UNIT = re.compile(r"^\s*(?:more\s+)?(?:hours?|hrs?|classes|class|lectures?|days?|weeks?|minutes?|sessions?|"
                  r"at a time|back[- ]to[- ]back|straight|in a row|consecutive|of us|people|members|students|attend)",
                  re.IGNORECASE)
WEEKISH = re.compile(r"(?:weeks?|wks?|wk)\s*(?:\d+\s*(?:,|and|to|-|–|through|till|until)\s*)*$", re.IGNORECASE)


def numbers(text: str, taken) -> list[tuple[int, int, int]]:
    # not a list's numbering ("2. Please ...") or a time's
    marker = {m.start(1) for m in re.finditer(r"(?:^|\n)\s*(\d+)[.)]\s", text)}
    out = [(a, b, int(w)) for a, b, w in words(text)
           if w.isdigit() and not _inside((a, b), taken) and a not in marker]
    # "a"/"an" is a number only before a unit ("an hour", "a day"), not in "a committee meeting"
    for m in re.finditer(r"\b(a couple of|an?(?=\s+(?:hour|day|class|lecture|break|gap|week|minute|session)s?\b)|"
                         r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
                         text, re.IGNORECASE):
        out.append((m.start(), m.end(), NUMBERS[m[1].lower()]))
    return out


def kinds(text: str) -> list[tuple[int, int, str]]:
    out = []
    for a, b, w in words(text):
        low = w.lower()
        if low in KINDS:
            out.append((a, b, KINDS[low]))
        elif len(low) >= 6:  # a typo of a full word: "practiical", "lectrues"
            score, k = max((difflib.SequenceMatcher(None, low, k).ratio(), k) for k in KINDS if len(k) >= 6)
            if score >= 0.8:
                out.append((a, b, KINDS[k]))
    return out


PARTS = {"morning": r"\bmornings?\b|\bbefore lunch\b|\bfirst half\b",
         "afternoon": r"\bafternoons?\b|\bafter lunch\b|\bpost[- ]lunch\b|\bsecond half\b|\bevenings?\b",
         "lunch": r"\blunch(?:time| hour| break)?\b"}


def parts(text: str) -> list[tuple[int, int, str]]:
    out = []
    for v, pat in PARTS.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            if v == "lunch" and re.search(r"(?:before|after|post[- ])\s*$", text[:m.start()], re.IGNORECASE):
                continue
            out.append((m.start(), m.end(), v))
    return out


def dates(text: str) -> list[tuple[int, int, str]]:
    """"23 November", "Nov 23rd", "23/11": as 2026-MM-DD (the semester's year)."""
    out = []
    mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}", text, re.IGNORECASE):
        out.append((m.start(), m.end(), int(m[1]), m[2]))
    for m in re.finditer(rf"\b{mon}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", text, re.IGNORECASE):
        out.append((m.start(), m.end(), int(m[2]), m[1]))
    res = []
    for a, b, d, mo in out:
        n = next(i for i, x in enumerate(MONTHS, 1) if x.startswith(mo.lower()[:3]))
        res.append((a, b, f"2026-{n:02d}-{d:02d}"))
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", text):
        res.append((m.start(), m.end(), f"2026-{int(m[2]):02d}-{int(m[1]):02d}"))
    return res


class Strings:
    """Where a string value (course, name, club, room, equipment) is said: exactly, nearly, or as an
    abbreviation of a course title."""

    def __init__(self, instance: Instance) -> None:
        self.abbr = _abbreviations(instance)

    def find(self, text: str, value: str) -> list[tuple[int, int]]:
        low, v = text.lower(), value.lower().strip()
        if not v:
            return []
        hits = [(m.start(), m.end()) for m in re.finditer(re.escape(v), low)]
        if hits:
            return hits
        ws = words(text)
        n = len(re.findall(r"[A-Za-z]+|\d+", v))
        for a, b, w in ws:  # an abbreviation that names only this title ("DBMS", "OS", "TOC")
            k = w.lower()
            titles = self.abbr.get(k, set())
            if (value in titles and len(titles) == 1 and (len(k) > 2 or w.isupper())) or \
                    (k in COMMON and COMMON[k] in v):
                hits.append((a, b))
        if hits:
            return hits
        for size in {max(n - 1, 1), n, n + 1}:  # nearly: a typo or a dropped word
            for i in range(len(ws) - size + 1):
                a, b = ws[i][0], ws[i + size - 1][1]
                if difflib.SequenceMatcher(None, low[a:b], v).ratio() >= 0.85:
                    hits.append((a, b))
        if not hits and " " in v:  # a person by surname ("Hegde" for "Dr. Hegde")
            last = v.split()[-1]
            hits = [(a, b) for a, b, w in ws if w.lower() == last and len(last) > 2]
        return sorted(set(hits))


# --- the values of an intent -----------------------------------------------------------------------------

def needs(intent: dict) -> tuple[list[tuple[str, str, object]], dict, dict]:
    """(role, type, value) for every value said in words, the chosen-from-a-set values, and refs."""
    out, classes, refs = [], {}, {}

    def add(role: str, typ: str, v) -> None:
        for x in v if isinstance(v, list) else [v]:
            out.append((role, typ, x))

    for k, v in intent.items():
        if k in ("kind", "id", "of") or v is None:
            continue
        if k in CLASS_FIELDS:
            classes[k] = v
        elif k in REF_FIELDS and isinstance(v, dict):
            for f, x in v.items():
                if f == "same":
                    refs[k] = x
                elif f in ("day",):
                    add(f"{k}.day", "day", x)
                elif f in ("time",):
                    add(f"{k}.time", "time", x)
                elif f == "kind":
                    add(f"{k}.kind", "kind", x)
                else:
                    add(f"{k}.{f}", "str", x)
        elif k in TIME_FIELDS and isinstance(v, dict):
            for f, x in v.items():
                if f == "part":  # a class, and where it is said (the anchor of "I can only do afternoons")
                    classes[f"{k}.part"] = x
                    add(f"{k}.part", "part", x)
                elif f == "days":
                    add(f"{k}.days", "day", x)
                else:  # at, after, before
                    add(f"{k}.{f}", "time", x)
        elif k == "period" and isinstance(v, dict):
            for f, x in v.items():
                if f in ("relative", "parity", "count"):  # "this week and next": a count said without a number
                    classes[f"period.{f}"] = x
                else:  # weeks, from_week, to_week, except_weeks
                    add(f"period.{f}", "int", x)
        elif k == "who" and isinstance(v, dict):
            if v.get("id"):
                add("who", "str", v["id"])
        elif k == "day":
            add(k, "day", v)
        elif k == "days":  # relate's day_gap: "at least a day between"
            add(k, "int", v)
        elif k == "date":
            add(k, "date", str(v))
        elif k in ("start", "end"):
            add(k, "time", v)
        elif k in ("attendees", "n", "max_consecutive"):
            add(k, "int", v)
        else:  # courses, club, activity, room, room_kind, cohorts, equipment, date
            add(k, "str", str(v) if not isinstance(v, list) else [str(x) for x in v])
    return out, classes, refs


class Aligner:
    def __init__(self, instance: Instance) -> None:
        self.strings = Strings(instance)

    def row(self, row: dict) -> dict:
        text = row["request"]["raw_text"]
        gold = gold_of(row)
        t = times(text)
        mentions = {"time": [(a, b, v) for a, b, v in t], "day": days(text), "kind": kinds(text),
                    "int": numbers(text, [(a, b) for a, b, _ in t]), "date": dates(text), "part": parts(text)}
        items = gold.get("intents") or []
        ids = {it["id"]: i for i, it in enumerate(items) if it.get("id")}
        composites = [{"kind": it["kind"], "of": [ids.get(x, x) for x in it.get("of", [])]}
                      for it in items if it["kind"] in CHOICES]
        plain = [(i, it) for i, it in enumerate(items) if it["kind"] not in CHOICES]
        info = []
        for _, it in plain:
            want, classes, refs = needs(it)
            info.append((want, classes, refs, [self._cands(text, mentions, role, typ, v) for role, typ, v in want]))
        picks = self._assign([c for *_, c in info])

        out_intents, used = [], set()
        for (i, it), (want, classes, refs, _), pick in zip(plain, info, picks, strict=True):
            spans, missing, typed = [], [], []
            for (role, typ, v), span in zip(want, pick, strict=True):
                if span is None:
                    missing.append((role, typ, v))
                else:
                    s = {"role": role, "start": span[0], "end": span[1], "text": text[span[0]:span[1]]}
                    spans.append(s)
                    typed.append(((role, typ, v), s))
            # a day or week inside a range said by its two ends is the range's, even when another request
            # says that day too ("Wednesday through Friday" beside "no Thursday classes")
            for n, s in typed:
                if n[1] in ("day", "int"):
                    rest = [x for x in spans if x is not s]
                    if not self._ranges(text, [n], rest, mentions):
                        spans.remove(s)
            # a class named by its day and time: its kind is the timetable's, not the message's (a swap's
            # "slot"), unless the kind is said beside them
            said = {s["role"].split(".")[0] for s in spans if s["role"].endswith((".day", ".time"))}
            inferred = []
            for n, s in typed:
                ref = n[0].split(".")[0]
                if n[1] != "kind" or ref not in said:
                    continue
                near = [x for x in spans if x["role"].startswith(ref + ".") and x is not s]
                if not any(abs(x["start"] - s["start"]) <= 30 for x in near):
                    spans.remove(s)
                    inferred.append(f"{n[0]}={n[2]}")
            missing = self._ranges(text, missing, spans, mentions)
            # the part of the day is a class: its words are an anchor when found, never missing ("early")
            missing = [m for m in missing if m[1] != "part"]
            inferred += [f"{r}={v}" for r, typ, v in missing if typ == "kind" and r.split(".")[0] in said]
            missing = [m for m in missing if not (m[1] == "kind" and m[0].split(".")[0] in said)]
            used |= {(s["start"], s["end"]) for s in spans}
            out_intents.append({"index": i, "kind": it["kind"], "spans": spans, "classes": classes,
                                "refs": {k: ids.get(v, v) for k, v in refs.items()},
                                "extent": [min(s["start"] for s in spans), max(s["end"] for s in spans)] if spans else None,
                                "inferred": inferred, "missing": [f"{r}={v}" for r, _, v in missing]})
        return {"id": row["id"], "tier": row.get("tier") or tier(gold), "generator": row.get("kind"),
                "repeat": bool(row.get("repeat")), "text": text,
                "context": {"role": row["request"]["role"], "current_week": row.get("current_week"),
                            "today": row.get("today")},
                "action": gold["action"], "intents": out_intents, "composites": composites,
                "aligned": all(not x["missing"] for x in out_intents)}

    def _cands(self, text: str, mentions: dict, role: str, typ: str, v) -> list[tuple[int, int]]:
        if typ == "str":
            return self.strings.find(text, v)
        got = [(a, b) for a, b, x in mentions[typ] if x == v]
        if typ == "int":  # a week number after "week(s)", any other number before its unit, when there is one
            fit = [(a, b) for a, b in got if (WEEKISH.search(text[max(0, a - 40):a]) if role.startswith("period.")
                                              else UNIT.match(text[b:b + 25]))]
            got = fit or got
        return got

    @staticmethod
    def _combos(cands: list[list[tuple[int, int]]], per: int = 8) -> list:
        """An intent's best few choices of one mention per value, each (picks, cost, first, last): the
        smallest stretch of text, a mention shared by two of its values only when nothing else fits."""
        found = [i for i, c in enumerate(cands) if c]
        if not found:
            return [([None] * len(cands), 0.0, None, None)]
        options = [cands[i] for i in found]
        size = 1
        for o in options:
            size *= len(o)
        if size > 4000:  # keep the three nearest the middle of every mention
            mid = sorted(a for o in options for a, _ in o)[sum(len(o) for o in options) // 2]
            options = [sorted(o, key=lambda s: abs(s[0] - mid))[:3] for o in options]
        scored = []
        for combo in itertools.product(*options):
            lo, hi = min(a for a, _ in combo), max(b for _, b in combo)
            scored.append(((hi - lo) + 30 * (len(combo) - len(set(combo))), lo, hi, combo))
        scored.sort(key=lambda x: (x[0], x[1]))
        out = []
        for cost, lo, hi, combo in scored[:per]:
            full = [None] * len(cands)
            for i, s in zip(found, combo, strict=True):
                full[i] = s
            out.append((full, cost, lo, hi))
        return out

    @classmethod
    def _assign(cls, all_cands: list[list[list[tuple[int, int]]]], beam: int = 32) -> list:
        """Every intent's mentions chosen together (a beam over the intents in order): each intent's own
        cost, plus a mention another intent already has (40), one before the previous intent (200) or
        inside its words (20). One intent at a time took the tightest copy of a class said twice and left
        the other intent to share it."""
        states = [(0.0, [], frozenset(), 0, 0)]
        for cands in all_cands:
            options = cls._combos(cands)
            new = []
            for cost, picks, used, prev, prev_end in states:
                for full, local, lo, hi in options:
                    if lo is None:
                        new.append((cost + local, picks + [full], used, prev, prev_end))
                        continue
                    spans = {s for s in full if s}
                    starts = [a for a, _ in spans]
                    c = (local + 40 * sum(1 for s in spans if s in used) + 200 * sum(1 for a in starts if a < prev)
                         + 20 * sum(1 for a in starts if prev <= a < prev_end))
                    new.append((cost + c, picks + [full], used | spans, lo, hi))
            new.sort(key=lambda x: x[0])
            states = new[:beam]
        return states[0][1]

    @staticmethod
    def _ranges(text: str, missing: list, spans: list, mentions: dict) -> list:
        """A day or week left out because it sits inside a range said by its two ends."""
        left = []
        for role, typ, v in missing:
            ok = False
            if typ in ("day", "int"):
                ends = [s for s in spans if s["role"] == role]
                order = list(DAYS)
                for x, y in itertools.permutations(ends, 2):
                    if x["end"] <= y["start"] and RANGE.match(text[x["end"]:y["start"]]):
                        a, b = x["text"], y["text"]
                        if typ == "int" and a.isdigit() and b.isdigit():
                            ok = int(a) < v < int(b)
                        elif typ == "day":
                            da = [d for s, e, d in mentions["day"] if s == x["start"]]
                            db = [d for s, e, d in mentions["day"] if s == y["start"]]
                            ok = bool(da and db) and order.index(da[0]) < order.index(v) < order.index(db[0])
                    if ok:
                        break
            if not ok:
                left.append((role, typ, v))
        return left


def report(rows: list[dict]) -> dict:
    by_tier, by_gen = defaultdict(Counter), defaultdict(Counter)
    roles, missing = Counter(), Counter()
    for r in rows:
        by_tier[r["tier"]][r["aligned"]] += 1
        by_gen[r["generator"]][r["aligned"]] += 1
        for it in r["intents"]:
            for s in it["spans"]:
                roles[s["role"]] += 1
            for m in it["missing"]:
                missing[m.split("=")[0]] += 1
    share = lambda c: f"{c[True]}/{c[True] + c[False]} ({c[True] / max(c[True] + c[False], 1):.0%})"
    worst = sorted(by_gen.items(), key=lambda kv: kv[1][True] / max(sum(kv[1].values()), 1))[:12]
    return {"rows": len(rows), "aligned": share(Counter(r["aligned"] for r in rows)),
            "by_tier": {t: share(c) for t, c in by_tier.items()},
            "worst_generators": {g: share(c) for g, c in worst},
            "missing_by_role": dict(missing.most_common(20)), "spans_by_role": dict(roles.most_common())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/encoder-intents"))
    ap.add_argument("--test", type=Path, default=Path("data/compiler-intents/test.jsonl"))
    ap.add_argument("--instance", type=Path, default=Path("data/synthetic-cse-s0.json"))
    ap.add_argument("--suites", type=Path, default=Path("data/suites"), help="each suite -> spans/suite-<name>.jsonl")
    args = ap.parse_args()
    al = Aligner(Instance.model_validate_json(args.instance.read_text(encoding="utf-8")))
    out = args.data / "spans"
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    suites = [(f"suite-{p.stem}", p) for p in sorted(args.suites.glob("*.jsonl"))] if args.suites.exists() else []
    for name, path in [("train", args.data / "train.jsonl"), ("val", args.data / "val.jsonl"),
                       ("composed", args.data / "composed.jsonl"), ("test", args.test), *suites]:
        rows = [al.row(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rep[name] = report(rows)
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for name, r in rep.items():
        print(f"\n== {name}: {r['rows']} rows, aligned {r['aligned']}")
        print("  by tier ", r["by_tier"])
        print("  missing ", r["missing_by_role"])
        print("  worst   ", r["worst_generators"])


if __name__ == "__main__":
    main()
