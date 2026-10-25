"""The department's regulations, as numbered rules the agent can look up and cite.

    handbook = Handbook.load("data/handbook.md")    # one "## <number> <title> [RULE-ID]" heading per rule
    handbook.search("can I teach in the lunch hour?", k=3)

Retrieval is BM25 over each rule's title and text (copied from ``agents.policy``, lexical only: rule
wording and numbers matter, and it needs no model). The ``rules`` tool returns what it finds with
the rule's citation, so an answer or a refusal can say which rule it rests on.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT = Path("data/handbook.md")

_HEADING = re.compile(r"^##\s+(?P<num>[\d.]+)\s+(?P<title>.+?)\s+\[(?P<id>[A-Z0-9-]+)\]\s*$")
_TOKEN = re.compile(r"[a-z0-9]+")
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b(\d{1,2}):(\d{2})\b", re.IGNORECASE)
_STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "i", "if", "in", "is", "it", "may", "me", "my", "no", "not", "of", "on", "or", "our", "so", "that", "the", "their", "them", "then", "there", "these", "this", "to", "we", "with", "you", "your"])


@dataclass(frozen=True)
class Rule:
    id: str
    number: str
    title: str
    text: str

    def cite(self) -> str:
        return f"§{self.number} {self.title} [{self.id}]"


def _clock(m: re.Match) -> str:
    """"1 pm", "1pm", "13:00" -> "h13", so times match whatever their spelling."""
    h = int(m[1]) % 12 + (12 if m[3].lower() == "pm" else 0) if m[3] else int(m[4])
    return f" h{h} "


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(_CLOCK.sub(_clock, text).lower().replace("-", " ")):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: "hours" ~ "hour"
        out.append(t)
    return out


class Handbook:
    def __init__(self, rules: list[Rule], k1: float = 1.5, b: float = 0.75) -> None:
        if len({r.id for r in rules}) != len(rules):
            raise ValueError("duplicate rule IDs")
        self.rules = rules
        self.docs = [Counter(tokens(f"{r.title} {r.title} {r.text}")) for r in rules]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths) if rules else 1.0
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(rules) - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def load(cls, path: Path | str = DEFAULT) -> Handbook:
        rules: list[Rule] = []
        head: re.Match | None = None
        body: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines() + ["## 0 end [END]"]:
            if m := _HEADING.match(line):
                if head:
                    rules.append(Rule(head["id"], head["num"], head["title"], " ".join(body).strip()))
                head, body = m, []
            elif head is not None and not line.startswith(("#", ">")):
                body.append(line.strip())
        return cls(rules)

    def by_id(self, rid: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rid.upper().strip("[]")), None)

    def search(self, query: str, k: int = 3) -> list[Rule]:
        q = tokens(query)
        scored = []
        for i, (d, n) in enumerate(zip(self.docs, self.lengths, strict=True)):
            s = sum(self.idf[t] * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * n / self.avg))
                    for t in q if t in d)
            scored.append((-s, i))
        return [self.rules[i] for s, i in sorted(scored)[:k] if s < 0]
