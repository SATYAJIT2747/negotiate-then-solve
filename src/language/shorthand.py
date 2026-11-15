"""Shorthand in a message spelled out before it is read: course abbreviations ("DBMS", "OS", "InfoSec",
"TOC") and text-speak ("lec", "frm", "pls"), so "pls shift my DBMS lec frm monday" names Database Systems.

An abbreviation is spelled out only into a course title this department has, and only when it means one
title: "DS" in a department with both Data Structures and Distributed Systems is left as written (the
reader then asks which). Initials are taken from the titles themselves ("Computer Networks" -> CN, "Theory
of Computation" -> TOC and TC); a few common forms that are not initials ("DBMS", "InfoSec", "DSA") are
listed. Initials of two letters count only in capitals ("OS", not "os"), so ordinary words are not read as
courses.
"""

from __future__ import annotations

import re

from core.instance import Instance

# common names that are not the title's initials: abbreviation -> words the title must contain
COMMON = {
    "dbms": "database", "rdbms": "database", "db": "database",
    "infosec": "security", "isec": "security", "netsec": "security",
    "dsa": "data structures", "algo": "algorithms", "algos": "algorithms", "daa": "algorithms",
    "coa": "architecture", "nlp": "language processing", "dmath": "discrete mathematics",
}
SPEAK = {
    "lec": "lecture", "lecs": "lectures", "lect": "lecture", "lects": "lectures", "frm": "from", "pls": "please",
    "plz": "please", "pls.": "please", "thru": "through", "wk": "week", "wks": "weeks", "prac": "practical",
    "pract": "practical", "tmrw": "tomorrow", "b/w": "between", "bw": "between",
}
STOP = {"of", "and", "the", "for", "to", "in", "a", "an", "with"}


def _abbreviations(instance: Instance) -> dict[str, set[str]]:
    titles = {instance.course_title(s.course) for s in instance.sessions}
    out: dict[str, set[str]] = {}
    for t in titles:
        words = re.findall(r"[A-Za-z]+", t)
        for ws in (words, [w for w in words if w.lower() not in STOP]):
            if len(ws) >= 2:
                out.setdefault("".join(w[0] for w in ws).lower(), set()).add(t)
        for short, part in COMMON.items():
            if part in t.lower():
                out.setdefault(short, set()).add(t)
    return out


# "Tuesday at 10 am would be best, otherwise Thursday": the first choice and a fallback, said the way readers know
RANKED = re.compile(r"\s*(?:would be|is|would suit me)\s+(?:best|ideal|preferable|preferred|my first choice)\s*,?\s*"
                    r"(?:but\s+)?(?:otherwise|else|if not|failing that)\b,?", re.IGNORECASE)
FOR_MY = re.compile(r"^\s*for my (.+?) (lecture|practical|class|tutorial)\s*,\s*", re.IGNORECASE)
WHEN = re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\b\d{1,2}(?::\d\d)?\s*(?:am|pm)\b|\bweeks?\s+\d|"
                  r"\bmornings?\b|\bafternoons?\b|\bevenings?\b|\b(?:next|this) week\b", re.IGNORECASE)
CANCEL = re.compile(r"^\s*(?:please\s+)?(?:cancel|call off|drop|skip)\b", re.IGNORECASE)
REASON = re.compile(r"\s*(?:;|,?\s+(?:because|as|since)\s+)(?P<why>[^;]*)$", re.IGNORECASE)


def phrasing(text: str) -> str:
    """Two ways of putting a request that readers stumble on, put the way they read: a first choice with a
    fallback ("Tuesday would be best, otherwise Thursday" -> "Tuesday, or failing that Thursday"), and a reason
    trailing a cancellation that names no day, time or week ("...; the lab exam is the week after"), which is
    left out of what is read (it is still in the request as sent)."""
    if RANKED.search(text):
        text = RANKED.sub(", or failing that", text)
        if m := FOR_MY.match(text):
            text = f"My {m[1]} {m[2]} should move to " + text[m.end():]
    if CANCEL.search(text) and (m := REASON.search(text)) and not WHEN.search(m["why"]):
        text = text[:m.start()].rstrip() + "."
    return text


def expand(instance: Instance, text: str) -> str | None:
    """``text`` with its shorthand spelled out and its phrasing evened out (``phrasing``), or None when
    there is nothing to change."""
    original, text = text, phrasing(text)
    known = _abbreviations(instance)
    titles_said = {t.lower() for ts in known.values() for t in ts if t.lower() in text.lower()}

    def course(m: re.Match) -> str:
        word = m[0]
        hits = known.get(word.lower(), set())
        if len(hits) != 1:
            return word
        title = next(iter(hits))
        if title.lower() in titles_said:  # the title is already there ("Database Systems (DBMS)")
            return word
        if len(word) <= 2 and not word.isupper():  # "os", "is", "ai" in lower case: ordinary words
            return word
        return title

    out = re.sub(r"\b[A-Za-z]{2,8}\b", course, text)
    out = re.sub(r"(?<![\w/])(" + "|".join(re.escape(k) for k in sorted(SPEAK, key=len, reverse=True)) + r")(?![\w/])",
                 lambda m: SPEAK[m[0].lower()], out, flags=re.IGNORECASE)
    return out if out != original else None
