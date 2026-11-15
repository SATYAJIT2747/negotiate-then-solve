"""Preferences and changes for the semester timetable, from plain words.

    "Dr. Menon prefers not to teach before 10 am"
    "UG1 CSE students would like no classes after 5 pm on Saturday"
    "Dr. Iyer is unavailable on Fridays"
    "CS1.301 should be in the morning"
    "Room R-101 is closed on Tuesdays for renovation"

Who it is about comes from the offering document (faculty names, course
codes and names, programmes and years); days and times from the words.
"can't", "unavailable", "away", "must" make it a hard rule; anything else is
a preference the solver weighs. "off", "bad for me", "travelling" and the like mean
keep it free; "only after 11:30 on Mondays" keeps that day's earlier hours free and
leaves the other days alone. Dates ("on leave 10-20 August") are not a weekly
preference and are sent back with a question rather than misread as times.
"""

from __future__ import annotations

import re
import uuid

from .offerings import OfferingDoc
from .solver import Preference, Room, Window, faculty_key

DAYS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat"}
DAY_RE = re.compile(r"\b(mon|tue|tues|wed|thu|thur|thurs|fri|sat)(?:day|nesday|sday|urday|rsday|s)?s?\b", re.IGNORECASE)
HARD = re.compile(r"\b(can'?t|cannot|unavailable|not available|away|must|on leave|closed|shut|travell?ing|"
                  r"out of station|on duty)\b", re.IGNORECASE)
AVOID = re.compile(r"\b(no|not|avoid|without|free|don'?t|never|rather not|can'?t|cannot|unavailable|away|"
                   r"on leave|closed|shut|off|bad|busy|clash(?:es)?|conflicts?|travell?ing|out of station|on duty|"
                   r"meetings?|difficult|impossible|(?:does ?n[o']?t|won'?t) (?:suit|work))\b", re.IGNORECASE)
MONTH = (r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*")
DATES = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s*(?:{MONTH}\s*)?(?:-|–|to|until|till)\s*\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\b"
                   rf"|\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH}\b",
                   re.IGNORECASE)
YEAR = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "first": "I", "second": "II", "third": "III",
        "fourth": "IV", "fifth": "V"}
BRANCHES = ("CSE", "CSD", "ECE", "ECD", "CND", "CLD", "CHD", "CGD")
GENERIC = {"TECH", "YEAR", "YEARS", "SEMESTER", "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "AND", "THE", "FOR",
           "STUDENTS", "CLASSES", "CLASS", "SHOULD", "HAVE", "WOULD", "LIKE", "PLEASE", "NOT", "ON", "SECTION"}


class Parsed:
    def __init__(self) -> None:
        self.preferences: list[Preference] = []
        self.closures: list[tuple[str, Window]] = []
        self.summary: list[str] = []
        self.error: str | None = None


def _clock(h: str, m: str | None, ampm: str | None) -> str:
    hour = int(h)
    if ampm and ampm.lower().startswith("p") and hour < 12:
        hour += 12
    elif not ampm and 1 <= hour <= 7:
        hour += 12  # "after 5" in a timetable means 5 pm
    return f"{hour:02d}:{int(m or 0):02d}"


_T = r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"


def _window(text: str) -> tuple[str | None, str | None, str]:
    t = text.lower()
    if m := re.search(r"between\s+" + _T + r"\s+(?:and|to|-|–)\s+" + _T, t):
        a = _clock(m[1], m[2], m[3] or m[6])
        return a, _clock(m[4], m[5], m[6]), f"between {a} and {_clock(m[4], m[5], m[6])}"
    # a range: "2-4 pm", "from 2 to 4", "10:30 am - 12:00"; needs am/pm or "from", so "slot 2-3" is not a time
    if (m := re.search(r"(from\s+)?\b" + _T + r"\s*(?:-|–|to|until|till)\s*" + _T, t)) and (m[1] or m[4] or m[7]):
        a, b = _clock(m[2], m[3], m[4] or m[7]), _clock(m[5], m[6], m[7])
        if a < b:
            return a, b, f"between {a} and {b}"
    if re.search(r"\bbefore lunch\b", t):
        return None, "13:05", "before lunch"
    if re.search(r"\bafter lunch\b", t):
        return "14:00", None, "after lunch"
    if m := re.search(r"\b(?:before|until|till|earlier than)\s+" + _T, t):
        return None, _clock(*m.groups()), f"before {_clock(*m.groups())}"
    if m := re.search(r"\b(?:after|from|later than)\s+" + _T, t):
        return _clock(*m.groups()), None, f"after {_clock(*m.groups())}"
    if re.search(r"\bmornings?\b", t):
        return None, "13:05", "in the morning"
    if re.search(r"\bafternoons?\b", t):
        return "14:00", "18:40", "in the afternoon"
    if re.search(r"\bevenings?\b|\blate\b", t):
        return "17:00", None, "in the evening"
    if m := re.search(r"\bslot\s*([1-6])\b", t):
        slots = {"1": ("08:30", "09:55"), "2": ("10:05", "11:30"), "3": ("11:40", "13:05"), "4": ("14:00", "15:25"),
                 "5": ("15:35", "17:00"), "6": ("17:10", "18:40")}[m[1]]
        return slots[0], slots[1], f"in slot {m[1]}"
    if re.search(r"8[:.]30", t):
        return "08:30", "09:55", "in the 8:30 slot"
    return None, None, ""


def _days(text: str) -> list[str] | None:
    found = []
    for m in DAY_RE.finditer(text):
        d = DAYS.get(m[1][:3].lower())
        if d and d not in found:
            found.append(d)
    if re.search(r"\bweekends?\b", text, re.IGNORECASE):
        found.append("Sat")
    return found or None


def _cohorts(text: str, doc: OfferingDoc) -> list[str]:
    named, spans = [], []
    for c in doc.cohorts:
        if m := re.search(rf"\b{re.escape(c.name)}\b", text, re.IGNORECASE):
            named.append(c.id)
            spans.append(m.span())
    for m in re.finditer(r"\bsection\s+\w+\b", text, re.IGNORECASE):  # "Section 2" inside a programme heading
        if any(a <= m.start() < b for a, b in spans):  # part of a cohort named in full: that cohort only
            continue
        named += [c.id for c in doc.cohorts
                  if re.search(rf"\b{re.escape(m[0])}\b", c.name, re.IGNORECASE) and c.id not in named]
    if named:  # a cohort by its own name ("Section 2"), as in the demo department
        return named
    t = text.upper()
    year = None
    if m := re.search(r"\bUG\s?([1-5])\b", t):
        year = YEAR[m[1]]
    elif m := re.search(r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|[1-5](?:ST|ND|RD|TH)?)[- ]YEARS?\b", t):
        year = YEAR.get(m[1].lower()[:1] if m[1][0].isdigit() else m[1].lower())
    pg = re.search(r"\bM\.?\s?TECH|\bPG\b", t) is not None
    branches = [b for b in BRANCHES if re.search(rf"\b{b}\b", t)]
    if not year and not pg:
        return []
    out = []
    for c in doc.cohorts:
        name = c.name.upper()
        if pg != ("M.TECH" in name or "MTECH" in name):
            continue
        if year and not re.search(rf"\b{year} YEAR\b", name):
            continue
        if branches and not any(b in name for b in branches):
            continue
        out.append(c)
    # words of a programme's own name in the text ("M.Tech CASE first years"): only those programmes
    said = set(re.findall(r"[A-Z]{3,}", t)) - GENERIC
    overlap = {c.id: len(said & (set(re.findall(r"[A-Z]{3,}", c.name.upper())) - GENERIC)) for c in out}
    best = max(overlap.values(), default=0)
    return [c.id for c in out if not best or overlap[c.id] == best]


def _faculty(text: str, doc: OfferingDoc) -> list[str]:
    names = {faculty_key(f): f for c in doc.courses.values() for f in c.faculty if f}
    t = faculty_key(text)
    # whole words, so short names ("Rao", "Das") count and do not match inside other words
    hits = [full for key, full in names.items() if len(key) >= 3 and re.search(rf"\b{re.escape(key)}\b", t)]
    if not hits:  # a surname alone, when it is unique
        words = set(re.findall(r"[a-z]{3,}", t))
        by_last = {}
        for key, full in names.items():
            by_last.setdefault(key.split()[-1], []).append(full)
        hits = [v[0] for w, v in by_last.items() if w in words and len(v) == 1]
    return sorted(set(hits), key=len, reverse=True)[:1]


def _courses(text: str, doc: OfferingDoc) -> list[str]:
    codes = [c for c in re.findall(r"\b[A-Z]{2}\d\.\d{3}[a-z]?\b", text) if c in doc.courses]
    if codes:
        return codes
    t = text.lower()
    return [c.code for c in sorted(doc.courses.values(), key=lambda c: -len(c.name))
            if len(c.name) > 6 and c.name.lower() in t][:1]


def parse(text: str, doc: OfferingDoc, rooms: list[Room], source: str = "") -> Parsed:
    out = Parsed()
    if DATES.search(text):
        out.error = ("That names dates. A semester preference is about days and times each week (\"no classes on "
                     "Fridays\", \"not before 10 am\"); an absence on dates goes in as a request on the weekly timetable "
                     "once the semester has started.")
        return out
    days = _days(text)
    start, end, when = _window(text)
    hard = bool(HARD.search(re.sub(r"\baway from\b", "", text, flags=re.IGNORECASE)))
    room_ids = {r.id.lower(): r.id for r in rooms}
    room = next((room_ids[w.lower()] for w in re.findall(r"[A-Za-z0-9-]+", text) if w.lower() in room_ids), None)
    if room and re.search(r"\b(closed|shut|unavailable|renovation|maintenance|out of service)\b", text, re.IGNORECASE):
        w = Window(days=days or ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], start=start or "00:00", end=end or "23:59",
                   reason=text)
        out.closures.append((room, w))
        out.summary.append(f"close {room} {'on ' + ', '.join(w.days) if days else 'all week'}"
                           f"{' ' + when if when else ''}")
        return out
    if not days and not start and not end:
        out.error = "Say when: a day (\"on Fridays\") or a time (\"before 10 am\", \"in the evening\")."
        return out
    avoid = bool(AVOID.search(text))
    mode = "avoid" if avoid else "prefer"
    windows = [(start, end, when)]
    if mode == "prefer" and days and (start or end) and re.search(r"\bonly\b", text, re.IGNORECASE):
        # "only after 11:30 on Mondays": on those days, keep the other hours free; other days are not touched
        mode, windows = "avoid", []
        if start:
            windows.append((None, start, f"before {start}"))
        if end:
            windows.append((end, None, f"after {end}"))
    targets: list[tuple[str, str, str]] = []
    faculty, courses = _faculty(text, doc), _courses(text, doc)
    courses = [c for c in courses if not faculty or faculty[0] in doc.courses[c].faculty] or \
        ([] if faculty else courses)
    for code in courses:  # a course named (with or without its teacher): only that course
        targets.append(("course", code, f"{code} {doc.courses[code].name}"))
    if not targets:
        for f in faculty:
            targets.append(("faculty", f, f))
    if not targets:
        for cid in _cohorts(text, doc):
            name = next(c.name for c in doc.cohorts if c.id == cid)
            targets.append(("cohort", cid, name))
    if not targets:
        out.error = "Who is this about? Name a faculty member, a course (code or name) or a programme and year (\"UG1 CSE\")."
        return out
    for kind, who, label in targets:
        for start, end, when in windows:
            p = Preference(id=uuid.uuid4().hex[:8], target=kind, who=who, mode=mode, days=days, start=start, end=end,
                           hard=hard, weight=3, text=text.strip(), source=source)
            out.preferences.append(p)
            out.summary.append(f"{'must' if hard else 'prefer to'} {'avoid' if mode == 'avoid' else 'keep'} "
                               f"{label}{' on ' + ', '.join(days) if days else ''}{' ' + when if when else ''}"
                               f"{' (hard)' if hard else ''}")
    return out
