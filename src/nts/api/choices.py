"""Answers a person can pick for the agent's question, read off the question's own words.

    choices("Wednesday 9 am, 10 am, or 11 am is free. Would any of those work?")
    # ['Wednesday 9 am', 'Wednesday 10 am', 'Wednesday 11 am', 'None of these']
    choices("Which kind of class (lecture or practical) would you like to add?")   # ['lecture', 'practical']
    choices("Shall I send it to the office?")                                      # ['Yes', 'No']

The page shows them as buttons above the chat box; typing an answer still works, since a list read off a
sentence can miss what the person means. Nothing here is sent to the model: a pick is sent as the answer text.
"""

from __future__ import annotations

import re

from nts.tools.names import FULL_DAY

_DAY = "|".join(FULL_DAY.values())
_TIME = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)"
# a day with times after it ("Wednesday 9 am, 10 am or 11 am"), or times alone
_TIMES = re.compile(rf"(?:\b({_DAY})\b(?:\s+at)?\s+)?({_TIME}(?:\s*(?:,|or|and)\s*(?:or\s+)?{_TIME})*)", re.IGNORECASE)
_EACH = re.compile(_TIME, re.IGNORECASE)
_OR = re.compile(r"\(([^()]{1,60}?\s+or\s+[^()]{1,60}?)\)"
                 r"|\b(?:either\s+)?((?:the\s+)?[\w-]+(?:\s[\w-]+){0,2})\s+or\s+((?:the\s+)?[\w-]+(?:\s[\w-]+){0,2})\s*\?",
                 re.IGNORECASE)
_YES_NO = re.compile(r"(?:^|[.!?]\s+)(would|will|do|does|did|shall|should|can|could|is|are|may|have|has)\b[^.!?]*\?",
                     re.IGNORECASE)
_STOP = {"which", "what", "the", "a", "an", "any"}
_CLAUSES = re.compile(r"(?:^|[.!?]\s+)([^.!?]{3,80}?),\s+or\s+([^.!?]{3,80}?)\?", re.IGNORECASE)
_LEAD = re.compile(r"^(?:(?:would|will|do|does|shall|should|can|could|is|are)\s+(?:you\s+|i\s+|we\s+)?(?:rather\s+|prefer\s+|like\s+"
                   r"|want\s+)?(?:to\s+)?|i\s+could\s+|should\s+i\s+)", re.IGNORECASE)
_TRAIL = re.compile(r"\s+(?:work|be\s+(?:ok|okay|fine|better|all\s+right))(?:\s+for\s+you)?$", re.IGNORECASE)


def _bare(clause: str) -> str:
    """One side of "A, or B?" as an answer: "would you prefer a different day" -> "A different day"."""
    text = _TRAIL.sub("", _LEAD.sub("", clause.strip())).strip()
    return text[:1].upper() + text[1:] if 2 < len(text) <= 60 else ""


def choices(question: str, *, most: int = 6) -> list[str]:
    out: list[str] = []
    for m in _TIMES.finditer(question):  # times offered: one button each, the day carried to every time
        day = m[1]
        out += [f"{day} {t}" if day else t for t in (re.sub(r"\s+", " ", x) for x in _EACH.findall(m[2]))]
    if len(out) < 2:
        out = []
        for m in _OR.finditer(question):  # "(lecture or practical)", "the Monday one or the Friday one?"
            parts = re.split(r",\s*|\s+or\s+", m[1]) if m[1] else [m[2], m[3]]
            parts = [re.sub(r"^(?:the|a|an)\s+", "", p.strip(), flags=re.IGNORECASE) for p in parts]
            out += [p for p in parts if p and p.lower() not in _STOP]
            if out:
                break
    if len(out) < 2 and (m := _CLAUSES.search(question)):  # "Would another time on Tuesday work, or would you
        out = [_bare(m[1]), _bare(m[2])]                       # prefer a different day?": two answers, not yes / no
        if all(out):
            return out
    yes_no = bool(_YES_NO.search(question))
    if len(out) >= 2:
        out = list(dict.fromkeys(out))[:most]
        return out + (["None of these"] if yes_no else [])
    return ["Yes", "No"] if yes_no else []
