"""Reading everyday English the same way everywhere: typos, chat shorthand and abbreviations.

Small functions with no knowledge of intents, shared by the rule parser, the checks on a reading
(``language.checks``) and the resolution of names (``language.intents``):

* ``normalise`` spells the words the timetable cares about the usual way ("thrusday" -> "thursday",
  "b4" -> "before", "aftrnoon" -> "afternoon"), so a regular expression sees one spelling. Names and
  numbers are left alone: only words close to a short list of timetable words are changed;
* ``mentions`` says whether a text names a phrase, allowing a typo or two in a long one ("computer
  architecure") and its acronym ("CA") when asked to;
* ``hedged`` and ``necessary`` find the words that say how strongly something is asked: a wish
  ("if possible", "no worries if not") or a requirement ("must", "no other time works").
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache

# chat shorthand, by exact token
SHORTHAND = {
    "b4": "before", "bfr": "before", "aft": "after", "pls": "please", "plz": "please", "pl": "please",
    "u": "you", "ur": "your", "r": "are", "thx": "thanks", "ty": "thanks", "cant": "can't", "wont": "won't",
    "dont": "don't", "didnt": "didn't", "isnt": "isn't", "im": "i'm", "ive": "i've",
    "hrs": "hours", "hr": "hour", "mins": "minutes", "wk": "week", "wks": "weeks", "nxt": "next",
    "tmrw": "tomorrow", "tmr": "tomorrow", "lec": "lecture", "lecs": "lectures",
    "tut": "tutorial", "tuts": "tutorials", "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "weds": "wednesday", "thu": "thursday", "thur": "thursday", "thurs": "thursday",
    "fri": "friday", "sat": "saturday", "sun": "sunday", "noon": "12 pm", "midday": "12 pm",
}
# the words a typo is corrected to (at least five letters: shorter words are too easily confused)
VOCABULARY = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mondays", "tuesdays",
    "wednesdays", "thursdays", "fridays", "morning", "mornings", "afternoon", "afternoons", "evening", "lunch",
    "before", "after", "between", "until", "classes", "lecture", "lectures", "tutorial", "tutorials",
    "practical", "practicals", "session", "sessions", "available", "unavailable", "schedule", "scheduled",
    "possible", "prefer", "please", "cancel", "cancelled", "extra", "swap", "week", "weeks", "next",
    "teach", "teaching", "instead", "rather", "anymore", "anytime", "except", "every",
)
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?::\d\d)?|[^\sa-z\d]", re.IGNORECASE)
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)


@lru_cache(maxsize=4096)
def _correct(token: str) -> str:
    low = token.lower()
    if low in SHORTHAND:
        return SHORTHAND[low]
    if len(low) < 5 or low in VOCABULARY or not low.isalpha():
        return low
    hit = get_close_matches(low, VOCABULARY, n=1, cutoff=0.84)
    return hit[0] if hit else low


def normalise(text: str) -> str:
    """``text`` in lower case with the timetable's words spelled the usual way ("no clases b4 11 on thrusday"
    -> "no classes before 11 on thursday"). "3pm" becomes "3 pm"; punctuation and everything else stay."""
    low = re.sub(r"(\d)(am|pm)\b", r"\1 \2", text.lower())
    return re.sub(r"[a-z][a-z\d]*(?:'[a-z]+)?", lambda m: _correct(m[0]), low)


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


_STOP = {"and", "of", "the", "for", "in", "to", "a", "an", "with", "on"}


def acronym(phrase: str) -> str:
    """"Computer Architecture" -> "ca", "Theory of Computation" -> "toc"."""
    return "".join(w[0] for w in words(phrase) if w not in _STOP)


def mentions(text: str, phrase: str, *, typos: bool = True, acronyms: bool = False) -> bool:
    """Whether ``text`` names ``phrase``: exactly, with a typo or two when the phrase is long enough
    (``typos``), or by its acronym of two or more letters as a word of its own (``acronyms``)."""
    t, p = words(text), words(phrase)
    if not p:
        return False
    n = len(p)
    joined = " ".join(p)
    for i in range(len(t) - n + 1):
        if t[i:i + n] == p:
            return True
    if typos and len(joined) >= 8:
        for size in {n - 1, n, n + 1} - {0}:
            for i in range(len(t) - size + 1):
                window = " ".join(t[i:i + size])
                if abs(len(window) - len(joined)) <= 3 and SequenceMatcher(None, window, joined).ratio() >= 0.9:
                    return True
    if acronyms and len(a := acronym(phrase)) >= 2:
        return re.search(rf"(?<![\w.]){re.escape(a)}(?![\w.])", text, re.IGNORECASE) is not None
    return False


# how strongly something is asked
_HEDGES = re.compile(
    r"\b(if (at all )?possible|if you can|if that can be managed|where (possible|feasible|the timetable allows)|"
    r"ideally|would prefer|i'd prefer|prefer|i'd rather|rather not|would be nice|would suit|"
    r"no (problem|worries)( at all)? if (it can't|not|it cannot)|not urgent|not essential|only a preference|"
    r"would you mind|try to|if it's not too much trouble)\b",  # "is it possible", "any chance": politeness
    re.IGNORECASE)
_NEEDS = re.compile(
    r"\b(must|mustn't|cannot|can't|can not|unable|not able|has to|have to|needs? to|i need|required|"
    r"mandatory|no other time|only time|only slot|nothing else works|under any circumstances|"
    r"not possible|impossible|won't be (able|available)|not available|unavailable)\b",
    re.IGNORECASE)


def hedged(text: str) -> bool:
    """The text softens what it asks ("if possible", "no worries if not", "would you mind")."""
    return _HEDGES.search(normalise(text)) is not None


def necessary(text: str) -> bool:
    """The text says it must be so ("must", "can't", "no other time works"), outside its hedges
    ("no problem if it can't" is a hedge, not a requirement)."""
    return _NEEDS.search(_HEDGES.sub(" ", normalise(text))) is not None


# a conflict that rules out the time a class is at now ("it clashes with a committee meeting")
_CONFLICT = re.compile(r"\b(clash(es|ing)?|conflicts? with|overlaps? with|committee|meeting|exam duty|duty|"
                       r"appointment|conference|on leave|can't make|cannot make)\b", re.IGNORECASE)


def firmness(text: str) -> str:
    """How strongly a message asks for a time: "firm" (it says it must be so), "hedged" (it softens what
    it asks) or "neutral" (a plain request: "please move my class to Friday")."""
    return "firm" if necessary(text) else "hedged" if hedged(text) else "neutral"


def placement_is_hard(text: str) -> bool:
    """The labelling rule for a request to put a class at a time (``language.intents.Move``), from the
    project brief's tiers: a preferred time is a preference (Tier 5), so a move is a requirement only when the
    message says it must be so, or names a conflict that rules out the time the class is at now."""
    return firmness(text) == "firm" or (_CONFLICT.search(normalise(text)) is not None and not hedged(text))
