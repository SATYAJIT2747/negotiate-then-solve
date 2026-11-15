"""Reading everyday English the same way everywhere: typos, chat shorthand and abbreviations.

Small functions with no knowledge of intents, shared by the rule parser, the checks on a reading
(``language.checks``) and the resolution of names (``language.intents``):

* ``normalise`` spells the words the timetable cares about the usual way ("thrusday" -> "thursday",
  "b4" -> "before", "aftrnoon" -> "afternoon"), so a regular expression sees one spelling. Names and
  numbers are left alone: only words close to a short list of timetable words are changed;
* ``mentions`` says whether a text names a phrase, allowing a typo or two in a long one ("computer
  architecure") and its acronym ("CA") when asked to;
* ``hedged`` and ``necessary`` find the words that say how strongly something is asked: a wish
  ("if possible", "no worries if not") or a requirement ("must", "no other time works"); ``firmness_rule``
  is the labelling rule built on them, by kind of request.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache
from itertools import pairwise

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
    "teach", "teaching", "instead", "rather", "anymore", "anytime", "except", "every", "conference",
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
    r"\b(if (at all )?possible|if you can|if (that|it) can be managed|where (possible|feasible|the timetable allows)|"
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


def _close(a: str, b: str) -> bool:
    """One typo apart: a letter added, dropped, changed or two neighbours swapped ("wek", "trem", "rset").
    For the short words ``normalise`` leaves alone."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 3:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def _says(ws: list[str], vocabulary: tuple[str, ...]) -> bool:
    return any(_close(w, v) for w in ws for v in vocabulary)


# words that name weeks or a stretch of the term; a typo still counts, so a period is never dropped for one
_SPAN = ("week", "weeks", "fortnight", "semester", "term", "today", "tomorrow", "tonight", "until", "till",
         "onwards", "onward", "rest", "remainder", "remaining", "left", "january", "february", "march", "april",
         "june", "july", "august", "september", "october", "november", "december", "fortnightly",
         "alternate", "alternating", "biweekly")
_FROM_NOW = ("this", "next", "coming", "following")
_DAY = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WHOLE = ("whole", "entire", "throughout", "all", "full", "duration")
_TERM = ("semester", "term", "sem")
_REST = ("rest", "remainder", "remaining", "left", "until", "till", "onwards", "onward", "now")


def names_weeks(text: str) -> bool:
    """Whether the text says when, beyond the weekday: weeks, a stretch of the term, a date or "this Friday".
    A message that does not ("Monday morning would suit me") asks for every week."""
    ws = words(normalise(text))
    return (_says(ws, _SPAN) or any(re.fullmatch(r"\d{4}", w) for w in ws)
            or any(_says([a], _FROM_NOW) and b.startswith(_DAY) for a, b in pairwise(ws)))


def whole_term(text: str) -> bool:
    """The text asks for the whole semester ("for the whole semester", "throughout the term"), which is every
    week, and not for what is left of it."""
    ws = words(normalise(text))
    return _says(ws, _WHOLE) and _says(ws, _TERM) and not _says(ws, _REST)


def firmness(text: str) -> str:
    """How strongly a message asks for a time: "firm" (it says it must be so), "hedged" (it softens what
    it asks) or "neutral" (a plain request: "please move my class to Friday")."""
    return "firm" if necessary(text) else "hedged" if hedged(text) else "neutral"


# a must about where a class is now, not about where it should go ("it can't stay where it is")
_LEAVE = re.compile(r"\b(can't|cannot|can not|won't) (stay|remain|be kept|keep)\b[^.,;!?]*|"
                    r"\b(has|have|needs?|must) to (move|go|change)\b|\bmust (move|go|change)\b")


def placement_firm_only_on_leaving(text: str) -> bool:
    """Whether a must about a move is only about leaving where the class is ("it can't stay where it is", a
    clash with the time it is at now), not about where it should go ("I need it on Thursday at 2 pm"). Only
    when it says so: a message this does not recognise (a typo'd "nede") keeps the must on where it goes."""
    low = normalise(text)
    if not (_LEAVE.search(low) or _CONFLICT.search(low)):
        return False
    return not necessary(_LEAVE.sub(" ", low))


# the sender cannot be there: a reason that takes them away, or saying so ("can't", "unavailable", "on leave")
_UNABLE = re.compile(
    r"\b(can't|cannot|can not|unable|not able|won't be (able|available|around|in)|will not be (able|available)|"
    r"not available|unavailable|not free|away|on leave|leave|out of (station|town)|travel(l?ing)?|conference|"
    r"duty|not (be )?(around|on campus|in)|sick|medical|hospital|surgery|appointment|workshop|seminar|talk|"
    r"committee|meeting|exam|invigilat\w*|clinic|field ?work|visit\w*|busy|tied up|commitment|engaged|"
    r"miss|missing)\b", re.IGNORECASE)


def unable(text: str) -> bool:
    """Whether a message that rules times out says the sender cannot be there (or why not), rather than only
    that they want those times kept free ("no Monday mornings for me, a hard rule"). This, and not the
    reader's choice between ``unavailable`` and ``avoid``, decides whether a requirement is an unavailability:
    the tiers rank an unavailability above an operational requirement, which is policy, not reading."""
    return _UNABLE.search(normalise(text)) is not None


def placement_is_hard(text: str) -> bool:
    """The labelling rule for a request to put a class at a time (``language.intents.Move``), from the
    project brief's tiers: a preferred time is a preference (Tier 5), so a move is a requirement only when the
    message says it must be so, or names a conflict that rules out the time the class is at now."""
    return firmness(text) == "firm" or (_CONFLICT.search(normalise(text)) is not None and not hedged(text))


# saying outright that it is a requirement ("a hard rule", "not negotiable"); "hard for me" is a difficulty;
# what the sender can only do ("I can only teach afternoons"), and what may not be done ("nothing of mine may
# be placed on Wednesday")
_SAID_FIRM = re.compile(r"\b(hard|strict|firm) (rule|requirement|constraint|limit)|\b(not|non)[- ]?negotiable\b|"
                        r"\bis a must\b|\babsolutely (essential|necessary|required)\b|"
                        r"\b(i|we) can only\b|\bmay (not )?be (placed|scheduled|put|booked)\b|"
                        r"\bthe only (days?|times?|hours?) (i'm|i am|i can be) (on campus|in|here|around|free|available)\b",
                        re.IGNORECASE)
# the sender's own time, not the timetable's ("that's my only time to eat"): no must about a class
_OWN_ONLY = re.compile(r"\bmy only (time|slot)\b", re.IGNORECASE)
# asking for something, with a reason that is why it is wanted ("I'd like ... so I can do fieldwork")
_WISH = re.compile(r"\b(i'd like|i would like|i'd love|i want)\b", re.IGNORECASE)


def firmness_rule(kind: str, text: str) -> bool:
    """The labelling rule for how firm a message of one request is, by the request's kind: an unavailability
    is always a requirement, a move follows ``placement_is_hard``, and anything else is a requirement when the
    message says it must be so ("must", "I need", "a hard rule", "not negotiable", "I can only", "may not be
    placed"), or, for teaching kept to
    some times only, when it says the sender cannot be there otherwise ("I'm away the rest of the week") and
    does not ask for it as a wish ("I'd like to teach only on Mondays so I can do fieldwork").
    The generators label by it and the checks hold a reading to it, so data and checks follow one rule."""
    if kind == "unavailable":
        return True
    if kind == "move":
        return placement_is_hard(text)
    low = normalise(text)
    if hedged(text) and not necessary(text):
        return False
    return (necessary(_OWN_ONLY.sub(" ", low)) or _SAID_FIRM.search(low) is not None
            or (kind == "only" and unable(text) and not _WISH.search(low)))
