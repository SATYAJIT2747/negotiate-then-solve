"""The department's names for things, both ways: what the model writes -> ids, ids -> what it reads.

The model only ever sees and writes names ("Dr. Iyer", "Lab 2", "Machine Learning", "Tue", "2 pm"), never
ids. A name that matches nothing, or more than one thing, is refused with the closest names, so the
model can fix it on its next turn instead of guessing.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

from nts.kernel.instance import Faculty, Group, Instance, Room

from .registry import Refusal

FIRST_HOUR = 9  # slot 0 starts at 9 am
FULL_DAY = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
            "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
SELF = {"me", "i", "my", "myself", "mine"}
_TITLES = re.compile(r"^(dr|prof|professor|mr|mrs|ms)\.?\s+")


def norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return _TITLES.sub("", " ".join(text.split()))


def hour(slot: int) -> str:
    h = FIRST_HOUR + slot
    return f"{h if h <= 12 else h - 12} {'am' if h < 12 else 'pm'}"


class Directory:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        courses = sorted({s.course for s in instance.sessions})
        self._teachers = [(f, [f.id, f.name, f.name.split()[-1]]) for f in instance.faculty]
        self._sections = [(g, [g.id, g.name]) for g in instance.groups]
        self._rooms = [(r, [r.id, r.name]) for r in instance.rooms]
        self._courses = [(c, [c, instance.course_title(c)]) for c in courses]

    # -- names -> things ------------------------------------------------------

    def teacher(self, text: str, actor: str | None = None) -> Faculty:
        if norm(text) in SELF:
            if actor not in self.instance.faculty_by_id:
                raise Refusal("not_a_teacher", "the person asking does not teach here, so 'me' names no teacher")
            return self.instance.faculty_by_id[actor]
        return _match("teacher", text, self._teachers, lambda f: f.name)

    def section(self, text: str) -> Group:
        return _match("section", text, self._sections, lambda g: g.name)

    def room(self, text: str) -> Room:
        return _match("room", text, self._rooms, lambda r: r.name)

    def course(self, text: str) -> str:
        return _match("course", text, self._courses, self.instance.course_title)

    def day(self, text: str) -> str:
        days = self.instance.calendar.days
        for d in days:
            if norm(text) in (d.lower(), FULL_DAY[d].lower()):
                return d
        raise Refusal("unknown_day", f"{text!r} is not a teaching day", teaching_days=[FULL_DAY[d] for d in days])

    def slot(self, text: str | int) -> int:
        """'2 pm', '2pm', '14:00', '14' or 14 -> the slot starting then."""
        m = re.fullmatch(r"\s*(\d{1,2})(?::00)?\s*(am|pm)?\s*", str(text).lower())
        h = int(m.group(1)) if m else -1
        bare_afternoon = m is not None and not m.group(2) and 1 <= h < FIRST_HOUR - 1  # "2" in teaching hours: 2 pm
        if (m is not None and m.group(2) == "pm" and h < 12) or bare_afternoon:
            h += 12
        n = self.instance.calendar.slots_per_day
        if not FIRST_HOUR <= h < FIRST_HOUR + n:
            raise Refusal("unknown_hour", f"{text!r} is not a teaching hour",
                          teaching_hours=[hour(q) for q in range(n)])
        return h - FIRST_HOUR

    def find(self, text: str) -> list[dict[str, str]]:
        """Everything ``text`` could name, of every kind."""
        out = []
        for kind, entries, show in (("teacher", self._teachers, lambda f: f.name),
                                    ("section", self._sections, lambda g: g.name),
                                    ("room", self._rooms, lambda r: r.name),
                                    ("course", self._courses, self.instance.course_title)):
            hits = _exact(text, entries) or _close(text, entries)
            out += [{"kind": kind, "name": show(x)} for x in hits]
        return out

    # -- things -> names ------------------------------------------------------

    def name(self, kind: str, id_: str) -> str:
        inst = self.instance
        if kind == "teacher":
            return inst.faculty_by_id[id_].name
        if kind == "section":
            return inst.group_by_id[id_].name
        if kind == "room":
            return inst.room_by_id[id_].name
        return inst.course_title(id_)


def _exact(text: str, entries: Iterable[tuple[object, list[str]]]) -> list:
    t = norm(text)
    return _unique(x for x, keys in entries if any(norm(k) == t for k in keys))


def _close(text: str, entries: list[tuple[object, list[str]]]) -> list:
    by_key: dict[str, list] = {}
    for x, keys in entries:
        for k in keys:
            by_key.setdefault(norm(k), []).append(x)
    near = difflib.get_close_matches(norm(text), list(by_key), n=5, cutoff=0.6)
    return _unique(x for k in near for x in by_key[k])


def _unique(xs: Iterable) -> list:
    out: list = []
    for x in xs:
        if all(x is not y for y in out):
            out.append(x)
    return out


def _match(kind: str, text: str, entries: list[tuple[object, list[str]]], show) -> object:
    hits = _exact(text, entries)
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise Refusal("ambiguous_name", f"{text!r} could be more than one {kind}", candidates=[show(x) for x in hits])
    raise Refusal("unknown_name", f"no {kind} called {text!r}", did_you_mean=[show(x) for x in _close(text, entries)])
