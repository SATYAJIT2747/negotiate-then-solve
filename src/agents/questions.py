"""Questions about the timetable, in English: what a question asks (``QuestionReader``, regex for now) and the
reply (``Answerer``), with the facts from ``agents.lookup``.

    reader = QuestionReader(instance, sender="F-102", current_week=13)
    call = reader.read("when are Dr. Iyer and I both free on Friday?")
    # ToolCall(tool="free_hours", args={"teachers": ["F-102", "F-103"], "days": ["Fri"]}, week=13)
    Answerer(instance, sender="F-102").answer(call, WeekTimetable(instance, store, call.week))
    # "You and Dr. Iyer are both free on Friday 10 am–12 pm and 3–4 pm (week 13)."

The reader only says WHAT to look up; ``lookup`` says what is true. A model can later replace the reader:
it produces the same ``ToolCall``, and nothing else changes (ROADMAP.md, step 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from core.instance import Faculty, Group, Instance, Room
from language.corpus import FULL_DAY, hour

from .lookup import Directory, ScheduledClass, WeekTimetable

# what makes a message a timetable question, and what makes it a request instead
FREE = re.compile(r"\b(free|available|availability|empty|vacant|gaps?|unbooked)\b", re.IGNORECASE)
CLASSES = re.compile(r"\b(teach(es|ing)?|classes|class|lectures?|practicals?|tutorials?|timetable|schedule)\b|"
                     r"\bwhat do (i|we) have\b|\bwhen (is|are|does|do)\b", re.IGNORECASE)
REQUEST = re.compile(r"\b(cancel|move|reschedule|shift|swap|exchange|postpone|prepone|book|please|kindly)\b|"
                     r"\b(i|we) (want|need|would like|'d like)\b|"
                     r"\b(can|could|may) (i|we|you) (have|get|put|hold|teach|keep|take|make)\b", re.IGNORECASE)
ME = re.compile(r"\b(i|me|my|mine|myself|we|us|our)\b", re.IGNORECASE)
DAY = re.compile(r"\b(mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri)(day)?s?\b", re.IGNORECASE)
WEEK = re.compile(r"\bweek\s*(\d{1,2})\b", re.IGNORECASE)
TIME = re.compile(r"\b(\d{1,2})(?::\d\d)?\s*(am|pm)\b", re.IGNORECASE)
TEACHER = re.compile(r"\b(?:dr|prof|professor)\.?\s+([a-z]+)", re.IGNORECASE)
SECTION = re.compile(r"\bsection\s*(\d+)\b", re.IGNORECASE)
ROOM = re.compile(r"\b(lab|room)\s*(\d+)\b", re.IGNORECASE)
INITIALS = re.compile(r"\b[A-Z]{2,4}\b")  # "ML", "TOC" as written (not lower case: "is", "do")


@dataclass
class ToolCall:
    """What to look up: the same shape whether the regex or (later) a model decided it."""

    tool: str  # "free_hours" | "classes_of"
    args: dict = field(default_factory=dict)
    week: int = 1
    confident: bool = True
    note: str = ""  # when not confident: what to ask or say instead


class QuestionReader:
    """Reads a question's shape with regex, names with ``Directory``. Returns None for anything that is not a
    timetable question (a request for a change, a question about the rules), so another part handles it."""

    def __init__(self, instance: Instance, sender: str, current_week: int, today: date | None = None) -> None:
        self.instance = instance
        self.sender = sender
        self.current_week = current_week
        self.today = today or date.today()
        self.directory = Directory(instance)

    def read(self, text: str) -> ToolCall | None:
        if REQUEST.search(text):
            return None  # a change is being asked for, not information
        free, classes = FREE.search(text), CLASSES.search(text)
        if not free and not classes:
            return None
        week, days = self._when(text)
        teachers, sections, rooms, courses, unclear = self._names(text)
        if ME.search(text) or not (teachers or sections or rooms or courses):  # "my free hours", or no one named
            self._add_me(teachers, sections)
        if unclear:
            return ToolCall("unclear", week=week, confident=False, note=unclear)
        slots = sorted({_slot(int(h), ap) for h, ap in TIME.findall(text)} & set(range(self.instance.calendar.slots_per_day)))
        if free:
            args = {"teachers": teachers, "sections": sections, "rooms": rooms, "days": days, "slots": slots}
            return ToolCall("free_hours", {k: v for k, v in args.items() if v}, week)
        args = {"teacher": teachers[0] if teachers else None, "section": sections[0] if sections else None,
                "room": rooms[0] if rooms else None, "course": courses[0] if courses else None, "days": days}
        return ToolCall("classes_of", {k: v for k, v in args.items() if v}, week)

    def _when(self, text: str) -> tuple[int, list[str]]:
        """The week asked about (this week unless it says otherwise) and the days, if any."""
        low, week, days = text.lower(), self.current_week, []
        if m := WEEK.search(text):
            week = int(m.group(1))
        elif "next week" in low:
            week += 1
        for word, shift in (("today", 0), ("tomorrow", 1)):
            if re.search(rf"\b{word}\b", low):
                d = self.today + timedelta(days=shift)
                if d.weekday() > 4:  # a weekend: the next Monday, in the next week
                    d += timedelta(days=7 - d.weekday())
                    week += 1
                days.append(self.instance.calendar.days[d.weekday()])
        for m in DAY.finditer(text):
            day = m.group(1)[:3].title()
            if day in self.instance.calendar.days and day not in days:
                days.append(day)
        weeks = self.instance.calendar.weeks
        return min(max(week, 1), weeks), days

    def _names(self, text: str) -> tuple[list[str], list[str], list[str], list[str], str]:
        """Teachers, sections, rooms and courses the question names (IDs), and what is unclear, if anything."""
        teachers, sections, rooms, courses, unclear = [], [], [], [], ""
        said = [m.group(1) for m in TEACHER.finditer(text)]
        said += [f.name.split()[-1] for f in self.instance.faculty
                 if re.search(rf"\b{re.escape(f.name.split()[-1])}\b", text, re.IGNORECASE)]
        for name in dict.fromkeys(s.lower() for s in said):
            found = [x for x in self.directory.find(name) if isinstance(x, Faculty)]
            if len(found) == 1:
                teachers.append(found[0].id)
            else:
                unclear = (f"I don't know a teacher called {name.title()}." if not found else
                           f"Which one do you mean: {', '.join(f.name for f in found)}?")
        for m in SECTION.finditer(text):
            found = [x for x in self.directory.find(m.group(0)) if isinstance(x, Group)]
            sections += [g.id for g in found[:1]]
            unclear = unclear or ("" if found else f"I don't know {m.group(0)}.")
        for m in ROOM.finditer(text):
            found = [x for x in self.directory.find(m.group(0)) if isinstance(x, Room)]
            rooms += [r.id for r in found[:1]]
            unclear = unclear or ("" if found else f"I don't know {m.group(0)}.")
        low = text.lower()
        for course, title in self.instance.course_titles.items():
            if title.lower() in low or course.lower() in low:
                courses.append(course)
        for abbr in INITIALS.findall(text):
            courses += [c for c in self.directory.find(abbr) if isinstance(c, str) and c not in courses]
        return teachers, sections, rooms, courses, unclear

    def _add_me(self, teachers: list[str], sections: list[str]) -> None:
        """The sender: a teacher by their ID, a class representative ("ST-G-04") by their section."""
        if self.sender.startswith("ST-"):
            if self.sender[3:] not in sections:
                sections.insert(0, self.sender[3:])
        elif self.sender not in teachers:
            teachers.insert(0, self.sender)


class Answerer:
    """Runs a ToolCall on the week's timetable and writes the reply from what it returns."""

    def __init__(self, instance: Instance, sender: str) -> None:
        self.instance = instance
        self.sender = sender

    def answer(self, call: ToolCall, week: WeekTimetable) -> str:
        if not call.confident:
            return call.note
        if call.tool == "free_hours":
            return self._free(call, week)
        if call.tool == "classes_of":
            return self._classes(call, week)
        return "I can answer questions like: my free hours this week, when Dr. Iyer teaches Machine Learning, " \
               "or whether Lab 2 is free on Friday."

    def _free(self, call: ToolCall, week: WeekTimetable) -> str:
        a = call.args
        free = week.free_hours(a.get("teachers", ()), a.get("sections", ()), a.get("rooms", ()), a.get("days"))
        who, plural = self._who(a)
        when = f"week {call.week}"
        if a.get("slots"):  # "is Lab 2 free on Thursday at 3 pm?": yes or no for those hours
            days = a.get("days") or self.instance.calendar.days
            asked = [(d, s) for d in days for s in a["slots"]]
            taken = [h for h in asked if h not in free]
            at = ", ".join(f"{FULL_DAY[d]} {hour(s)}" for d, s in asked)
            if not taken:
                return f"Yes, {who} {'are' if plural else 'is'} free on {at} ({when})."  # who mid-sentence
            return f"No, {who} {'are' if plural else 'is'} not free on " \
                   f"{', '.join(f'{FULL_DAY[d]} {hour(s)}' for d, s in taken)} ({when})."
        if not free:
            on = f" on {', '.join(FULL_DAY[d] for d in a['days'])}" if a.get("days") else ""
            return _cap(f"{who} {'have' if plural else 'has'} no free hours{on} in {when}.")
        both = " both" if plural and len(self._people(a)) == 2 else ""
        return _cap(f"{who} {'are' if plural else 'is'}{both} free on {_spans(free)} ({when}).")

    def _classes(self, call: ToolCall, week: WeekTimetable) -> str:
        a = call.args
        found = week.classes_of(a.get("teacher"), a.get("section"), a.get("room"), a.get("course"))
        if a.get("days"):
            found = [c for c in found if c.placement.day in a["days"]]
        who, _ = self._who({"teachers": [a["teacher"]] if a.get("teacher") else [],
                            "sections": [a["section"]] if a.get("section") else [],
                            "rooms": [a["room"]] if a.get("room") else []})
        course = self.instance.course_title(a["course"]) if a.get("course") else ""
        when = f"week {call.week}"
        by_teacher = not a.get("teacher")  # name each class's teacher unless the question is about one teacher
        listed = "; ".join(self._one(c, teacher=by_teacher) for c in found)
        if not found and a.get("teacher") and a.get("course"):
            # "When does Dr. Iyer teach ML?" when someone else does: say who
            others = week.classes_of(course=a["course"])
            if others:
                teachers = _listed(list(dict.fromkeys(self.instance.faculty_by_id[c.session.faculty].name for c in others)))
                lead = f"{who} {"don't" if who == "you" else "doesn't"} teach {course} in {when}. It is taught by {teachers}: "
                return _cap(lead) + "; ".join(self._one(c) for c in others) + "."
        if not found:
            return f"No{' ' + course if course else ''} classes found for {who} in {when}."
        if a.get("teacher"):
            verb = "teach" if who == "you" else "teaches"
            return _cap(f"{who} {verb}{' ' + course if course else ''} in {when}: ") + listed + "."
        if a.get("section"):
            return _cap(f"{who}'s{' ' + course if course else ''} classes in {when}: ") + listed + "."
        if a.get("room"):
            return f"{course + ' c' if course else 'C'}lasses in {who} in {when}: " + listed + "."
        return f"{course} in {when}: " + listed + "."  # a course asked about, nobody named: everyone's classes of it

    def _one(self, c: ScheduledClass, teacher: bool = False) -> str:
        s, p = c.session, c.placement
        room = self.instance.room_by_id[p.room].name if p.room in self.instance.room_by_id else p.room
        by = f", {self.instance.faculty_by_id[s.faculty].name}" if teacher else ""
        when = f"{FULL_DAY[p.day]} {_range(p.slot, p.slot + s.duration)}"
        return f"{self.instance.course_title(s.course)} {s.kind.value}, {when} ({room}{by})"

    def _people(self, a: dict) -> list[str]:
        return [*a.get("teachers", []), *a.get("sections", []), *a.get("rooms", [])]

    def _who(self, a: dict) -> tuple[str, bool]:
        """ "you", "you and Dr. Iyer", "Section 4", "Lab 2": and whether it is plural ("are", "have")."""
        names = []
        for t in a.get("teachers", []):
            names.append("you" if t == self.sender else self.instance.faculty_by_id[t].name)
        for g in a.get("sections", []):
            names.append("your section" if self.sender == f"ST-{g}" else self.instance.group_by_id[g].name)
        names += [self.instance.room_by_id[r].name for r in a.get("rooms", [])]
        if not names:
            return "anyone", False
        return _listed(names), len(names) > 1 or names == ["you"]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _listed(items: list[str]) -> str:
    """["a"] -> "a"; ["a", "b"] -> "a and b"; ["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _slot(h: int, ampm: str) -> int:
    return h % 12 + (12 if ampm.lower() == "pm" else 0) - 9


def _range(start: int, end: int) -> str:
    """Slots 5 to 6 -> "2–3 pm"; 2 to 4 -> "11 am–1 pm" (am/pm written once when both ends share it)."""
    a, b = hour(start), hour(end)
    return f"{a.split()[0]}–{b}" if a.split()[1] == b.split()[1] else f"{a}–{b}"


def _spans(free: list[tuple[str, int]]) -> str:
    """[("Mon", 2), ("Mon", 3), ("Mon", 5), ("Mon", 7)] -> "Monday 11 am–1 pm, 2–3 pm and 4–5 pm"."""
    out, by_day = [], {}
    for d, s in free:
        by_day.setdefault(d, []).append(s)
    for d, slots in by_day.items():
        runs, start = [], slots[0]
        for prev, s in zip(slots, slots[1:] + [None], strict=False):
            if s != prev + 1:
                runs.append(_range(start, prev + 1))
                start = s
        out.append(f"{FULL_DAY[d]} {_listed(runs)}")
    return "; ".join(out)
