"""Answers to questions about the timetable, computed from it (never from a model): which classes are held
in a week, when people, sections and rooms are free, and who or what a name in a question means.

    week = WeekTimetable(instance, store, week=13)
    week.classes_of(teacher="F-103", course="C-015")    # when does Dr. Iyer teach Machine Learning?
    week.free_hours(teachers=["F-102", "F-103"])         # when are Dr. Menon and Dr. Iyer both free?
    Directory(instance).find("lab 2")                    # [Room Lab 2]

A model may later decide which of these to call (ROADMAP.md, steps 2 and 4); what they return is the
timetable's answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.instance import Faculty, Group, Instance, Room, Session
from core.schemas import Placement
from pipeline.store import Store


class ScheduledClass:
    """One class held in one week: a Session (what it is) at a Placement (when and where)."""

    def __init__(self, session: Session, placement: Placement, week: int) -> None:
        self.session = session
        self.placement = placement
        self.week = week

    def hours(self) -> list[tuple[str, int]]:
        """The (day, slot) pairs it takes up: a 2-hour practical takes two."""
        p = self.placement
        return [(p.day, slot) for slot in range(p.slot, p.slot + self.session.duration)]

    def __repr__(self) -> str:
        return f"ScheduledClass({self.session.id} on {self.placement.day} slot {self.placement.slot}, week {self.week})"


class WeekTimetable:
    """The timetable as it stands in one week, as the Timetable page shows it: the week's own version if it
    has one (an absence, a closed room), else the semester's; classes not held that week left out; that
    week's extra classes added."""

    def __init__(self, instance: Instance, store: Store, week: int) -> None:
        self.instance = instance
        self.week = week
        version = store.current_version(week) or store.current_version()
        placed = dict(version.assignment) if version else {}
        not_held = set(version.cancelled) if version else set()
        sessions = dict(instance.session_by_id)
        for extra in store.extras(week):
            sessions[extra.session.id] = extra.session
            placed.setdefault(extra.session.id, extra.placement)
        self.classes = [ScheduledClass(sessions[sid], p, week)
                        for sid, p in placed.items() if sid in sessions and sid not in not_held]

    def classes_of(self, teacher: str | None = None, section: str | None = None, room: str | None = None,
                   course: str | None = None, day: str | None = None) -> list[ScheduledClass]:
        """The week's classes that match every filter given (IDs, e.g. teacher="F-103"), in time order."""
        found = [c for c in self.classes
                 if (teacher is None or c.session.faculty == teacher)
                 and (section is None or section in c.session.groups)
                 and (room is None or c.placement.room == room)
                 and (course is None or c.session.course == course)
                 and (day is None or c.placement.day == day)]
        days = self.instance.calendar.days
        return sorted(found, key=lambda c: (days.index(c.placement.day), c.placement.slot))

    def free_hours(self, teachers: Sequence[str] = (), sections: Sequence[str] = (), rooms: Sequence[str] = (),
                   days: Sequence[str] | None = None) -> list[tuple[str, int]]:
        """The (day, slot) hours when ALL the given teachers, sections and rooms are free; never the lunch hour."""
        busy = set()
        for c in self.classes:
            if (c.session.faculty in teachers or set(c.session.groups) & set(sections)
                    or c.placement.room in rooms):
                busy.update(c.hours())
        cal = self.instance.calendar
        return [(day, slot) for day in cal.days if days is None or day in days
                for slot in range(cal.slots_per_day)
                if slot != cal.lunch_slot and (day, slot) not in busy]


    def free_rooms(self, kind: str | None = None, equipment: Sequence[str] = (), min_capacity: int | None = None,
                   days: Sequence[str] | None = None) -> list[tuple[Room, list[tuple[str, int]]]]:
        """Rooms of a kind ("lab", "lecture"; None: any) with every piece of equipment asked for and at least
        ``min_capacity`` seats, each with its free (day, slot) hours this week; rooms with none left out."""
        rooms = [r for r in self.instance.rooms
                 if (kind is None or r.type.value == kind) and set(equipment) <= set(r.equipment)
                 and (min_capacity is None or r.capacity >= min_capacity)]
        out = [(r, self.free_hours(rooms=[r.id], days=days)) for r in rooms]
        return [(r, hours) for r, hours in out if hours]


class Directory:
    """Finds the teachers, sections, rooms and courses a question names, the way people name them:
    "iyer", "Dr. Iyer", "section 4", "lab 2", "machine learning", "ML"."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def find(self, text: str) -> list[Faculty | Group | Room | str]:
        """Everything ``text`` could mean: Faculty, Group and Room objects, and course IDs (e.g. "C-015")."""
        words = _words(text)
        if not words:
            return []
        found: list[Faculty | Group | Room | str] = []
        found += [f for f in self.instance.faculty if _matches(words, f.name)]
        found += [g for g in self.instance.groups if _matches(words, g.name)]
        found += [r for r in self.instance.rooms if _matches(words, r.name)]
        for course, title in self.instance.course_titles.items():
            if _matches(words, title) or _matches(words, course) or "".join(words) in _initials(title):
                found.append(course)
        return found


_TITLES = {"dr", "prof", "professor", "mr", "ms", "mrs"}


def _words(text: str) -> list[str]:
    """Lower-case words without titles: "Dr. Iyer" -> ["iyer"], "Lab 2" -> ["lab", "2"]."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _TITLES]


def _matches(words: list[str], name: str) -> bool:
    """Every word asked for starts a word of the name: "iyer" ~ "Dr. Iyer", "machine" ~ "Machine Learning",
    "lab 2" ~ "Lab 2" (but not "Lab 20")."""
    names = _words(name)
    return all(any(n == w or (not w.isdigit() and n.startswith(w)) for n in names) for w in words)


def _initials(title: str) -> set[str]:
    """How a course is abbreviated: "Machine Learning" -> {"ml"}, "Theory of Computation" -> {"toc", "tc"}."""
    words = _words(title)
    return {"".join(w[0] for w in words), "".join(w[0] for w in words if w not in {"of", "and", "the", "in", "for"})}
