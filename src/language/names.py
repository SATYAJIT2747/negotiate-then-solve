"""Names in a reading: a parser writes courses, colleagues and rooms as the message names them, and this
module finds their IDs.

Finding which course "Data Structures" is, or who "Dr. Sharma" is, is a lookup, not reading: done here it is
exact, needs no directory in the prompt (most of a prompt's length was the directory), and a name that fits
nothing stays as it is, so resolution asks about it instead of guessing. An ID is still read as itself, so a
model trained on IDs reads the same as before.

* ``to_names``: a reading with IDs, written with names (the training targets);
* ``to_ids``: a reading with names (or IDs), with IDs where a name fits exactly one thing.

Which section's course a title means is not settled here: ``intents.classes`` takes a course for every
course of its title the owner teaches, and ``intents.ground_references`` puts a student's clash report in
the section the message names.
"""

from __future__ import annotations

import re
from difflib import get_close_matches

from core.instance import Instance

from .intents import ClashReport, ClassRef, IntentOutput, RoomClosed, Who
from .lexicon import _close, acronym, words


def _title_ids(instance: Instance) -> dict[str, str]:
    """Each course title (lower case) and the first course of it: the one a title is read as."""
    out: dict[str, str] = {}
    for cid in sorted(instance.course_titles):
        out.setdefault(instance.course_title(cid).lower(), cid)
    return out


def course_id(instance: Instance, name: str) -> str:
    """The course ``name`` is: an ID as itself, a title (a typo allowed) or an acronym in capitals ("ML") that
    only one title has; else ``name`` unchanged."""
    if name in instance.course_titles:
        return name
    titles = _title_ids(instance)
    low = " ".join(words(name))
    if low in titles:
        return titles[low]
    hit = get_close_matches(low, list(titles), n=1, cutoff=0.85)
    if hit:
        return titles[hit[0]]
    if name.isupper() and len(name) >= 2:
        same = [t for t in titles if acronym(t).upper() == name]
        if len(same) == 1:
            return titles[same[0]]
    return name


def faculty_id(instance: Instance, name: str) -> str:
    """The colleague ``name`` is ("Dr. Sharma", "sharma", "Prof Sharma", a typo allowed); else unchanged."""
    if name in instance.faculty_by_id:
        return name
    ws = [w for w in words(name) if w not in ("dr", "prof", "professor", "mr", "mrs", "ms", "sir", "madam")]
    if not ws:
        return name
    last = ws[-1]
    found = [f.id for f in instance.faculty if f.name.split()[-1].lower() == last]
    if not found and len(last) >= 5:
        found = [f.id for f in instance.faculty if _close(f.name.split()[-1].lower(), last)]
    return found[0] if len(found) == 1 else name


def room_id(instance: Instance, name: str) -> str:
    """The room ``name`` is ("Room 101", "room 101", "Lab 5"); else unchanged."""
    if any(r.id == name for r in instance.rooms):
        return name
    low = " ".join(words(name))
    found = [r.id for r in instance.rooms if " ".join(words(r.name)) == low]
    if not found and (num := re.findall(r"\d+", name)):  # "101", "lab-5": the number, and lab or not
        lab = "lab" in low
        found = [r.id for r in instance.rooms if re.findall(r"\d+", r.name) == num and ("lab" in r.name.lower()) == lab]
    return found[0] if len(found) == 1 else name


def _ref(instance: Instance, ref: ClassRef, to_id: bool) -> ClassRef:
    update = {}
    if ref.course is not None:
        update["course"] = course_id(instance, ref.course) if to_id else instance.course_title(ref.course) \
            if ref.course in instance.course_titles else ref.course
    if ref.owner is not None:
        update["owner"] = faculty_id(instance, ref.owner) if to_id else instance.faculty_by_id[ref.owner].name \
            if ref.owner in instance.faculty_by_id else ref.owner
    if ref.group is not None:
        group = {g.name.lower(): g.id for g in instance.groups}
        update["group"] = group.get(ref.group.lower(), ref.group) if to_id else instance.group_by_id[ref.group].name \
            if ref.group in instance.group_by_id else ref.group
    return ref.model_copy(update=update) if update else ref


def _who(instance: Instance, who: Who, to_id: bool) -> Who:
    if who.kind == "faculty":
        fid = faculty_id(instance, who.id) if to_id else instance.faculty_by_id[who.id].name \
            if who.id in instance.faculty_by_id else who.id
        return who.model_copy(update={"id": fid})
    if who.kind == "course":
        cid = course_id(instance, who.id) if to_id else instance.course_title(who.id) \
            if who.id in instance.course_titles else who.id
        return who.model_copy(update={"id": cid})
    return who


def _room_name(instance: Instance, rid: str) -> str:
    return next((r.name for r in instance.rooms if r.id == rid), rid)


def _convert(instance: Instance, out: IntentOutput, to_id: bool) -> IntentOutput:
    intents = []
    for intent in out.intents:
        update = {}
        for name in ("which", "mine", "theirs"):
            ref = getattr(intent, name, None)
            if isinstance(ref, ClassRef):
                update[name] = _ref(instance, ref, to_id)
        who = getattr(intent, "who", None)
        if isinstance(who, Who):
            update["who"] = _who(instance, who, to_id)
        if isinstance(intent, ClashReport):
            update["courses"] = [course_id(instance, c) if to_id else instance.course_title(c)
                                 if c in instance.course_titles else c for c in intent.courses]
        if isinstance(intent, RoomClosed):
            update["room"] = room_id(instance, intent.room) if to_id else _room_name(instance, intent.room)
        intents.append(intent.model_copy(update=update) if update else intent)
    return out.model_copy(update={"intents": intents})


def to_names(instance: Instance, out: IntentOutput) -> IntentOutput:
    """``out`` with names for its IDs: course titles, colleagues' names, room names. A class reference by
    session ID has no name and is kept."""
    return _convert(instance, out, to_id=False)


def to_ids(instance: Instance, out: IntentOutput) -> IntentOutput:
    """``out`` with IDs for the names that fit exactly one course, colleague or room; IDs as they are."""
    return _convert(instance, out, to_id=True)
