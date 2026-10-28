"""Notices to the people a committed change affects, built from the timetables, never from what the agent wrote.

    notices(before, after, actor="F-103")   # {"F-106": ["Your Algorithms lecture (Section 2), week 6: moves ..."]}

Whoever's class moved or is not held, other than the person who asked, is told what changed and when. No
reason is given: the request and what was said about it stay between the requester, the agent and the
office (a moved class's teacher learns only that the timetable changed).
"""

from __future__ import annotations

from nts.state import World
from nts.tools.names import FULL_DAY, hour


def notices(before: World, after: World, actor: str) -> dict[str, list[str]]:
    inst = after.instance
    out: dict[str, list[str]] = {}
    for w in sorted(after.timetables, key=lambda x: 0 if x is None else x):
        new, old = after.timetables[w], before.timetable(w)
        if old is None or (w in before.timetables and before.timetables[w] is new):
            continue
        when = "every week" if w is None else f"week {w}"
        for sid, p in sorted(new.assignment.items()):
            s = inst.session_by_id[sid]
            if s.faculty == actor:
                continue
            what = f"Your {inst.course_title(s.course)} {s.kind.value} ({', '.join(inst.group_by_id[g].name for g in s.groups)}), {when}"
            q = old.assignment.get(sid)
            if sid in new.cancelled and sid not in old.cancelled:
                out.setdefault(s.faculty, []).append(f"{what}: not held.")
            elif q is not None and q != p and sid not in new.cancelled:
                out.setdefault(s.faculty, []).append(
                    f"{what}: moves from {_at(inst, q)} to {_at(inst, p)}.")
    return out


def _at(inst, p) -> str:
    return f"{FULL_DAY[p.day]} {hour(p.slot)} in {inst.room_by_id[p.room].name}"
