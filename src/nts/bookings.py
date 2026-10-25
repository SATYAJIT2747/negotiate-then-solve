"""Room bookings for clubs and events, checked against the timetable and the bookings already made.

    desk = Desk(store)
    decision = desk.request(Booking(club="Robotics", purpose="build night", week=6, day="Wed",
                                    from_hour="3 pm", to_hour="5 pm", attendees=40))
    decision.status          # "booked", "needs_approval" (very large: the office decides) or "refused"
    decision.checks          # each check, passed or not, in words; decision.alternatives when refused

A booking is a firm rule in the store: the room is closed for teaching then (``source.rule`` says
"booking: <club> | <purpose> | <attendees>"), committed through the kernel's gate. So a later repair never puts a class in a
booked room, and the bookings are part of the department's history like every other change. A booking
never moves a class: the room must be free in that week's timetable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from nts.kernel.schemas import (
    Constraint,
    ConstraintType,
    Scope,
    Source,
    Tier,
    Validity,
    When,
)
from nts.kernel.semantics import occupied
from nts.state import ConstraintAdded, ConstraintRemoved, Store
from nts.tools import Refusal
from nts.tools.draft import OFFICE
from nts.tools.names import FULL_DAY, Directory, hour

BIG = 150  # attendees: above this the office decides
PREFIX = "booking: "


class Booking(BaseModel):
    club: str
    purpose: str
    week: int
    day: str
    from_hour: str
    to_hour: str
    attendees: int = Field(ge=1)
    room: str | None = None  # none: the smallest free room that holds them


@dataclass
class Decision:
    status: Literal["booked", "needs_approval", "refused"]
    checks: list[dict]
    room: str | None = None
    id: str | None = None  # the booking's rule id, once booked or waiting
    alternatives: list[dict] = field(default_factory=list)


class Desk:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.pending: dict[str, tuple[Booking, Constraint]] = {}  # big bookings waiting for the office

    def bookings(self, week: int | None = None) -> list[dict]:
        """The bookings in force (and those waiting for the office), in words."""
        inst = self.store.world.instance
        rows = [(c, "booked") for c in self.store.world.constraints.values() if (c.source.rule or "").startswith(PREFIX)]
        rows += [(c, "needs_approval") for _, c in self.pending.values()]
        out = []
        for c, status in rows:
            w = c.valid.from_week
            if week is not None and w != week:
                continue
            club, purpose, attendees = ((c.source.rule or "").removeprefix(PREFIX).split(" | ") + ["", "", "0"])[:3]
            slots = c.when.slots or [0]
            out.append({"id": c.id, "status": status, "club": club, "purpose": purpose, "week": w,
                        "day": FULL_DAY[(c.when.days or ["Mon"])[0]], "from": hour(min(slots)), "to": hour(max(slots) + 1),
                        "room": inst.room_by_id[c.scope.room].name if c.scope.room else "", "attendees": int(attendees or 0)})
        return sorted(out, key=lambda r: (r["week"] or 0, r["day"], r["from"]))

    def request(self, b: Booking, by: str = "") -> Decision:
        world = self.store.world
        inst = world.instance
        d = Directory(inst)
        checks: list[dict] = []

        def check(name: str, ok: bool, detail: str) -> bool:
            checks.append({"check": name, "ok": ok, "detail": detail})
            return ok

        try:
            day = d.day(b.day)
            lo, hi = d.slot(b.from_hour), d.slot(b.to_hour, end=True)
        except Refusal as r:
            check("when", False, r.error.message)
            return Decision("refused", checks)
        check("when", 1 <= b.week <= inst.calendar.weeks and lo < hi,
              f"week {b.week}, {FULL_DAY[day]} {hour(lo)} to {hour(hi)}")
        if not checks[-1]["ok"]:
            return Decision("refused", checks)
        slots = list(range(lo, hi))
        lunch = inst.calendar.lunch_slot
        rooms = [d.room(b.room)] if b.room else sorted(inst.rooms, key=lambda r: r.capacity)
        busy = self._busy(b.week, day, slots)
        fits = [r for r in rooms if r.capacity >= b.attendees]
        free = [r for r in fits if r.id not in busy]
        check("size", bool(fits), f"{b.attendees} people" + ("" if fits else ": no room is big enough"
                                                            if not b.room else f": {rooms[0].name} holds {rooms[0].capacity}"))
        check("room free", bool(free), (f"{free[0].name} is free then" if free else
                                         "taken then by " + "; ".join(sorted({busy[r.id] for r in fits if r.id in busy}))))
        if lunch is not None and lunch in slots:
            checks.append({"check": "lunch hour", "ok": True, "detail": "it includes the lunch hour (allowed for clubs)"})
        if not all(c["ok"] for c in checks):
            return Decision("refused", checks, alternatives=self._alternatives(b, day, slots))
        room = free[0]
        c = Constraint(id=f"book-{b.week}-{day}-{lo}-{room.id}-{len(world.constraints)}", type=ConstraintType.UNAVAILABLE,
                       hard=True, tier=Tier.COMMITMENT, owner=None, scope=Scope(room=room.id),
                       when=When(days=[day], slots=slots), valid=Validity(from_week=b.week, to_week=b.week),
                       source=Source(rule=f"{PREFIX}{b.club} | {b.purpose} | {b.attendees}", request=by or b.club))
        if b.attendees > BIG:
            check("size of event", False, f"over {BIG} people: the timetable office decides")
            self.pending[c.id] = (b, c)
            return Decision("needs_approval", checks, room.name, c.id)
        self._commit(c, by)
        return Decision("booked", checks, room.name, c.id)

    def approve(self, cid: str) -> None:
        _, c = self.pending.pop(cid)
        self._commit(c, OFFICE)

    def refuse(self, cid: str) -> None:
        self.pending.pop(cid)

    def cancel(self, cid: str, by: str) -> None:
        c = self.store.world.constraints.get(cid)
        if c is None or not (c.source.rule or "").startswith(PREFIX):
            raise KeyError(cid)
        draft = self.store.draft(by, f"cancel-{cid}")
        draft.add(ConstraintRemoved(id=cid))
        self.store.commit(draft)

    def _commit(self, c: Constraint, by: str) -> None:
        draft = self.store.draft(by or OFFICE, f"booking-{c.id}")
        draft.add(ConstraintAdded(constraint=c))
        self.store.commit(draft)  # the gate checks no class is in the room then

    def _busy(self, week: int, day: str, slots: list[int]) -> dict[str, str]:
        """Room id -> what holds it in those hours: a class that week, or another booking."""
        world = self.store.world
        inst = world.instance
        tt = world.timetable(week)
        out: dict[str, str] = {}
        di = inst.day_index(day)
        if tt is not None:
            for sid, p in tt.assignment.items():
                s = inst.session_by_id[sid]
                if sid in tt.cancelled or p.day != day:
                    continue
                if {q for dd, q in occupied(s, di, p.slot)} & set(slots):
                    out.setdefault(p.room, f"a {inst.course_title(s.course)} {s.kind.value}")
        for c in [*world.constraints.values(), *(c for _, c in self.pending.values())]:
            if (c.type == ConstraintType.UNAVAILABLE and c.hard and c.scope.room and c.active_in(week)
                    and (c.when.days is None or day in c.when.days) and (c.when.slots is None or set(c.when.slots) & set(slots))):
                out.setdefault(c.scope.room, "a booking" if (c.source.rule or "").startswith(PREFIX) else "a closure")
        return out

    def _alternatives(self, b: Booking, day: str, slots: list[int]) -> list[dict]:
        """Up to three that would pass: another room at the same time, or the same length at another hour that day."""
        inst = self.store.world.instance
        out = []
        busy = self._busy(b.week, day, slots)
        for r in sorted(inst.rooms, key=lambda r: r.capacity):
            if r.capacity >= b.attendees and r.id not in busy:
                out.append({"room": r.name, "day": FULL_DAY[day], "from": hour(slots[0]), "to": hour(slots[-1] + 1)})
                break
        n = len(slots)
        for start in range(inst.calendar.slots_per_day - n + 1):
            alt = list(range(start, start + n))
            if alt == slots:
                continue
            taken = self._busy(b.week, day, alt)
            room = next((r for r in sorted(inst.rooms, key=lambda r: r.capacity)
                         if r.capacity >= b.attendees and r.id not in taken and (not b.room or r.name == b.room)), None)
            if room is not None:
                out.append({"room": room.name, "day": FULL_DAY[day], "from": hour(alt[0]), "to": hour(alt[-1] + 1)})
            if len(out) == 3:
                break
        return out
