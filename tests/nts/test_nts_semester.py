import time

import pytest
from fastapi.testclient import TestClient

import nts.bookings
from nts.api import Service, create_app
from nts.bookings import Booking, Desk
from nts.kernel.schemas import Constraint, ConstraintType, Scope, Tier, When
from nts.models import ScriptedModel
from nts.semester import rebuild, unmet
from nts.state import ConstraintAdded, InvalidChange, TimetableSet
from nts.telemetry import MemorySink, Tracer
from nts.tools.names import FULL_DAY, hour


def with_wishes(store, n=4):
    """Wishes ("no classes then, if possible") on hours where teachers teach now: the timetable breaks them all."""
    inst = store.world.instance
    tt = store.world.timetable(None)
    draft = store.draft("system")
    seen = set()
    for sid, p in tt.assignment.items():
        s = inst.session_by_id[sid]
        if s.faculty in seen or len(seen) == n:
            continue
        seen.add(s.faculty)
        draft.add(ConstraintAdded(constraint=Constraint(
            id=f"wish-{s.faculty}", type=ConstraintType.AVOID, hard=False, tier=Tier.PREFERENCE, owner=s.faculty,
            scope=Scope(faculty=s.faculty), when=When(days=[p.day], slots=[p.slot]))))
    store.commit(draft)
    return store


def test_a_rebuild_meets_wishes_that_repairs_did_not(store):
    with_wishes(store)
    inst, world = store.world.instance, store.world
    assert len(unmet(inst, world.timetable(None).assignment, world.constraints.values())) == 4
    r = rebuild(store, from_week=5, time_limit=30)
    st = r.stats
    assert st["wishes"] == 4 and st["met_before"] == 0 and st["met_after"] > st["met_before"] and st["moved"] > 0
    assert len(r.unmet) == st["wishes"] - st["met_after"]
    before = store.version
    store.commit(r.draft)  # published through the gate like any change
    assert store.version != before
    assert store.world.timetable(None).assignment == r.draft.world.timetable(None).assignment


def test_a_rebuild_keeps_weeks_that_already_ran(store):
    world = store.world
    tt = world.timetable(None)
    draft = store.draft("system")
    for w in (3, 8):
        draft.add(TimetableSet(week=w, assignment=dict(tt.assignment)))
    store.commit(draft)
    r = rebuild(store, from_week=5, time_limit=30)
    assert r.stats["weeks_kept"] == [3] and r.stats["weeks_repaired"] == [8]


def free_slot(store, week):
    """A day and hour when some room has no class in ``week``, and that room."""
    world = store.world
    inst = world.instance
    tt = world.timetable(week)
    used = {(p.room, p.day, q) for sid, p in tt.assignment.items()
            for q in range(p.slot, p.slot + inst.session_by_id[sid].duration)}
    for day in inst.calendar.days:
        for q in range(inst.calendar.slots_per_day):
            for r in sorted(inst.rooms, key=lambda r: r.capacity):
                if (r.id, day, q) not in used:
                    return day, q, r
    raise AssertionError("every room is full")


def booking(store, week=6, **kw):
    day, q, room = free_slot(store, week)
    args = {"club": "Robotics", "purpose": "build night", "week": week, "day": FULL_DAY[day], "from_hour": hour(q),
            "to_hour": hour(q + 1), "attendees": 20, "room": room.name} | kw
    return Booking(**args), room


def test_a_booking_closes_the_room_for_teaching_then(store):
    desk = Desk(store)
    b, room = booking(store)
    d = desk.request(b)
    assert d.status == "booked" and d.room == room.name and all(c["ok"] for c in d.checks)
    c = store.world.constraints[d.id]
    assert c.hard and c.type == ConstraintType.UNAVAILABLE and c.scope.room == room.id and c.active_in(6)
    assert not c.active_in(7)
    rows = desk.bookings(6)
    assert rows == [{**rows[0], "club": "Robotics", "purpose": "build night", "attendees": 20, "room": room.name}]

    again = desk.request(b.model_copy(update={"club": "Drama"}))
    assert again.status == "refused" and "a booking" in next(x["detail"] for x in again.checks if not x["ok"])
    assert again.alternatives  # another room, or another hour

    desk.cancel(d.id, "club:Robotics")
    assert d.id not in store.world.constraints and desk.bookings() == []


def test_a_booking_never_moves_a_class(store):
    world = store.world
    inst = world.instance
    _, p = next(iter(world.timetable(6).assignment.items()))
    b = Booking(club="Chess", purpose="meet", week=6, day=FULL_DAY[p.day], from_hour=hour(p.slot),
                to_hour=hour(p.slot + 1), attendees=10, room=inst.room_by_id[p.room].name)
    d = Desk(store).request(b)
    assert d.status == "refused" and "taken then by a" in next(x["detail"] for x in d.checks if x["check"] == "room free")
    with pytest.raises(InvalidChange):  # and the gate would refuse the closure anyway
        draft = store.draft("system")
        draft.add(ConstraintAdded(constraint=Constraint(id="x", type=ConstraintType.UNAVAILABLE, hard=True,
                                                        tier=Tier.COMMITMENT, scope=Scope(room=p.room),
                                                        when=When(days=[p.day], slots=[p.slot]))))
        store.commit(draft)


def test_too_big_or_too_large_events(store, monkeypatch):
    desk = Desk(store)
    b, _ = booking(store)
    huge = desk.request(b.model_copy(update={"attendees": 5000, "room": None}))
    assert huge.status == "refused" and "no room is big enough" in huge.checks[1]["detail"]
    monkeypatch.setattr(nts.bookings, "BIG", 10)
    big = desk.request(b)
    assert big.status == "needs_approval" and big.id not in store.world.constraints
    assert desk.bookings()[0]["status"] == "needs_approval"
    desk.approve(big.id)
    assert big.id in store.world.constraints and desk.bookings()[0]["status"] == "booked"


def test_clubs_book_and_the_office_rebuilds_through_the_api(store):
    with_wishes(store, 2)
    app = create_app({"demo": Service(store, ScriptedModel([]), Tracer([MemorySink()]))})
    club, office, teacher = (TestClient(app) for _ in range(3))
    assert club.post("/api/login", json={"dept": "demo", "name": "club: Robotics"}).json()["role"] == "club"
    office.post("/api/login", json={"dept": "demo", "name": "office"})
    teacher.post("/api/login", json={"dept": "demo", "name": store.world.instance.faculty[0].name})

    b, room = booking(store)
    body = b.model_dump(exclude={"club"})
    got = club.post("/api/bookings", json=body).json()
    assert got["status"] == "booked" and got["room"] == room.name
    assert [r["club"] for r in club.get("/api/bookings").json()] == ["Robotics"]
    assert teacher.post("/api/bookings", json=body).status_code == 403
    assert club.post("/api/episodes", json={"request": "hi"}).status_code == 403
    assert office.post("/api/bookings", json=body).status_code == 422  # the office names the club
    assert club.delete(f"/api/bookings/{got['id']}").json() == {"status": "cancelled"}

    assert teacher.post("/api/semester/rebuild", json={}).status_code == 403
    office.post("/api/semester/rebuild", json={"from_week": 5})
    for _ in range(300):  # the rebuild runs in a thread: poll it
        st = office.get("/api/semester/rebuild").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "ready" and st["stats"]["met_after"] >= st["stats"]["met_before"] and not st["stale"]
    before = store.version
    assert office.post("/api/semester/publish").json()["status"] == "published" and store.version != before
    assert office.post("/api/semester/publish").status_code == 409
