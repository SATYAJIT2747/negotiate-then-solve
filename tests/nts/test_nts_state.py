import itertools

import pytest

from nts.kernel.schemas import Constraint, ConstraintType, Scope, Tier, Validity, When
from nts.kernel.solver import TimetableSolver
from nts.state import (
    ConstraintAdded,
    ConstraintRemoved,
    CorruptLog,
    InvalidChange,
    StaleDraft,
    Store,
    TimetableSet,
    bootstrap,
)


def away(teacher: str, day: str, week: int) -> Constraint:
    return Constraint(id=f"U-{teacher}-{day}-{week}", type=ConstraintType.UNAVAILABLE, hard=True,
                      tier=Tier.VERIFIED_UNAVAILABILITY, owner=teacher, scope=Scope(faculty=teacher),
                      when=When(days=[day]), valid=Validity(from_week=week, to_week=week))


def repair(store: Store, c: Constraint, week: int) -> TimetableSet:
    world = store.world
    result = TimetableSolver(world.instance, [*world.active(week), c], week=week,
                             baseline=world.timetable(week).assignment, time_limit=20).solve()
    assert result.ok
    return TimetableSet(week=week, assignment=result.assignment)


def busy_teacher(store: Store, day: str) -> str:
    inst, tt = store.world.instance, store.world.timetable()
    return next(inst.session_by_id[s].faculty for s, p in tt.assignment.items() if p.day == day)


def test_log_is_a_hash_chain(store):
    events = store.events
    assert events[0].change.kind == "created"
    assert all(b.parent == a.hash for a, b in itertools.pairwise(events))
    assert store.version == events[-1].hash
    assert store.world.timetable() is not None


def test_draft_is_private_until_commit(store):
    before = store.version
    t = busy_teacher(store, "Tue")
    c = away(t, "Tue", 7)
    draft = store.draft(actor=t, episode="ep-1")
    draft.add(ConstraintAdded(constraint=c))
    assert c.id in draft.world.constraints
    assert c.id not in store.world.constraints and store.version == before

    draft.add(repair(store, c, 7))
    world = store.commit(draft)
    assert world.version != before and c.id in world.constraints
    assert store.events[-1].actor == t and store.events[-1].episode == "ep-1"
    moved = world.timetable(7).assignment
    assert not any(p.day == "Tue" and world.instance.session_by_id[s].faculty == t for s, p in moved.items())
    assert world.timetable(8) is world.timetable(None)  # other weeks keep the semester timetable


def test_stale_draft_is_refused(store):
    a, b = store.draft(actor="x"), store.draft(actor="y")
    first = next(iter(store.world.constraints))
    a.add(ConstraintRemoved(id=first))
    store.commit(a)
    b.add(ConstraintRemoved(id=first))  # b still sees the old version, where it exists
    with pytest.raises(StaleDraft):
        store.commit(b)


def test_commit_rejects_a_double_booking(store):
    tt = dict(store.world.timetable().assignment)
    inst = store.world.instance
    a, b = [s.id for s in inst.sessions if s.duration == 1][:2]
    tt[b] = tt[a]
    draft = store.draft(actor="agent")
    draft.add(TimetableSet(week=3, assignment=tt))
    before = store.version
    with pytest.raises(InvalidChange) as e:
        store.commit(draft)
    assert any("double-booked" in p for p in e.value.problems)
    assert store.version == before


def test_commit_rejects_a_timetable_that_breaks_a_new_constraint(store):
    t = busy_teacher(store, "Tue")
    draft = store.draft(actor=t)
    draft.add(ConstraintAdded(constraint=away(t, "Tue", 7)))
    draft.add(TimetableSet(week=7, assignment=dict(store.world.timetable().assignment)))  # not repaired
    with pytest.raises(InvalidChange, match="violated"):
        store.commit(draft)


def test_commit_rejects_unknown_names(store):
    draft = store.draft(actor="agent")
    draft.add(ConstraintAdded(constraint=away("F-999", "Mon", 2)))
    with pytest.raises(InvalidChange, match="unknown faculty"):
        store.commit(draft)


def test_illegal_change_leaves_the_draft_as_it_was(store):
    draft = store.draft(actor="agent")
    with pytest.raises(ValueError):
        draft.add(ConstraintRemoved(id="no-such"))
    assert draft.changes == [] and draft.world is store.world


def test_reopen_and_history(solved, tmp_path):
    inst, cons, _ = solved
    path = tmp_path / "dept.jsonl"
    store = bootstrap(inst, cons, path=path, time_limit=20)
    start = store.version
    draft = store.draft(actor="agent")
    draft.add(ConstraintRemoved(id=cons[0].id))
    store.commit(draft)

    again = Store.open(path)
    assert again.version == store.version
    assert dict(again.world.timetable().assignment) == dict(store.world.timetable().assignment)
    assert cons[0].id in again.at(start).constraints and cons[0].id not in again.world.constraints
    with pytest.raises(FileExistsError):
        Store.create(inst, path=path)


def test_edited_log_is_detected(store, tmp_path):
    path = tmp_path / "dept.jsonl"
    path.write_text("".join(e.model_dump_json() + "\n" for e in store.events), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"actor":"system"', '"actor":"someone"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(CorruptLog):
        Store.open(path)
