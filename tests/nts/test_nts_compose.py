from nts.agents import COORDINATOR_V5, COORDINATOR_V6
from nts.harness import Harness
from nts.kernel.schemas import Constraint, ConstraintType, Tier
from nts.kernel.schemas import Scope as Who
from nts.models import ScriptedModel, call, say
from nts.tools import ALL_TOOLS, Registry
from nts.tools.names import FULL_DAY, hour


def episode(store, steps, actor, request="a request", week=5):
    return Harness(Registry(ALL_TOOLS), ScriptedModel(steps)).run(COORDINATOR_V6, request, store, actor, week)


def busiest(store, week=5):
    """The teacher with the most classes on one day, that day, and their classes there."""
    inst, tt = store.world.instance, store.world.timetable(week)
    by: dict = {}
    for sid, p in tt.assignment.items():
        by.setdefault((inst.session_by_id[sid].faculty, p.day), []).append(sid)
    (teacher, day), sids = max(by.items(), key=lambda kv: len(kv[1]))
    return teacher, day, sids


def test_only_v6_has_the_compose_tools():
    names = {s["function"]["name"] for s in Registry(ALL_TOOLS).specs(COORDINATOR_V6.scopes)}
    older = {s["function"]["name"] for s in Registry(ALL_TOOLS).specs(COORDINATOR_V5.scopes)}
    assert names - older == {"move_to_window", "limit_load", "relate_classes", "choose", "set_weeks"}


def test_older_kinds_serialise_as_before():  # the event log's hashes depend on it
    c = Constraint(id="x", type=ConstraintType.AVOID, hard=False, tier=Tier.PREFERENCE, scope=Who(faculty="F-1"))
    assert not {"measure", "sessions", "others", "relation", "amount", "any_other", "negate", "members"} & set(c.model_dump())


def test_a_limit_on_classes_a_day_holds_in_every_week_after(store):
    teacher, day, sids = busiest(store)
    cap = len(sids) - 1
    out = episode(store, [call("limit_load", measure="classes_per_day", n=cap, teacher="me", from_week=5),
                          call("check_draft"), call("submit", message="Limited.")], teacher)
    assert out.status == "submitted", out.calls
    assert f"at most {cap} classes a day" in out.calls[0].content["added"][0]
    world = store.commit(out.draft)
    inst = world.instance
    for week in (5, 9):
        per_day: dict = {}
        for sid, p in world.timetable(week).assignment.items():
            if inst.session_by_id[sid].faculty == teacher:
                per_day[p.day] = per_day.get(p.day, 0) + 1
        assert max(per_day.values()) <= cap


def test_two_moves_made_a_choice_put_the_class_on_one_of_them(store):
    teacher, day, sids = busiest(store)
    inst, tt = store.world.instance, store.world.timetable(5)
    s = next(inst.session_by_id[x] for x in sids if inst.session_by_id[x].duration == 1)
    p = tt.assignment[s.id]
    a, b = [d for d in inst.calendar.days if d != p.day][:2]
    pick = dict(course=inst.course_title(s.course), kind=s.kind.value, on_day=FULL_DAY[p.day], at_hour=hour(p.slot))
    out = episode(store, [call("move_class", **pick, to_day=FULL_DAY[a], from_week=6, to_week=6),
                          call("move_class", **pick, to_day=FULL_DAY[b], from_week=6, to_week=6),
                          call("choose", changes=[1, 2], ranked=True), call("list_draft"),
                          call("check_draft"), call("submit", message="Moved.")], teacher)
    assert out.status == "submitted", out.calls
    listed = out.calls[3].content  # the two moves are one change now
    assert len(listed) == 1 and listed[0]["rules"][0].startswith("one of:")
    assert listed[0]["rules"][1].startswith("in order of preference:")
    world = store.commit(out.draft)
    assert world.timetable(6).assignment[s.id].day in (a, b)


def test_set_weeks_keeps_a_rule_to_even_weeks(store):
    teacher, day, _ = busiest(store)
    out = episode(store, [call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]], from_week=5),
                          call("set_weeks", change=1, every="even"), call("check_draft"),
                          call("submit", message="Done.")], teacher)
    assert out.status == "submitted", out.calls
    world = store.commit(out.draft)
    inst = world.instance

    def there(week):
        return [s for s, p in world.timetable(week).assignment.items()
                if p.day == day and inst.session_by_id[s].faculty == teacher]

    assert not there(6) and not there(8) and there(7)


def test_classes_kept_on_different_days(store):
    inst, tt = store.world.instance, store.world.timetable(5)
    by: dict = {}
    for s in inst.sessions:
        if s.id in tt.assignment and s.kind.value == "lecture":
            by.setdefault((s.faculty, s.course), []).append(s)
    (teacher, course), lectures = max(by.items(), key=lambda kv: len(kv[1]))
    out = episode(store, [call("relate_classes", classes={"course": inst.course_title(course), "kind": "lecture"},
                               relation="different_day", from_week=5),
                          call("check_draft"), call("submit", message="Done.")], teacher)
    assert out.status == "submitted", out.calls
    world = store.commit(out.draft)
    days = [world.timetable(7).assignment[s.id].day for s in lectures]
    assert len(set(days)) == len(days)
