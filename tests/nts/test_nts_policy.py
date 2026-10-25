import json
import random

from test_nts_talk import last_result, negotiation_steps

from nts.agents import COORDINATOR_V2, COORDINATOR_V3
from nts.eval import run_suite
from nts.eval.scenarios import negotiate, rules_question, weekend
from nts.fairness import given_way, standing
from nts.harness import Harness, approve
from nts.models import ScriptedModel, call, say
from nts.policy import Handbook
from nts.state import Store
from nts.telemetry import MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry, Scope

HANDBOOK = Handbook.load()


def test_the_handbook_is_searched_and_cited():
    assert len(HANDBOOK.rules) == 16
    assert HANDBOOK.search("can I teach in the lunch hour, 1 pm?")[0].id == "P-LUNCH"
    assert HANDBOOK.search("classes on a Saturday")[0].id == "P-HOURS"
    assert HANDBOOK.by_id("[p-lunch]").cite() == "§2.2 Common lunch break [P-LUNCH]"
    assert HANDBOOK.search("zzz qqq") == []


def test_the_rules_tool_is_only_for_agents_with_the_policy_scope(store):
    names = lambda a: [t["function"]["name"] for t in Registry(ALL_TOOLS).specs(a.scopes)]
    assert "rules" in names(COORDINATOR_V3) and "rules" not in names(COORDINATOR_V2)
    assert COORDINATOR_V3.scopes == COORDINATOR_V2.scopes | {Scope.POLICY}

    model = ScriptedModel([call("rules", question="lunch hour"), say("No: rule 2.2 [P-LUNCH] keeps 1 to 2 pm free.")])
    out = Harness(Registry(ALL_TOOLS), model, handbook=HANDBOOK).run(COORDINATOR_V3, "lunch?", store, "coordinator")
    assert last_result(model.seen[1])[0]["rule"].endswith("[P-LUNCH]")
    model = ScriptedModel([call("rules", question="lunch hour"), say("I cannot look it up.")])
    out = Harness(Registry(ALL_TOOLS), model).run(COORDINATOR_V3, "lunch?", store, "coordinator")
    assert out.calls[0].error.code == "no_handbook"


def test_policy_scenarios_want_the_rule_cited(store):
    q = rules_question(store.world, random.Random(0), 5)
    rid = q.name.split("-", 1)[1].rsplit("-F", 1)[0]
    rule = HANDBOOK.by_id(rid)
    good = ScriptedModel([call("rules", question=q.request), say(f"See {rule.cite()}: {rule.text[:60]}")])
    vague = ScriptedModel([say("I think that is fine.")])
    verdicts = [run_suite([q], m, store, COORDINATOR_V3, Tracer([MemorySink()]), handbook=HANDBOOK)[0]
                for m in (good, vague)]
    assert verdicts[0].passed and not verdicts[1].passed and "does not cite" in verdicts[1].failures[0]

    w = weekend(store.world, random.Random(1), 5)
    out = run_suite([w], ScriptedModel([call("decline", message="Not on Saturdays: rule 2.1 [P-HOURS].")]), store,
                    COORDINATOR_V3, Tracer([MemorySink()]))[0]
    assert out.passed, out.failures


def test_an_agreed_waiver_is_a_concession_in_the_log(store, tmp_path):
    sc = negotiate(store.world, random.Random(1), 5)
    owner = next(iter(sc.cast.teachers))
    path = tmp_path / "dept.jsonl"
    disk = Store.create(store.world.instance, [e.change for e in store.events[1:]], path=path)
    draft = disk.draft("system")
    for ch in sc.setup:
        draft.add(ch)
    disk.commit(draft)

    steps = negotiation_steps(sc, disk.world, then=[call("check_draft"), call("submit", message="Moved.")])
    tracer = Tracer([MemorySink()])
    out = Harness(Registry(ALL_TOOLS), ScriptedModel(steps), tracer).run(COORDINATOR_V2, sc.request, disk, sc.actor,
                                                                          sc.week, people=sc.cast)
    assert out.status == "submitted" and given_way(disk.world, owner) == 0
    approve(disk, out, tracer)
    assert given_way(disk.world, owner) == 1
    e = disk.world.concessions[0]
    assert (e.stakeholder, e.conceded_to, e.week) == (owner, sc.actor, int(sc.name.rsplit("-w", 1)[1]))
    assert 0 < e.credit < 1  # a waiver for one week is a partial concession of an operational rule

    rows, spread = standing(Store.open(path).world)  # the ledger is in the log on disk
    top = rows[0]
    assert top["teacher"] == disk.world.instance.faculty_by_id[owner].name and top["given_way"] == 1
    assert spread["gini"] > 0 and spread["total"] == round(e.credit, 3)


def test_a_clash_says_how_often_the_owner_has_given_way(store):
    sc = negotiate(store.world, random.Random(1), 5)
    model = ScriptedModel(negotiation_steps(sc, store.world, then=[])[:2] + [say("I will ask.")])
    run_suite([sc], model, store, COORDINATOR_V2, Tracer([MemorySink()]))
    clash = json.loads(next(m.content for m in reversed(model.seen[2]) if m.role == "tool"))
    theirs = next(r for r in clash["clashing"] if "number" in r)
    assert theirs["has_given_way"] == "0 times this term"
