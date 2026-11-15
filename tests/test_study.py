"""Human study: item sets, storage, analysis, and the endpoints a participant uses."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from evaluation.study import (
    Study,
    analyse,
    build_items,
    build_rule_items,
    rule_label,
    steps_done,
    judge_id,
    reply_label,
    write_ratings_csv,
)


def _row(scenario, config, n_msgs=2):
    facts = [{"id": "C-1", "text": "Dr. Rao needs Lab 2 on Tuesday."}, {"id": "OPT-A", "text": "Option A: Wednesday at 2 pm."}]
    claims = [{"text": "You both need Lab 2 on Tuesday.", "facts": ["C-1"], "supported": True},
              {"text": "Option A is on Friday.", "facts": ["OPT-A"], "supported": False}]
    mode = "grounded" if config == "ours-llm" else "free"
    replies = [{"decision": "accept", "choice": "A", "text": "A works."},
               {"decision": "counter", "counter_days": ["Fri"], "text": "Friday instead?"}]
    return {"scenario": scenario, "seed": 0,
            "messages": [f"{config} message {k} for {scenario}\n\nOptions:\nA) one\nB) two" for k in range(n_msgs)],
            "explanations": [{"to": "F-1", "mode": mode, "facts": facts, "claims": claims} for _ in range(n_msgs)],
            "replies": replies[:n_msgs]}


def _report(n=14):
    scen = [f"S-{i:03d}-{'contention' if i % 2 else 'deadlock'}" for i in range(1, n + 1)]
    return {"configs": {"ours-llm": {"rows": [_row(s, "ours-llm") for s in scen]},
                        "A2": {"rows": [_row(s, "A2") for s in scen[: n // 2]]}}}


def test_item_sets_pair_grounded_and_free_and_keep_every_rejected_claim():
    items = build_items(_report())
    claims = items["claims"]
    assert all(not c["machine"] for c in claims if c["claim"] == "Option A is on Friday.")
    assert sum(not c["machine"] for c in claims) == min(40, 14 * 2 + 7 * 2)  # every rejected claim, up to 40
    assert {r["machine"] for r in items["replies"]} == {"accept:A", "counter"}
    ratings = items["ratings"]
    assert len(ratings) == 7 * 2 * 2 + (14 - 7) * 2  # pairs, then unpaired grounded messages (fewer than 110 here)
    pilot = [r for r in ratings if r["pilot"]]
    assert {r["mode"] for r in pilot} == {"grounded", "free"} and len(pilot) % 2 == 0
    assert ratings[0]["id"] == judge_id("ours-llm", ratings[0]["scenario"], ratings[0]["message"])


def test_reply_labels():
    assert reply_label({"decision": "accept", "choice": " b"}) == "accept:B"
    assert reply_label({"decision": "propose"}) == "other" and reply_label({"decision": "reject"}) == "reject"


def test_storage_queue_and_analysis(tmp_path):
    s = Study(tmp_path)
    s.save_items(build_items(_report()))
    t1, t2 = s.add_participants("team", 2)
    (p1,) = s.add_participants("pilot", 1, persona="F-102")
    assert (t1, t2, p1) == ("T-1", "T-2", "P-01")
    assert s.queue(t1, "team", "claims") != s.queue(t2, "team", "claims")  # each rater has an order of their own
    assert all(i["pilot"] for i in s.queue(p1, "pilot", "ratings"))
    for code in (t1, t2):
        for it in s.items()["claims"]:
            s.label(code, "claims", it["id"], {"supported": it["machine"]})
    first = s.items()["claims"][0]
    s.label(t1, "claims", first["id"], {"supported": not first["machine"]})  # a change of mind: the latest counts
    s.label(t1, "claims", first["id"], {"supported": first["machine"]})
    for it in s.queue(p1, "pilot", "ratings"):
        s.label(p1, "ratings", it["id"], {"clarity": 5 if it["mode"] == "grounded" else 3, "acceptability": 4})
    s.live(p1, "F-102", "understood", value="yes")
    s.live(p1, "F-102", "reply", confirmed=False)
    a = analyse(s, human_only=False)
    assert a["claims"]["human_vs_verifier"][t1]["kappa"] == 1.0 and a["claims"]["human_vs_verifier"][t1]["ok"]
    assert a["claims"]["human_vs_human"][f"{t1} vs {t2}"]["kappa"] == 1.0
    h3 = a["ratings"]["h3_pilot"]
    assert h3["pairs"] == len(s.queue(p1, "pilot", "ratings")) // 2
    assert h3["clarity"]["grounded"] == 5 and h3["clarity"]["free"] == 3
    assert a["live"]["understood"] == {"yes": 1} and a["live"]["reply_reading_confirmed"] == 0
    assert s.progress(p1, "pilot") == {"ratings": {"done": len(s.queue(p1, "pilot", "ratings")),
                                                  "total": len(s.queue(p1, "pilot", "ratings"))}}
    assert write_ratings_csv(s, tmp_path / "h.csv", human_only=False) == 0  # pilot ratings are not team ratings


# ---------------------------------------------------------------------------
# Endpoints (real pipeline, offline parser)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    from web.app import World, create_web_app

    root = tmp_path_factory.mktemp("study")
    os.environ["NTS_STUDY_DIR"] = str(root)
    try:
        world = World(parser_mode="offline", seed_history=False, deadline=60)
    finally:
        del os.environ["NTS_STUDY_DIR"]
    world.study.save_items(build_items(_report()))
    return TestClient(create_web_app(world)), world


def test_labelling_is_blind_and_needs_consent(web):
    c, world = web
    (code,) = world.study.add_participants("team", 1)
    h = {"X-Study": code}
    assert c.get("/api/study/me", headers={"X-Study": "T-99"}).status_code == 401
    assert c.get("/api/study/next/claims", headers=h).status_code == 403  # consent first
    assert c.post("/api/study/consent", headers=h).json()["consented"]
    nxt = c.get("/api/study/next/claims", headers=h).json()
    assert set(nxt["item"]) == {"id", "claim", "facts"} and nxt["done"] == 0  # no config, no machine label
    assert c.post("/api/study/labels", json={"task": "claims", "item": nxt["item"]["id"], "value": {"supported": "yes"}},
                  headers=h).status_code == 422
    assert c.post("/api/study/labels", json={"task": "claims", "item": nxt["item"]["id"], "value": {"supported": True}},
                  headers=h).json()["ok"]
    assert c.get("/api/study/next/claims", headers=h).json()["done"] == 1
    rating = c.get("/api/study/next/ratings", headers=h).json()["item"]
    assert set(rating) == {"id", "text"}
    assert c.post("/api/study/labels", json={"task": "ratings", "item": rating["id"],
                                             "value": {"clarity": 6, "acceptability": 3}}, headers=h).status_code == 422


@pytest.mark.slow
def test_pilot_session_in_the_portal(web):
    """Practice clash -> own request -> negotiator's message -> reply in own words -> rating."""
    c, _world = web
    coord = {"X-User": "C-TT"}
    code = c.post("/api/study/participants", json={"kind": "pilot", "n": 1}, headers=coord).json()["codes"][0]
    h = {"X-Study": code}
    assert c.get("/api/study/me", headers=h).json()["persona"] == "F-102"
    c.post("/api/study/consent", headers=h)
    assert c.get("/api/study/next/claims", headers=h).status_code == 403  # pilots only rate
    assert c.post("/api/study/practice", headers=h | {"X-User": "F-101"}).status_code == 403  # not their persona
    # the live session runs in the demo department only: F-102 may be someone else in another world
    assert c.post("/api/study/practice", headers=h | {"X-User": "F-102", "X-World": "campus"}).status_code == 409
    task = c.post("/api/study/practice", headers=h | {"X-User": "F-102"}).json()
    assert "routers" in task["task"]
    rid = c.post("/api/requests", json={"text": f"My {task['course']} practical needs routers too; it must be on "
                                                "Tuesday afternoon."}, headers={"X-User": "F-102"}).json()["id"]
    end, item = time.time() + 120, None
    while time.time() < end and item is None:
        item = next((i for i in c.get("/api/inbox", headers={"X-User": "F-102"}).json() if i["case"] == rid), None)
        time.sleep(0.3)
    assert item, "the negotiator never wrote to the participant"
    letter = item["message"]["offers"][0]["key"]
    peek = c.post(f"/api/inbox/{item['id']}/reply-text", json={"text": f"Option {letter} is fine with me.",
                                                              "preview": True}, headers={"X-User": "F-102"}).json()
    assert peek["reply"]["decision"] == "accept" and peek["reading"].startswith(f"You accept option {letter}")
    assert c.post(f"/api/inbox/{item['id']}/reply-text", json={"text": "x"},
                  headers={"X-User": "F-101"}).status_code == 403  # addressed to someone else
    sent = c.post(f"/api/inbox/{item['id']}/reply-text", json={"text": f"Option {letter} is fine with me.",
                                                              "preview": False}, headers={"X-User": "F-102"}).json()
    assert sent["item"]["reply"]["choice"] == letter
    for body in ({"kind": "reply", "item": item["id"], "confirmed": True},
                 {"kind": "rating", "item": item["id"], "value": {"clarity": 4, "acceptability": 5}},
                 {"kind": "understood", "case": rid, "value": "partly"}):
        assert c.post("/api/study/live", json=body, headers=h | {"X-User": "F-102"}).json()["ok"]
    live = c.get("/api/study/admin", headers=coord).json()["analysis"]["live"]
    assert live["reply_reading_confirmed"] == 1 and live["clarity"] == 4 and live["understood"] == {"partly": 1}
    assert c.get("/api/study/admin", headers={"X-User": "F-102"}).status_code == 403
    assert {r["world"] for r in c.get("/api/study/export", headers=coord).json()["live"]} == {"demo"}


def test_revoke_keeps_or_deletes_answers(tmp_path):
    s = Study(tmp_path)
    s.save_items(build_items(_report()))
    t1, t2 = s.add_participants("team", 2)
    for code in (t1, t2):
        for it in s.items()["claims"]:
            s.label(code, "claims", it["id"], {"supported": it["machine"]})
    s.live(t2, "F-102", "understood", value="yes")

    s.revoke(t1)  # code stops working, answers kept but out of every result
    assert s.get(t1) is None and s.get(t2)
    assert t1 not in analyse(s, human_only=False)["claims"]["human_vs_verifier"]
    assert s.labels()[("claims", t1)]
    s.restore(t1)
    assert t1 in analyse(s, human_only=False)["claims"]["human_vs_verifier"]

    s.revoke(t2, delete_answers=True)  # a withdrawal: erased from disk, cannot come back
    assert ("claims", t2) not in s.labels() and not s.live_rows()
    assert ("claims", t1) in s.labels()
    with pytest.raises(ValueError):
        s.restore(t2)
    assert s.add_participants("team", 1) == ["T-3"]  # codes are never reused


def test_team_work_is_shared_two_raters_per_item(tmp_path):
    s = Study(tmp_path)
    s.save_items(build_items(_report()))
    team = s.add_participants("team", 4)
    claims = s.items()["claims"]
    shares = {c: s.assigned(c, "claims") for c in team}
    assert all(sum(i["id"] in sh for sh in shares.values()) == 2 for i in claims)  # every item: two raters
    assert max(map(len, shares.values())) - min(map(len, shares.values())) <= 2  # even shares, about half each
    assert all(len(sh) < len(claims) for sh in shares.values())
    for c in team:  # everyone finishes their share
        for it in s.queue(c, "team", "claims"):
            s.label(c, "claims", it["id"], {"supported": it["machine"]})
    assert s.coverage()["claims"]["double"] == len(claims)
    assert all(not [i for i in s.queue(c, "team", "claims") if i["id"] not in s.labels()[("claims", c)]] for c in team)
    a = analyse(s, human_only=False)
    assert a["claims"]["human_vs_human"]["all raters (pooled)"]["n"] == len(claims)
    assert a["coverage"]["claims"] == {"items": len(claims), "double": len(claims), "single": 0}


def test_clear_needs_the_phrase_and_keeps_the_items(web):
    c, world = web
    world.study.add_participants("team", 2)
    coord = {"X-User": "C-TT"}
    assert c.post("/api/study/clear", json={"confirm": "yes"}, headers=coord).status_code == 422
    assert c.post("/api/study/clear", json={"confirm": "delete all study data"}, headers={"X-User": "F-102"}).status_code == 403
    removed = c.post("/api/study/clear", json={"confirm": "Delete all study data"}, headers=coord).json()["removed"]
    assert removed["participants"] >= 2
    assert world.study.participants() == {} and world.study.items() is not None  # the material stays
    assert world.study.add_participants("team", 1) == ["T-1"]


# ---------------------------------------------------------------------------
# Rule answers (team) and the pilot's tour of the system
# ---------------------------------------------------------------------------


class _Rule:
    def __init__(self, rid):
        self.id, self.title, self.text = rid, f"Title of {rid}", f"Text of {rid}."


def _rag_report():
    def row(cond, i, gold, **v):
        return {"key": f"{cond}/Q-{i}", "query": f"question {i}", "answer": f"answer {i}", "gold": gold,
                "no_rule": False, "abstains": False, "agrees": None, **v}
    out = {}
    for cond in ("rag", "no-rag"):
        rows = [row(cond, i, ["LAB-1"], agrees=i % 2 == 0) for i in range(30)]
        rows += [row(cond, 100 + i, [], abstains=i % 3 == 0) for i in range(30)]
        out[cond] = {"rows": rows}
    return {"conditions": out}


def test_rule_answers_carry_the_judges_verdict_and_the_rule_text():
    assert rule_label({"no_rule": True}) == "declines" and rule_label({"abstains": True}) == "declines"
    assert rule_label({"agrees": True}) == "consistent" and rule_label({"agrees": None}) == "not_consistent"
    items = build_rule_items(_rag_report(), {"LAB-1": _Rule("LAB-1")})
    assert len(items) == 60 and {i["condition"] for i in items} == {"rag", "no-rag"}
    assert {i["machine"] for i in items} == {"declines", "consistent", "not_consistent"}
    answerable = [i for i in items if i["rules"]]
    assert answerable and answerable[0]["rules"][0]["text"] == "Text of LAB-1."
    assert len(answerable) == 30  # half answerable, half not


def test_rules_task_is_blind_and_scored_against_the_judge(tmp_path):
    s = Study(tmp_path)
    items = build_items(_report())
    items["rules"] = build_rule_items(_rag_report(), {"LAB-1": _Rule("LAB-1")})
    s.save_items(items)
    t1, t2 = s.add_participants("team", 2)
    assert "rules" in s.tasks_for("team") and s.tasks_for("pilot") == ["ratings"]
    for code in (t1, t2):
        for it in s.items()["rules"]:
            s.label(code, "rules", it["id"], {"label": it["machine"]})
    a = analyse(s, human_only=False)
    assert a["rules"]["human_vs_judge"][t1]["kappa"] == 1.0 and a["rules"]["human_vs_human"][f"{t1} vs {t2}"]["kappa"] == 1.0


def test_tour_steps_are_done_by_their_checks(tmp_path):
    s = Study(tmp_path)
    s.save_items(build_items(_report()))
    (p,) = s.add_participants("pilot", 1, persona="F-102")
    s.live(p, "F-102", "answer", step="timetable", value="yes")
    s.live(p, "F-102", "understood", step="click", value="partly")
    s.live(p, "F-102", "survey", value={"understood": 4, "timetable": 5, "rules": 3, "clash": 4, "next": 4, "use": 2})
    assert steps_done(s.live_rows(), p) == {"timetable", "click", "survey"}
    live = analyse(s, human_only=False)["live"]
    assert live["timetable_answers"] == {"yes": 1} and live["rules_answers"] == {}
    assert live["survey"]["n"] == 1 and live["survey"]["use"] == 2
    assert live["steps_done"][p] == ["click", "survey", "timetable"]


def test_tour_checks_are_validated(web):
    c, _world = web
    coord = {"X-User": "C-TT"}
    code = c.post("/api/study/participants", json={"kind": "pilot", "n": 1}, headers=coord).json()["codes"][0]
    h = {"X-Study": code, "X-User": "F-102"}
    c.post("/api/study/consent", headers=h)
    me = c.get("/api/study/me", headers=h).json()
    assert [s["step"] for s in me["live_steps"]] == ["timetable", "rules", "preference", "practice", "click", "survey"]
    assert me["steps_done"] == [] and len(me["survey"]) == 6
    bad = [{"kind": "answer", "step": "practice", "value": "yes"}, {"kind": "answer", "step": "rules", "value": "maybe"},
           {"kind": "survey", "value": {"understood": 6}}]
    for body in bad:
        assert c.post("/api/study/live", json=body, headers=h).status_code == 422
    assert c.post("/api/study/live", json={"kind": "answer", "step": "rules", "n": 3, "value": "partly"}, headers=h).json()["ok"]
    full = {k["key"]: 4 for k in me["survey"]}
    assert c.post("/api/study/live", json={"kind": "survey", "value": full}, headers=h).json()["ok"]
    assert c.get("/api/study/me", headers=h).json()["steps_done"] == ["rules", "survey"]


def test_only_paced_labels_count_as_human():
    from evaluation.study import paced

    def row(code, item, at):
        return {"code": code, "task": "claims", "item": item, "at": f"2026-11-15T09:00:{at:02d}+00:00",
                "value": {"supported": True}}

    # a person (10 s apart), then a burst one second apart; another code's labels are judged on their own
    rows = [row("T-1", "a", 0), row("T-1", "b", 10), row("T-1", "c", 20), row("T-1", "d", 21), row("T-1", "e", 22),
            row("T-2", "a", 21)]
    assert [(r["code"], r["item"]) for r in paced(rows)] == [("T-1", "a"), ("T-1", "b"), ("T-2", "a")]
