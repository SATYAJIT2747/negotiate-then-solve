"""Conversations: the requests of a thread, the agent remembering the earlier ones, and who sees what."""

from fastapi.testclient import TestClient

from nts.agents import COORDINATOR_V3
from nts.api import Service, create_app
from nts.harness import Harness
from nts.harness.loop import STUCK
from nts.models import ScriptedModel, call, say
from nts.telemetry import MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry
from nts.tools.names import FULL_DAY


def app(store, steps):
    service = Service(store, ScriptedModel(steps), Tracer([MemorySink()]))
    return create_app({"demo": service}), service


def signed_in(application, name):
    c = TestClient(application)
    assert c.post("/api/login", json={"dept": "demo", "name": name, "password": ""}).status_code == 200
    return c


def away(day, week=6):
    return [call("constrain", rule="unavailable", teacher="me", days=[FULL_DAY[day]], from_week=week, to_week=week,
                 reason="a conference"), call("check_draft"), call("submit", message="Your classes that day move.")]


def test_a_follow_up_sees_the_earlier_messages_of_its_thread(store):
    t = store.world.instance.faculty[0]
    a, service = app(store, [say("You teach Algorithms."), say("On Monday at 10 am.")])
    me = signed_in(a, t.name)
    first = me.post("/api/episodes", json={"request": "what do I teach?", "week": 5}).json()
    service.wait(first["id"])
    second = me.post("/api/episodes", json={"request": "and when?", "week": 5, "thread": first["thread"]}).json()
    service.wait(second["id"])
    assert second["thread"] == first["thread"] == first["id"]
    sent = [(m.role, m.content) for m in service.harness.model.seen[1][1:]]
    assert sent == [("user", "what do I teach?"), ("assistant", "You teach Algorithms."), ("user", "and when?")]
    chat = me.get(f"/api/threads/{first['thread']}").json()
    assert [e["request"] for e in chat["episodes"]] == ["what do I teach?", "and when?"] and chat["mine"]
    assert [e["type"] for e in chat["episodes"][1]["events"]] == ["model", "end"]
    assert [(t["title"], t["messages"]) for t in me.get("/api/threads").json()] == [("what do I teach?", 2)]


def test_a_message_while_the_agent_waits_for_you_is_your_answer(store):
    t = store.world.instance.faculty[0]
    a, service = app(store, [call("ask_requester", question="Which class?"), say("Noted: the Monday one.")])
    me = signed_in(a, t.name)
    ep = me.post("/api/episodes", json={"request": "move my class", "week": 5}).json()
    service.wait(ep["id"])
    got = me.post("/api/episodes", json={"request": "The Monday one.", "thread": ep["thread"]}).json()
    service.wait(ep["id"])
    assert got["id"] == ep["id"]  # the same request goes on: no new one
    done = service.get(ep["id"])
    assert done.status == "answered" and done.outcome.asks[0][1].text == "The Monday one."


def test_a_new_message_replaces_a_draft_the_office_has_not_decided(store):
    inst = store.world.instance
    t, days = inst.faculty[1], inst.calendar.days
    a, service = app(store, [*away(days[0]), *away(days[1])])
    me, office = signed_in(a, t.name), signed_in(a, "office")
    first = me.post("/api/episodes", json={"request": f"away {days[0]} week 6", "week": 5}).json()
    service.wait(first["id"])
    second = me.post("/api/episodes", json={"request": f"no, {days[1]}", "week": 5, "thread": first["thread"]}).json()
    service.wait(second["id"])
    assert service.get(first["id"]).status == "replaced" and service.get(second["id"]).status == "submitted"
    assert office.post(f"/api/episodes/{first['id']}/approve").status_code == 409
    assert "Drafted:" in service.remembered(service.get(first["id"]))  # what the agent remembers of it
    assert office.post(f"/api/episodes/{second['id']}/approve").json()["status"] == "approved"


def test_no_one_else_writes_in_your_thread(store):
    t, other = store.world.instance.faculty[:2]
    a, service = app(store, [say("Hello.")])
    ep = signed_in(a, t.name).post("/api/episodes", json={"request": "hi", "week": 5}).json()
    service.wait(ep["id"])
    for who in (signed_in(a, other.name), signed_in(a, "office")):
        assert who.post("/api/episodes", json={"request": "x", "thread": ep["thread"]}).status_code == 404
        assert who.get(f"/api/threads/{ep['thread']}").status_code == 404


def test_the_office_sees_only_what_reaches_it(store):
    """A teacher's questions stay theirs; a change sent for approval reaches the office, without the rest."""
    inst = store.world.instance
    t = inst.faculty[1]
    a, service = app(store, [say("You teach Algorithms."), *away(inst.calendar.days[0])])
    me, office = signed_in(a, t.name), signed_in(a, "office")
    q = me.post("/api/episodes", json={"request": "what do I teach?", "week": 5}).json()
    service.wait(q["id"])
    assert office.get("/api/threads").json() == [] and office.get(f"/api/episodes/{q['id']}").status_code == 404
    ch = me.post("/api/episodes", json={"request": "away week 6", "week": 5, "thread": q["thread"]}).json()
    service.wait(ch["id"])
    [seen] = office.get("/api/threads").json()
    assert seen["actor"] == t.name and not seen["mine"]
    assert [e["request"] for e in office.get(f"/api/threads/{q['thread']}").json()["episodes"]] == ["away week 6"]


def test_a_stuck_request_says_what_it_needs(store):
    t = store.world.instance.faculty[0]
    out = Harness(Registry(ALL_TOOLS), ScriptedModel([call("list_draft")] * 4), Tracer([MemorySink()])).run(
        COORDINATOR_V3, "a class on monday", store, t.id, 5)
    assert out.status == "stuck" and out.text == STUCK


def test_an_offer_before_any_check_is_told_the_order(store):
    t = store.world.instance.faculty[0]
    sink = MemorySink()
    Harness(Registry(ALL_TOOLS), ScriptedModel([call("make_offer", rule=1, to="Dr. Iyer", message="please"),
                                                say("ok")]), Tracer([sink])).run(COORDINATOR_V3, "x", store, t.id, 5)
    err = next(s for s in sink.spans if s.kind == "tool_call").attrs["result"]["error"]
    assert "check_draft" in err["message"] and "add_class" in err["message"]


def test_two_tabs_of_one_browser_can_be_two_people(store):
    """The page keeps its session per tab: signing in as the office in a second tab leaves the first a teacher."""
    t = store.world.instance.faculty[0]
    a, service = app(store, [say("Hello.")])
    browser = TestClient(a)  # one cookie jar, as one browser
    first = browser.post("/api/login", json={"dept": "demo", "name": t.name, "password": ""}).json()["token"]
    second = browser.post("/api/login", json={"dept": "demo", "name": "office", "password": ""}).json()["token"]
    tab = lambda token: {"Authorization": f"Bearer {token}"}  # noqa: E731
    assert browser.get("/api/me", headers=tab(first)).json()["name"] == t.name
    assert browser.get("/api/me", headers=tab(second)).json()["role"] == "office"
    ep = browser.post("/api/episodes", headers=tab(first), json={"request": "hi", "week": 5}).json()
    service.wait(ep["id"])
    assert browser.get(f"/api/episodes/{ep['id']}/events?token={first}").status_code == 200
    browser.post("/api/logout", headers=tab(second))
    assert browser.get("/api/me", headers=tab(first)).status_code == 200  # signing out one tab leaves the other
    assert browser.get("/api/me", headers=tab(second)).status_code == 401


def test_a_question_to_you_comes_with_answers_to_pick(store):
    t = store.world.instance.faculty[0]
    a, service = app(store, [call("ask_requester", question="Monday 9 am or 11 am is free. Would either work?"),
                             call("ask_requester", question="Which class do you mean?"), say("Done.")])
    me = signed_in(a, t.name)
    ep = me.post("/api/episodes", json={"request": "an extra class on Monday", "week": 5}).json()
    service.wait(ep["id"])
    assert me.get(f"/api/episodes/{ep['id']}").json()["pending"]["options"] == ["Monday 9 am", "Monday 11 am",
                                                                                 "None of these"]
    me.post("/api/episodes", json={"request": "Monday 9 am", "thread": ep["thread"]})  # a picked option is the answer
    service.wait(ep["id"])
    mine = me.get(f"/api/episodes/{ep['id']}").json()["pending"]["options"]  # "which class?": your own classes
    inst = store.world.instance
    assert mine == sorted({f"{inst.course_title(s.course)} {s.kind.value}" for s in inst.sessions if s.faculty == t.id})


def test_the_office_inbox_decides_and_writes_to_the_teacher(store, tmp_path):
    from nts.telemetry import JsonlSink

    inst = store.world.instance
    t, days = inst.faculty[1], inst.calendar.days
    service = Service(store, ScriptedModel([*away(days[0]), call("escalate", message="needs the dean")]),
                      Tracer([JsonlSink(tmp_path)]))
    a = create_app({"demo": service})
    me, office = signed_in(a, t.name), signed_in(a, "office")
    change = me.post("/api/episodes", json={"request": "away week 6", "week": 5}).json()
    service.wait(change["id"])
    passed = me.post("/api/episodes", json={"request": "a new course", "week": 5}).json()
    service.wait(passed["id"])
    inbox = {r["id"]: r for r in office.get("/api/office/requests").json()}
    assert inbox[change["id"]]["needs_office"] and inbox[passed["id"]]["needs_office"]
    assert inbox[passed["id"]]["status"] == "escalated"

    done = office.post(f"/api/episodes/{change['id']}/approve", json={"note": "Enjoy the conference."}).json()
    assert done["status"] == "approved" and not done["needs_office"]
    assert office.post(f"/api/episodes/{passed['id']}/note", json={"note": "Please mail the dean."}).status_code == 200
    assert not any(r["needs_office"] for r in office.get("/api/office/requests").json())
    assert me.post(f"/api/episodes/{passed['id']}/note", json={"note": "x"}).status_code == 403  # only the office

    seen = me.get(f"/api/threads/{change['thread']}").json()["episodes"][0]
    assert [n["text"] for n in seen["notes"]] == ["Enjoy the conference."]
    assert [e["type"] for e in seen["events"]][-2:] == ["decision", "note"]
    again = Service(store, ScriptedModel([]), Tracer([MemorySink()]))
    again.restore(tmp_path)  # the office's messages outlive a restart
    assert [n["text"] for n in again.get(passed["id"]).notes] == ["Please mail the dean."]


def test_the_teacher_is_told_what_the_office_decided(store, tmp_path):
    from nts.telemetry import JsonlSink

    inst = store.world.instance
    t, days = inst.faculty[1], inst.calendar.days
    service = Service(store, ScriptedModel([*away(days[0]), *away(days[1])]), Tracer([JsonlSink(tmp_path)]))
    a = create_app({"demo": service})
    me, office = signed_in(a, t.name), signed_in(a, "office")
    yes = me.post("/api/episodes", json={"request": "away week 6", "week": 5}).json()
    service.wait(yes["id"])
    office.post(f"/api/episodes/{yes['id']}/approve", json={"note": "Enjoy it."})
    no = me.post("/api/episodes", json={"request": "away again", "week": 5}).json()
    service.wait(no["id"])
    office.post(f"/api/episodes/{no['id']}/reject", json={"note": "Exams that day."})
    told = me.get("/api/notices").json()  # newest first
    assert [n["kind"] for n in told if n["mine"]] == ["rejected", "office", "approved"]
    assert "Exams that day." in told[0]["text"] and told[0]["thread"] == no["thread"]
    assert told[-1]["request"] == "away week 6" and "not available" in told[-1]["text"]
    again = Service(store, ScriptedModel([]), Tracer([MemorySink()]))
    again.restore(tmp_path)
    assert [n["kind"] for n in again.notices(t.id) if n["mine"]] == ["rejected", "office", "approved"]


def test_a_follow_up_is_told_to_answer_the_latest_message_and_what_is_in_the_timetable(store):
    inst = store.world.instance
    t = inst.faculty[1]
    a, service = app(store, [*away(inst.calendar.days[0]), say("You teach two courses.")])
    me = signed_in(a, t.name)
    first = me.post("/api/episodes", json={"request": "away week 6", "week": 5}).json()
    service.wait(first["id"])
    second = me.post("/api/episodes", json={"request": "what do I teach?", "week": 5, "thread": first["thread"]}).json()
    service.wait(second["id"])
    sent = service.harness.model.seen[-1]
    assert "Answer the latest message" in sent[0].content and "Your draft starts empty" in sent[0].content
    assert "Withdrawn by a later message: not in the timetable." in sent[2].content
