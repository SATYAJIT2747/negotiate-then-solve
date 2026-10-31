"""A request survives a busy model server, a full context and a server restart."""

import pytest

from nts.agents import COORDINATOR_V3
from nts.api import Service
from nts.api.service import to_event
from nts.harness import Harness
from nts.models import ScriptedModel, call, say
from nts.models.openai_chat import ModelUnavailable, OpenAIChat
from nts.telemetry import JsonlSink, MemorySink, Tracer
from nts.tools import ALL_TOOLS, Registry, Reply

FULL = 'HTTP 500: {"error":{"code":500,"message":"Context size has been exceeded."}}'


def _full(messages):
    raise ModelUnavailable(FULL)


def test_a_full_context_is_trimmed_and_asked_again(store):
    t = store.world.instance.faculty[0]
    model, sink = ScriptedModel([call("classes_of", teacher="me"), _full, say("Two classes.")]), MemorySink()
    out = Harness(Registry(ALL_TOOLS), model, Tracer([sink])).run(COORDINATOR_V3, "My classes?", store, t.id, 5)
    assert out.status == "answered" and out.text == "Two classes."
    retried = [s for s in sink.spans if s.kind == "model_call" and s.attrs.get("retried")]
    assert len(retried) == 1 and to_event(retried[0]) is None  # the page shows no failure
    assert "omitted" not in model.seen[1][-1].content  # the latest tool result is kept


def test_a_context_still_full_after_trimming_ends_the_episode(store):
    t = store.world.instance.faculty[0]
    model = ScriptedModel([_full, _full])
    out = Harness(Registry(ALL_TOOLS), model, Tracer([MemorySink()])).run(COORDINATOR_V3, "Hi", store, t.id, 5)
    assert out.status == "model_error" and "Context size" in out.detail


def test_a_busy_server_is_tried_again(monkeypatch):
    m, tries = OpenAIChat("http://x/v1", backoff=0), []

    def flaky(url, payload):
        tries.append(url)
        if len(tries) < 3:
            raise ModelUnavailable("HTTP 503: loading model", transient=True)
        return {"ok": True}

    monkeypatch.setattr(m, "_post_once", flaky)
    assert m._post("http://x/v1/chat/completions", {}) == {"ok": True} and len(tries) == 3


@pytest.mark.parametrize("error", [ModelUnavailable(FULL), ModelUnavailable("adapter 'x' is not loaded")])
def test_what_a_retry_cannot_fix_fails_at_once(monkeypatch, error):
    m, tries = OpenAIChat("http://x/v1", backoff=0), []

    def fails(url, payload):
        tries.append(url)
        raise error

    monkeypatch.setattr(m, "_post_once", fails)
    with pytest.raises(ModelUnavailable):
        m._post("http://x/v1/chat/completions", {})
    assert len(tries) == 1


def test_a_restarted_server_still_shows_earlier_requests(store, tmp_path):
    t = store.world.instance.faculty[0]
    spans = tmp_path / "spans"
    first = Service(store, ScriptedModel([say("You teach on Monday."), call("ask_requester", question="Which class?"),
                                          call("ask_requester", question="Which week?"), say("Thanks.")]),
                    Tracer([JsonlSink(spans)]))
    done = first.wait(first.start("What do I teach?", t.id, 5).id)
    waiting = first.wait(first.start("Move my class.", t.id, 5).id)
    finished = first.wait(first.start("Move my lab.", t.id, 5).id)
    first.reply(finished.id, Reply(text="Week 6."))
    first.wait(finished.id)
    assert (done.status, waiting.status, finished.status) == ("answered", "waiting", "answered")

    again = Service(store, ScriptedModel([]), Tracer([MemorySink()]))
    assert again.restore(spans) == 3 and again.restore(spans) == 0  # each once
    old = again.get(done.id)
    assert (old.request, old.actor_name, old.week, old.status, old.text) == ("What do I teach?", t.name, 5,
                                                                             "answered", "You teach on Monday.")
    assert [e["type"] for e in old.events] == ["model", "end"] and not old.busy
    assert again.get(waiting.id).status == "interrupted" and "restarted" in again.get(waiting.id).detail
    assert [e["type"] for e in again.get(finished.id).events] == ["model", "waiting", "reply", "model", "end"]


def test_requests_on_another_department_are_not_restored(store, solved, tmp_path):
    from nts.kernel.generator import generate_department
    from nts.kernel.instance import policy_constraints
    from nts.state import ConstraintAdded, Store

    t = store.world.instance.faculty[0]
    first = Service(store, ScriptedModel([say("Hello.")]), Tracer([JsonlSink(tmp_path)]))
    first.wait(first.start("Hi", t.id, 5).id)
    inst = generate_department(seed=2)
    other = Store.create(inst, [ConstraintAdded(constraint=c) for c in policy_constraints(inst)])
    assert Service(other, ScriptedModel([]), Tracer([MemorySink()])).restore(tmp_path) == 0
