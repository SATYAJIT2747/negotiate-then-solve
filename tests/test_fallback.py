"""The web app keeps working when the model server does not (``web.fallback``)."""

import time

import pytest

from language.llm import LLMError
from web.fallback import Fallback, ModelLink


class Server:
    """A model server that can be switched off."""

    def __init__(self) -> None:
        self.up = True
        self.calls = 0

    def probe(self, _client) -> None:
        if not self.up:
            raise LLMError("local server not reachable at http://x: refused")


class Model:
    def __init__(self, server: Server) -> None:
        self.server = server

    def parse(self, text):
        self.server.calls += 1
        if not self.server.up:
            raise LLMError("local server not reachable at http://x: timed out")
        return f"model:{text}"


class Rules:
    def parse(self, text):
        return f"rules:{text}"


def make(server: Server, events: list):
    link = ModelLink(lambda: object(), probe=server.probe, retry_after=0.2, recheck=0.0,
                     on_change=lambda up, err: events.append(up))
    return link, Fallback(link, lambda _c: Model(server), Rules(), "parser")


def test_the_stand_in_answers_while_the_server_is_down_and_the_model_after():
    server, events = Server(), []
    link, parser = make(server, events)
    assert parser.parse("a") == "model:a" and link.up
    server.up = False
    assert parser.parse("b") == "rules:b" and link.up is False
    calls = server.calls
    assert parser.parse("c") == "rules:c" and server.calls == calls  # not tried again until retry_after
    server.up = True
    time.sleep(0.25)
    assert parser.parse("d") == "model:d" and link.up
    assert events == [True, False, True]


def test_a_server_down_at_start_does_not_stop_anything():
    server, events = Server(), []
    server.up = False
    link = ModelLink(lambda: (_ for _ in ()).throw(LLMError("no model served at http://x")), retry_after=60)
    parser = Fallback(link, lambda _c: Model(server), Rules(), "parser")
    assert parser.parse("a") == "rules:a" and link.up is False and "no model served" in link.error


def test_one_bad_output_falls_back_without_marking_the_server_down():
    class Garbled(Model):
        def parse(self, text):
            raise ValueError("the output was not valid JSON")

    link = ModelLink(lambda: object())
    parser = Fallback(link, lambda _c: Garbled(Server()), Rules(), "parser")
    assert parser.parse("a") == "rules:a" and link.up


@pytest.mark.slow
def test_a_world_on_the_local_model_starts_and_works_with_no_server(monkeypatch):
    from web.world import World

    monkeypatch.setenv("NTS_LOCAL_URL", "http://127.0.0.1:9/v1")  # nothing listens on the discard port
    monkeypatch.setenv("NTS_HOLD_MINUTES", "0")  # a change waits for the model; here, not at all
    monkeypatch.setenv("NTS_HOLD_CHECK_SECONDS", "0.05")
    start = time.time()
    w = World(parser_mode="local", seed_history=False, deadline=5)
    assert time.time() - start < 60 and w.model_link.up is False
    r = w.intake.from_portal("F-105", "I'd prefer no classes before 10 am on Friday, if possible.")
    w.submit(r)
    case = _settled(w, r.id, 120)
    assert case.status.value in ("awaiting_approval", "published"), case.reply
    assert any(e["kind"] == "model_unreachable" for e in w.store.events())



def test_a_reset_connection_is_told_apart_from_an_outage():
    import urllib.error

    from web.fallback import _reset

    reset = ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
    assert _reset(reset) and _reset(urllib.error.URLError(reset))
    try:
        try:
            raise reset
        except OSError as e:
            raise LLMError(f"local server not reachable: {e}") from e
    except LLMError as wrapped:
        assert _reset(wrapped)
    assert not _reset(ConnectionRefusedError("no server")) and not _reset(LLMError("local server 500"))


def test_the_app_takes_the_fast_path_only_without_a_model():
    """The fine-tuned reader is right on 99.8% of the test split, the fast path's rule parser on 52%."""
    from web.world import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    assert w._fast_path_now()  # offline: the rule parser reads everything anyway
    w.model_link = ModelLink(lambda: object())
    assert not w._fast_path_now()  # the model is up: it reads
    w.model_link = ModelLink(lambda: object(), probe=lambda c: (_ for _ in ()).throw(ConnectionError("gone")))
    assert w._fast_path_now()  # unreachable: the fast path again


# -- a change waits for the model instead of being acted on as the rule parser reads it ----------------------


def _down(e=ConnectionError("gone")):
    return ModelLink(lambda: object(), probe=lambda c: (_ for _ in ()).throw(e))


def _settled(w, rid, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        t = w.threads.get(rid)
        if rid in w.orch.cases and t is not None and not t.is_alive():
            return w.orch.cases[rid]
        time.sleep(0.05)
    raise AssertionError(f"{rid} was not read")


@pytest.fixture
def local_world(monkeypatch):
    from web.world import World

    monkeypatch.setenv("NTS_HOLD_CHECK_SECONDS", "0.05")
    w = World(parser_mode="offline", seed_history=False, deadline=60)
    w.parser_mode = "local"  # as if served with a model; the parsers stay the offline ones
    w.model_link = _down()
    me = next(f.id for f in w.instance.faculty if f.name == "Dr. Menon")
    return w, me


def test_a_change_waits_for_the_model_and_is_read_when_it_is_back(local_world):
    w, me = local_world
    r = w.intake.from_portal(me, "I want Mondays off for the whole semester.")
    w.submit(r)
    assert r.id in w.held and r.id not in w.orch.cases  # nothing done with it
    assert w.store.request(r.id) is not None and [e["kind"] for e in w.store.events(r.id)] == ["held"]
    w.model_link = ModelLink(lambda: object())  # back
    case = _settled(w, r.id)
    assert r.id not in w.held and case.status.value != "received"
    assert [e for e in w.store.events(r.id) if e["kind"] == "released"][0]["reader"] == "model"


def test_a_rule_question_does_not_wait(local_world):
    w, me = local_world
    r = w.intake.from_portal(me, "Is it allowed to hold a class during the lunch hour?")
    w.submit(r, wait=True)
    assert r.id not in w.held and w.orch.cases[r.id].status.value == "answered"


def test_a_change_held_too_long_goes_ahead_on_the_rules(local_world, monkeypatch):
    w, me = local_world
    monkeypatch.setenv("NTS_HOLD_MINUTES", "0.001")
    r = w.intake.from_portal(me, "I want Mondays off for the whole semester.")
    w.submit(r)
    _settled(w, r.id)
    assert [e for e in w.store.events(r.id) if e["kind"] == "released"][0]["reader"] == "rules"


def test_offline_nothing_waits():
    from web.world import World

    w = World(parser_mode="offline", seed_history=False, deadline=60)
    me = next(f.id for f in w.instance.faculty if f.name == "Dr. Menon")
    r = w.intake.from_portal(me, "I want Mondays off for the whole semester.")
    w.submit(r, wait=True)
    assert not w.held and r.id in w.orch.cases
