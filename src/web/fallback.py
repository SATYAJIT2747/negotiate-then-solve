"""Keep working when the model server is not.

The parser, the policy check and the inbox's reply reader each have a model-backed version and a
rule-based stand-in (``RuleParser``, ``RulePolicyAgent``, ``RuleReplyParser``). ``ModelLink`` tracks
whether the model is reachable: it builds the client lazily (a server that is down at start-up
does not stop the app), probes it quickly before use (a dead host would otherwise hold a request
until the client's timeout), and after a failure leaves it alone for ``retry_after`` seconds.
``Fallback`` wraps one component: the model while the link is up, the stand-in otherwise, call by
call, so the app moves back to the model by itself when the server returns.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from language.llm import LLMError


class ModelLink:
    def __init__(self, make_client: Callable[[], Any], probe: Callable[[Any], None] | None = None,
                 retry_after: float = 30.0, recheck: float = 30.0,
                 on_change: Callable[[bool, str | None], None] | None = None) -> None:
        self.make_client = make_client
        self.probe = probe  # raises when the server cannot answer (gets the client, None before one is built)
        self.retry_after, self.recheck = retry_after, recheck
        self.on_change = on_change
        self.client: Any = None
        self.up: bool | None = None  # None: not tried yet
        self.error: str | None = None
        self.since = time.time()
        self._checked = 0.0
        self._down_until = 0.0
        self._lock = threading.Lock()

    def usable(self) -> bool:
        """Whether to try the model now (a cheap check; at most one probe per ``recheck`` seconds)."""
        now = time.time()
        with self._lock:
            if now < self._down_until:
                return False
            if self.client is not None and self.up and now - self._checked < self.recheck:
                return True
            try:
                if self.probe is not None:  # first: building a client may itself wait on a dead host
                    self.probe(self.client)
                if self.client is None:
                    self.client = self.make_client()
            except Exception as e:  # noqa: BLE001 - any failure to reach the server means "use the stand-in"
                self._down(e)
                return False
            self._checked = now
            self._set(True, None)
            return True

    def failed(self, e: BaseException) -> None:
        """A call through the link failed in a way that says the server is gone."""
        with self._lock:
            self._down(e)

    def _down(self, e: BaseException) -> None:
        self._down_until = time.time() + self.retry_after
        self._set(False, f"{type(e).__name__}: {e}"[:300])

    def _set(self, up: bool, error: str | None) -> None:
        changed = self.up is not up
        self.up, self.error = up, error
        if changed:
            self.since = time.time()
            print(f"[model] {'reachable again' if up else 'unreachable, using the rule-based stand-ins'}"
                  + (f": {error}" if error else ""), flush=True)
            if self.on_change is not None:
                self.on_change(up, error)

    def status(self) -> dict:
        return {"up": self.up, "error": self.error, "since": self.since}


def _outage(e: BaseException) -> bool:
    """Errors that mean the server is not answering, as opposed to one bad output."""
    if isinstance(e, (OSError, TimeoutError)):
        return True
    return isinstance(e, LLMError) and any(w in str(e).lower() for w in
                                           ("not reachable", "timed out", "timeout", "connection", "no model served",
                                            " 502", " 503", " 504"))


def _reset(e: BaseException) -> bool:
    """A connection the server closed mid-call (``ConnectionResetError``, WinError 10054): usually momentary,
    unlike a server that is not there at all."""
    seen: BaseException | None = e
    while seen is not None:
        if isinstance(seen, (ConnectionResetError, ConnectionAbortedError)) or isinstance(
                getattr(seen, "reason", None), (ConnectionResetError, ConnectionAbortedError)):
            return True
        if any(w in str(seen).lower() for w in ("connection reset", "forcibly closed", "10054", "connectionreset")):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


class Fallback:
    """One component: ``build(client)`` while the model is reachable, ``backup`` otherwise. A call
    that fails falls back for that call; an outage also marks the link down."""

    def __init__(self, link: ModelLink, build: Callable[[Any], Any], backup: Any, name: str) -> None:
        self.link, self.build, self.backup, self.name = link, build, backup, name
        self._primary: Any = None
        self._for: Any = None  # the client the primary was built on
        self._used = threading.local()  # who answered this thread's last call: "model" or "rules"

    def primary(self) -> Any | None:
        if not self.link.usable():
            return None
        if self._primary is None or self._for is not self.link.client:
            self._primary, self._for = self.build(self.link.client), self.link.client
        return self._primary

    def _call(self, method: str, *args, **kwargs):
        p = self.primary()
        if p is not None:
            try:
                out = getattr(p, method)(*args, **kwargs)
                self._used.who = "model"
                return out
            except Exception as e:  # noqa: BLE001 - the stand-in answers instead
                if _outage(e):
                    self.link.failed(e)
                print(f"[model] {self.name}: {type(e).__name__}: {str(e)[:160]}; rule-based stand-in used",
                      flush=True)
        self._used.who = "rules"
        return getattr(self.backup, method)(*args, **kwargs)

    def reader(self) -> str | None:
        """Who answered this thread's last call: "model", or "rules" (the stand-in)."""
        return getattr(self._used, "who", None)

    # the methods the pipeline calls
    def parse(self, *args, **kwargs):
        return self._call("parse", *args, **kwargs)

    def compile(self, *args, **kwargs):
        """Intents from a model that reads them; None otherwise (the stand-in, or a model trained on the
        older format), and the orchestrator then parses the plain way."""
        p = self.primary()
        if p is None or not hasattr(p, "compile"):
            return None
        try:
            out = p.compile(*args, **kwargs)
            if out is not None:
                self._used.who = "model"
            return out
        except Exception as e:  # noqa: BLE001 - the plain parse (and its stand-in) answers instead
            if _outage(e):
                self.link.failed(e)
            print(f"[model] {self.name}: {type(e).__name__}: {str(e)[:160]}; parsed without intents", flush=True)
            return None

    def review(self, *args, **kwargs):
        return self._call("review", *args, **kwargs)

    def read(self, *args, **kwargs):
        return self._call("read", *args, **kwargs)

    def set_rules(self, rules) -> None:  # a policy document was added: both versions read it
        self.backup.set_rules(rules)
        if self._primary is not None:
            self._primary.set_rules(rules)

    def __getattr__(self, name: str):  # anything else (``rules``, ``client``) from whichever is in use
        p = self._primary if self._primary is not None and self.link.up else self.backup
        return getattr(p, name)
