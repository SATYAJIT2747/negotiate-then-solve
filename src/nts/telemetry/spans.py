"""Spans: one record per episode, turn, model call and tool call, nested by ``parent_id``.

    tracer = Tracer([JsonlSink("runs/telemetry/spans"), SqliteSink("runs/telemetry/index.sqlite")])
    with tracer.span("episode", "coordinator", trace_id="ep-1", request=text) as ep:
        with tracer.span("model_call", model.name, messages=...) as mc:
            mc.attrs["response"] = turn.model_dump(mode="json")
        ep.attrs["status"] = "answered"

A span is written when it ends, so children are written before their parents; ``seq`` keeps the order
they started in. A span that ends by an exception is written with status "error" and the exception's
type and message, and the exception goes on.
"""

from __future__ import annotations

import contextvars
import itertools
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Kind = Literal["episode", "turn", "model_call", "tool_call", "approval", "evaluation"]  # the last two judge an episode


class Span(BaseModel):
    trace_id: str  # the episode id
    span_id: str
    parent_id: str | None = None
    seq: int  # start order within the tracer
    kind: Kind
    name: str
    start: datetime
    ms: float = 0.0
    status: Literal["ok", "error"] = "ok"
    attrs: dict[str, Any] = Field(default_factory=dict)


class Sink(Protocol):
    def write(self, span: Span) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def write(self, span: Span) -> None:
        self.spans.append(span)

    def trace(self, trace_id: str) -> list[Span]:
        return sorted((s for s in self.spans if s.trace_id == trace_id), key=lambda s: s.seq)


_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("nts_span", default=None)


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:12]


class Tracer:
    def __init__(self, sinks: Sequence[Sink] = ()) -> None:
        self.sinks = list(sinks)
        self._seq = itertools.count()

    @contextmanager
    def span(self, kind: Kind, name: str, *, trace_id: str | None = None, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        trace = trace_id or (parent.trace_id if parent else new_id("ep-"))
        span = Span(trace_id=trace, span_id=new_id(), parent_id=parent.span_id if parent else None,
                    seq=next(self._seq), kind=kind, name=name, start=datetime.now(), attrs=attrs)
        token = _current.set(span)
        t0 = time.perf_counter()
        try:
            yield span
        except BaseException as e:
            span.status = "error"
            span.attrs["error"] = {"type": type(e).__name__, "message": str(e)}
            raise
        finally:
            _current.reset(token)
            span.ms = round((time.perf_counter() - t0) * 1000, 3)
            for sink in self.sinks:
                sink.write(span)
