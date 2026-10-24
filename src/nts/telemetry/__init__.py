"""Everything the harness does, recorded as nested spans (episode > turn > model_call / tool_call).

The records hold enough to rebuild each model call exactly (the messages sent, the raw reply), so they
are both the history of the system and the raw material for fine-tuning and replayed evaluation.
"""

from __future__ import annotations

from pathlib import Path

from .sinks import JsonlSink, SqliteSink, load_trace, read_spans
from .spans import MemorySink, Sink, Span, Tracer, new_id

DEFAULT_ROOT = Path("runs/telemetry")  # runs/ is gitignored: the records hold real names


def default_tracer(root: Path | str = DEFAULT_ROOT) -> Tracer:
    root = Path(root)
    return Tracer([JsonlSink(root / "spans"), SqliteSink(root / "index.sqlite")])


__all__ = ["DEFAULT_ROOT", "JsonlSink", "MemorySink", "Sink", "Span", "SqliteSink", "Tracer", "default_tracer",
           "load_trace", "new_id", "read_spans"]
