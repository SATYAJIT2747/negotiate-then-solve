"""Where spans go: JSONL files (one per day, the record itself) and a SQLite index to query them.

The JSONL files are the source of truth: append-only, one span per line, readable without this code.
The SQLite index holds the same spans with their key fields as columns, for questions like "every
tool call that failed with unknown_name last week":

    SELECT json_extract(attrs, '$.call.arguments') FROM spans
    WHERE kind = 'tool_call' AND json_extract(attrs, '$.result.error.code') = 'unknown_name'
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .spans import Span


class JsonlSink:
    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def write(self, span: Span) -> None:
        """One locked write per span: processes appending to one file at once (parallel collections) would
        otherwise interleave a long line, which Windows writes in pieces."""
        path = self.folder / f"{span.start:%Y-%m-%d}.jsonl"
        data = (span.model_dump_json() + "\n").encode("utf-8")
        with path.open("ab") as f, _locked(f):
            f.seek(0, os.SEEK_END)
            f.write(data)
            f.flush()


@contextmanager
def _locked(f) -> Iterator[None]:
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)  # every writer locks the first byte; LK_LOCK retries for 10 s
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class SqliteSink:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_id TEXT, seq INTEGER, kind TEXT,
                name TEXT, start TEXT, ms REAL, status TEXT, attrs TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS spans_trace ON spans (trace_id, seq)")
            db.execute("CREATE INDEX IF NOT EXISTS spans_kind ON spans (kind, start)")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            with db:  # commits, or rolls back on an exception
                yield db
        finally:
            db.close()

    def write(self, span: Span) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (span.span_id, span.trace_id, span.parent_id, span.seq, span.kind, span.name,
                        span.start.isoformat(), span.ms, span.status, json.dumps(span.attrs, default=str)))

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._connect() as db:
            return db.execute(sql, params).fetchall()


def read_spans(folder: Path | str) -> Iterator[Span]:
    """Every span in the files; a damaged line (two writers interleaved, before writes were locked) is skipped
    and counted on stderr."""
    for path in sorted(Path(folder).glob("*.jsonl")):
        damaged = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                try:
                    yield Span.model_validate_json(line)
                except ValueError:
                    damaged += 1
        if damaged:
            print(f"{path.name}: skipped {damaged} damaged line(s)", file=sys.stderr)


def load_trace(folder: Path | str, trace_id: str) -> list[Span]:
    """Every span of one episode from the JSONL files, in the order they started (an approval may come from
    another process, whose ``seq`` counts from 0 again: the start time orders those)."""
    return sorted((s for s in read_spans(folder) if s.trace_id == trace_id), key=lambda s: (s.start, s.seq))
