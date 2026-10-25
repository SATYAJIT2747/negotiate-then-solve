"""Observability: what the agent did, from the telemetry index (``runs/telemetry/index.sqlite``) and eval reports.

    obs = Observability(Path("runs/telemetry"), Path("runs/eval"))
    obs.overview(days=7)        # episodes by status and model, tokens, time, refusals, the commonest tool errors
    obs.episodes(limit=50, status="submitted", model="gemini")
    obs.trace("ep-...")         # every span of one episode, in order
    obs.evals()                 # each eval report's summary, newest first

The index holds every span as a row (kind, name, start, ms, attrs as JSON), so these are SQL over it.
A paused episode's "resume" spans carry how it ended, and they override the episode span's.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


class Observability:
    def __init__(self, telemetry: Path, evals: Path) -> None:
        self.db = telemetry / "index.sqlite"
        self.evals_dir = evals

    def _rows(self, sql: str, params: tuple = ()) -> list[tuple]:
        if not self.db.exists():
            return []
        db = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            return db.execute(sql, params).fetchall()
        finally:
            db.close()

    def _episodes(self, since: str | None = None) -> list[dict]:
        where, params = ("AND start >= ?", (since,)) if since else ("", ())
        out: dict[str, dict] = {}
        for trace, start, attrs in self._rows(f"SELECT trace_id, start, attrs FROM spans WHERE kind = 'episode' {where} "
                                              "ORDER BY start", params):
            a = json.loads(attrs)
            u = a.get("usage") or {}
            out[trace] = {"id": trace, "start": start[:19], "model": a.get("model", ""), "agent": a.get("agent", ""),
                          "agent_version": a.get("agent_version", ""), "actor": a.get("actor", ""),
                          "request": a.get("request", ""), "status": a.get("status", ""), "turns": a.get("turns", 0),
                          "tool_calls": a.get("tool_calls", 0), "refused": a.get("refused", 0),
                          "tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0),
                          "seconds": a.get("seconds", 0.0), "verdict": None, "family": None}
        if not out:
            return []
        marks = ",".join("?" * len(out))
        for trace, kind, attrs in self._rows(f"SELECT trace_id, kind, attrs FROM spans WHERE trace_id IN ({marks}) "
                                             "AND kind IN ('resume', 'evaluation', 'approval') ORDER BY start",
                                             tuple(out)):
            a, e = json.loads(attrs), out[trace]
            if kind == "resume":
                u = a.get("usage") or {}
                e.update(status=a.get("status", e["status"]), turns=a.get("turns", e["turns"]),
                         tool_calls=a.get("tool_calls", e["tool_calls"]), refused=a.get("refused", e["refused"]),
                         tokens=u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) or e["tokens"])
            elif kind == "evaluation":
                e.update(verdict="passed" if a.get("passed") else "failed", family=a.get("family"),
                         failures=a.get("failures", []))
            elif kind == "approval" and e["verdict"] is None:
                e["verdict"] = "approved" if a.get("committed") else "rejected"
        return list(out.values())

    def overview(self, days: int = 7) -> dict:
        since = (datetime.now() - timedelta(days=days)).isoformat()
        eps = self._episodes(since)
        by_day: dict[str, Counter] = defaultdict(Counter)
        for e in eps:
            by_day[e["start"][:10]][e["status"]] += 1
        models: dict[str, dict] = {}
        for e in eps:
            m = models.setdefault(e["model"], {"model": e["model"], "episodes": 0, "passed": 0, "judged": 0,
                                               "tokens": 0, "seconds": 0.0, "refused": 0, "tool_calls": 0})
            m["episodes"] += 1
            m["tokens"] += e["tokens"]
            m["seconds"] += e["seconds"] or 0
            m["refused"] += e["refused"] or 0
            m["tool_calls"] += e["tool_calls"] or 0
            if e["verdict"] in ("passed", "failed"):
                m["judged"] += 1
                m["passed"] += e["verdict"] == "passed"
        errors = Counter(code for (code,) in self._rows(
            "SELECT json_extract(attrs, '$.result.error.code') FROM spans WHERE kind = 'tool_call' AND start >= ? "
            "AND json_extract(attrs, '$.ok') = 0", (since,)) if code)
        tools = Counter(name for (name,) in self._rows(
            "SELECT name FROM spans WHERE kind = 'tool_call' AND start >= ?", (since,)))
        return {"days": days, "episodes": len(eps), "status": dict(Counter(e["status"] for e in eps)),
                "by_day": [{"day": d, **c} for d, c in sorted(by_day.items())],
                "models": sorted(models.values(), key=lambda m: -m["episodes"]),
                "errors": errors.most_common(10), "tools": tools.most_common(20)}

    def episodes(self, *, limit: int = 50, status: str | None = None, model: str | None = None,
                 verdict: str | None = None) -> list[dict]:
        eps = self._episodes()
        eps = [e for e in eps if (not status or e["status"] == status) and (not model or model in e["model"])
               and (not verdict or e["verdict"] == verdict)]
        return sorted(eps, key=lambda e: e["start"], reverse=True)[:limit]

    def trace(self, trace_id: str) -> list[dict]:
        rows = self._rows("SELECT span_id, parent_id, kind, name, start, ms, status, attrs FROM spans "
                          "WHERE trace_id = ? ORDER BY start, seq", (trace_id,))
        return [{"id": r[0], "parent": r[1], "kind": r[2], "name": r[3], "start": r[4][:23], "ms": r[5],
                 "status": r[6], "attrs": json.loads(r[7])} for r in rows]

    def evals(self) -> list[dict]:
        out = []
        for path in sorted(self.evals_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            s = r.get("summary", {})
            out.append({"report": path.stem, "model": r.get("model", ""), "when": datetime.fromtimestamp(
                path.stat().st_mtime).isoformat(timespec="minutes"), **s.get("all", {}),
                "families": {f: f"{v['passed']}/{v['n']}" for f, v in s.get("families", {}).items()}})
        return out
