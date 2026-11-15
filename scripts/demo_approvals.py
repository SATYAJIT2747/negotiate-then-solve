"""Play the timetable office's approvals for the demo against the running web app.

Reads a plan (scripts/demo_approvals.toml by default) of approve/reject steps and carries them out as the
timetable office through the app's own API, so the change shows up live in every open browser: the
approved change is published and its people are told, a rejection reaches its sender with the reason.

    uv run python scripts/demo_approvals.py --dry-run         # what each step matches, nothing changed
    uv run python scripts/demo_approvals.py                   # play it, 4 s between steps
    uv run python scripts/demo_approvals.py --step            # wait for Enter before each step
    uv run python scripts/demo_approvals.py --rest approve    # then approve whatever else is waiting

Changes waiting for approval build on each other (each one on the change queued before it), so an approval
that builds on one still waiting is tried again after the others; a rejection is taken back cleanly by the
app (later changes are rebased).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Api:
    def __init__(self, url: str, user: str, world: str | None) -> None:
        self.url, self.headers = url.rstrip("/") + "/api", {"X-User": user, "Content-Type": "application/json"}
        if world:
            self.headers["X-World"] = world

    def call(self, path: str, body: dict | None = None) -> tuple[int, dict | list]:
        req = urllib.request.Request(self.url + path, headers=self.headers, method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", "")
            except ValueError:
                detail = e.reason
            return e.code, {"detail": detail}

    def pending(self) -> list[dict]:
        code, rows = self.call("/approvals")
        if code != 200:
            sys.exit(f"cannot read the approvals ({code}): {rows}. Is the app running, and is --user the office?")
        return rows  # type: ignore[return-value]


def find(step: dict, waiting: list[dict]) -> dict | None:
    m = step["match"].strip().lower()
    return next((c for c in waiting if c["id"].lower() == m), None) or next(
        (c for c in waiting if m in c["text"].lower()), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, default=HERE / "demo_approvals.toml")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--user", default="C-TT", help="who decides (the timetable office)")
    ap.add_argument("--world", default=None, help="X-World, when not the demo department")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between steps")
    ap.add_argument("--step", action="store_true", help="wait for Enter before each step")
    ap.add_argument("--dry-run", action="store_true", help="show what each step matches; change nothing")
    ap.add_argument("--rest", choices=["leave", "approve", "reject"], default="leave",
                    help="what to do with waiting changes no step names")
    args = ap.parse_args()

    steps = tomllib.loads(args.plan.read_text(encoding="utf-8")).get("step", [])
    for s in steps:
        if s.get("do") not in ("approve", "reject") or not s.get("match"):
            sys.exit(f"bad step in {args.plan}: {s}")
    api = Api(args.url, args.user, args.world)
    waiting = api.pending()
    print(f"{len(waiting)} change(s) waiting for approval")

    # pair steps with requests now, so the plan can be checked before anything happens
    todo: list[tuple[dict, dict]] = []
    for s in steps:
        c = find(s, [w for w in waiting if w["id"] not in {t[1]["id"] for t in todo}])
        if c is None:
            print(f"  -- no waiting change matches {s['match']!r}: skipped")
        else:
            todo.append((s, c))
    named = {c["id"] for _, c in todo}
    if args.rest != "leave":
        todo += [({"do": args.rest, "match": c["id"], "reason": "Not taken forward this term."}, c)
                 for c in waiting if c["id"] not in named]
    for s, c in todo:
        print(f"  {s['do']:7} {c['id']}  {c['sender_name']}: {c['text']}")
    if args.dry_run or not todo:
        return

    retry: list[tuple[dict, dict]] = []
    for n, (s, c) in enumerate(todo, 1):
        if n > 1 and not args.step:
            time.sleep(args.pause)
        if args.step:
            input(f"\n[Enter] {s['do']} {c['id']} ")
        retry += play(api, s, c)
    for s, c in retry:  # approvals that waited on an earlier change
        time.sleep(args.pause)
        play(api, s, c, again=True)
    left = api.pending()
    print(f"\ndone; {len(left)} change(s) still waiting for approval" + (
        ": " + ", ".join(c["id"] for c in left) if left else ""))


def play(api: Api, s: dict, c: dict, again: bool = False) -> list[tuple[dict, dict]]:
    print()
    if s.get("say"):
        print(s["say"])
    print(f"> {s['do']} {c['id']} ({c['sender_name']}): {c['text']}")
    if s["do"] == "approve":
        code, out = api.call(f"/approvals/{c['id']}/approve", {})
        if code == 200:
            v = out.get("version")
            told = ", ".join(sorted(out.get("notified") or {})) or "nobody else"
            print(f"  published as timetable version {v}; told: {told}" if v is not None
                  else "  confirmed: the timetable already met it, nothing to publish")
            return []
        if code == 409 and "still waiting for approval" in str(out.get("detail")) and not again:
            print(f"  waits on an earlier change ({out['detail']}); trying again at the end")
            return [(s, c)]
    else:
        code, out = api.call(f"/approvals/{c['id']}/reject", {"reason": s.get("reason", "")})
        if code == 200:
            print(f"  rejected; {c['sender_name']} is told" + (f": {s['reason']}" if s.get("reason") else ""))
            return []
    print(f"  failed ({code}): {out.get('detail', out)}")
    return []


if __name__ == "__main__":
    main()
