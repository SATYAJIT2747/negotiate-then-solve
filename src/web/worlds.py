"""Several independent worlds in one server: the built-in demo department, and
departments built from course offering documents (the demo campus, or any
offering PDF the timetable office uploads).

A world is everything the web app runs on: people, sections, rooms, the weekly
timetable and its history, inboxes, the semester plan. Each request names its
world in the ``X-World`` header (the demo when it names none); ``current()``
resolves it, so the endpoints are written for one world as before. The human
study is shared by all worlds.

Worlds built from offering documents are remembered in ``runs/worlds/`` (the
document, as parsed, and its section sizes) and rebuilt in the background at
start-up; a world still being built answers 503 until it is ready.
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextvars import ContextVar
from pathlib import Path

from semester.campus import campus_offerings
from semester.department import instance_from_offerings
from semester.offerings import OfferingDoc

from .world import ROOT, World

DEMO = "demo"
current_world: ContextVar[str] = ContextVar("current_world", default=DEMO)


class WorldStarting(RuntimeError):
    pass


class Registry:
    def __init__(self, demo: World, folder: Path | None = None, parser_mode: str | None = None,
                 restore: bool = True) -> None:
        self.folder = Path(folder or ROOT / "runs" / "worlds")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.parser_mode = parser_mode or demo.parser_mode
        self.study = demo.study
        self.seed_history = demo.seed_history  # worlds replay a start-up history when the demo does
        self.lock = threading.Lock()
        self.entries: dict[str, dict] = {
            DEMO: {"id": DEMO, "name": "CSE department (demo)", "kind": "demo", "status": "ready", "error": None,
                   "world": demo, "source": "built in", "created": time.time()}}
        if restore:
            for spec in self._saved():
                self._start(spec, OfferingDoc.model_validate_json((self.folder / spec["id"] / "offerings.json")
                                                                  .read_text(encoding="utf-8")), spec.get("sizes", {}))

    # -- persistence ------------------------------------------------------------------------

    def _saved(self) -> list[dict]:
        p = self.folder / "registry.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def _save(self) -> None:
        specs = [{k: e[k] for k in ("id", "name", "kind", "source", "created", "sizes")}
                 for e in self.entries.values() if e["kind"] != "demo"]
        (self.folder / "registry.json").write_text(json.dumps(specs, indent=1), encoding="utf-8")

    # -- worlds -----------------------------------------------------------------------------

    def get(self, world_id: str | None = None) -> World:
        e = self.entries.get(world_id or current_world.get()) or self.entries[DEMO]
        if e["status"] != "ready":
            raise WorldStarting(e["error"] or f"{e['name']} is still being built; try again in a moment")
        return e["world"]

    def list(self) -> list[dict]:
        out = []
        for e in self.entries.values():
            w = e["world"]
            row = {k: e[k] for k in ("id", "name", "kind", "status", "error", "source")}
            if w is not None:
                inst = w.instance
                row |= {"people": len(inst.faculty), "sections": len(inst.groups), "sessions": len(inst.sessions),
                        "rooms": len(inst.rooms), "seeding": w.seeding}
            out.append(row)
        return out

    def slug(self, name: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "world"
        wid, n = base, 2
        while wid in self.entries:
            wid, n = f"{base}-{n}", n + 1
        return wid

    def create(self, name: str, doc: OfferingDoc, sizes: dict[str, int] | None = None, source: str = "") -> dict:
        """A new world from an offering document, built in the background."""
        with self.lock:
            spec = {"id": self.slug(name), "name": name, "kind": "offerings", "source": source or doc.source,
                    "created": time.time(), "sizes": sizes or {}}
            d = self.folder / spec["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "offerings.json").write_text(doc.model_dump_json(), encoding="utf-8")
            self._start(spec, doc, sizes or {})
            self._save()
        return self.entries[spec["id"]]

    def campus(self) -> dict:
        existing = next((e for e in self.entries.values() if e["source"] == campus_offerings().source), None)
        return existing or self.create("Campus (demo)", campus_offerings())

    def rebuild(self, world_id: str, doc: OfferingDoc, sizes: dict[str, int] | None = None) -> dict:
        """Replace a world's people, sections, rooms and weekly timetable with an offering document's.
        Its weekly history starts again; its semester plan (folder) is kept."""
        with self.lock:
            e = self.entries[world_id]
            spec = {k: e.get(k) for k in ("id", "name", "created")} | {
                "kind": "offerings", "source": doc.source, "sizes": sizes or {}}
            d = self.folder / world_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "offerings.json").write_text(doc.model_dump_json(), encoding="utf-8")
            self._start(spec, doc, sizes or {})
            self._save()
        return self.entries[world_id]

    def restore(self, world_id: str) -> None:
        """Back to the demo state: the world as it stood when its start-up history had finished
        replaying (``World.restore``), at once, without replaying it."""
        e = self.entries.get(world_id) or self.entries[DEMO]
        if e["status"] != "ready" or e["world"] is None:
            raise ValueError(f"{e['name']} is still being built")
        e["world"].restore()

    def reset(self, world_id: str) -> None:
        """A fresh weekly history: the demo world anew, or an offering world rebuilt from its document."""
        e = self.entries.get(world_id) or self.entries[DEMO]
        if e["kind"] == "demo":
            e["world"] = World(parser_mode=self.parser_mode, study=self.study, seed_history=self.seed_history)
            return
        doc = OfferingDoc.model_validate_json((self.folder / e["id"] / "offerings.json").read_text(encoding="utf-8"))
        self.rebuild(e["id"], doc, e.get("sizes"))

    def delete(self, world_id: str) -> None:
        if world_id == DEMO:
            raise ValueError("the demo world cannot be deleted")
        with self.lock:
            self.entries.pop(world_id, None)
            self._save()

    def _start(self, spec: dict, doc: OfferingDoc, sizes: dict[str, int]) -> None:
        entry = {**spec, "status": "starting", "error": None, "world": None}
        self.entries[spec["id"]] = entry

        def build() -> None:
            try:
                inst = instance_from_offerings(doc, sizes, name=spec["name"])
                folder = self.folder / spec["id"] / "semester"
                # its semester plan starts from the same department (semester.demo derives the document)
                world = World(parser_mode=self.parser_mode, instance=inst, world_id=spec["id"],
                              semester_dir=folder, study=self.study, seed_history=self.seed_history,
                              campus=doc.source == campus_offerings().source)
                entry.update(world=world, status="ready")
            except Exception as ex:  # noqa: BLE001 - shown on the world's card
                entry.update(status="error", error=f"could not build this world: {type(ex).__name__}: {ex}")

        threading.Thread(target=build, daemon=True).start()
