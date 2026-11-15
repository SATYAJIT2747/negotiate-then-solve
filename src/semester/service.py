"""The semester timetable as a service: offerings, preferences, builds,
versions, in-semester changes and club bookings, saved to disk.

Builds and repairs run in a background thread (a full build takes about a
minute); the web app polls ``status``. Nothing a build produces is live until
the timetable office publishes that version.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .clubs import ClubDecision, ClubDesk, ClubRequest
from .offerings import OfferingDoc, parse_offerings
from .requests import parse
from .solver import (
    Preference,
    Room,
    SemesterCalendar,
    SemesterSolver,
    Timetable,
    _applies,
    _pref_violated,
    Window,
    cohort_size,
    components,
    default_rooms,
)


class Closure(BaseModel):
    room: str
    window: Window
    text: str = ""


class Version(BaseModel):
    version: int
    created: str
    kind: str  # build | change
    note: str = ""
    published: bool = False
    based_on: int | None = None
    diff: list[dict] = Field(default_factory=list)
    timetable: Timetable


class SemesterState(BaseModel):
    doc: OfferingDoc | None = None
    sizes: dict[str, int] = Field(default_factory=dict)
    preferences: list[Preference] = Field(default_factory=list)
    closures: list[Closure] = Field(default_factory=list)
    versions: list[Version] = Field(default_factory=list)
    bookings: list[ClubDecision] = Field(default_factory=list)
    semester_start: str = "2026-07-27"


class SemesterPlanner:
    def __init__(self, folder: Path, sample: Path | None = None, time_limit: float = 60.0,
                 demo: OfferingDoc | None = None, rooms: list[Room] | None = None,
                 preferences: list[tuple[str, str]] | None = None) -> None:
        """``demo``: the web app's demo department as an offering document (``semester.demo``),
        used as the sample instead of the PDF and loaded at start when nothing else is. ``preferences``:
        (sender, text) added before the start-up build (``bootstrap_demo``); the demo department's when None."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "state.json"
        self.sample = sample
        self.demo = demo
        self.time_limit = time_limit
        self.calendar = SemesterCalendar()
        self.demo_rooms = rooms
        self.demo_preferences = preferences
        self.rooms: list[Room] = default_rooms()  # follows the document: see _rooms_for
        self.state = SemesterState.model_validate_json(self.path.read_text(encoding="utf-8")) \
            if self.path.exists() else SemesterState()
        self.lock = threading.Lock()
        self.job: dict = {"running": False}
        self.clock: Callable[[], date] | None = None  # the day it is (a demo's fixed day); the real date when None
        if demo is not None and self._is_demo(self.state.doc):  # a demo PDF saved before ids were matched
            before = [c.id for c in self.state.doc.cohorts]
            self._demo_ids(self.state.doc)
            renamed = dict(zip(before, (c.id for c in self.state.doc.cohorts), strict=True))
            if any(a != b for a, b in renamed.items()):
                self.state.sizes = {renamed.get(k, k): v for k, v in self.state.sizes.items()}
                for pref in self.state.preferences:
                    if pref.target == "cohort":
                        pref.who = renamed.get(pref.who, pref.who)
                self.save()
        self._rooms_for(self.state.doc)
        if demo is not None:
            doc = self.state.doc
            bundled = doc is not None and sample is not None and doc.source == sample.name
            # the built-in demo saved under its earlier label; an uploaded demo PDF is the office's choice and stays
            stale = doc is not None and doc.source != demo.source and doc.source.endswith("(demo department)")
            if doc is None or bundled or stale:  # an offering document the office uploaded itself is kept
                if bundled:  # the state built on the institute's PDF is kept aside, not lost
                    backup = self.folder / "state-institute-offerings.json"
                    if not backup.exists():
                        backup.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
                    self.state = SemesterState()
                self.load_demo()

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(self.state.model_dump_json(), encoding="utf-8")
            tmp.replace(self.path)

    # -- offerings ---------------------------------------------------------------------

    def load_offerings(self, path: Path) -> OfferingDoc:
        doc = parse_offerings(path)
        if not doc.courses:
            raise ValueError("no courses found: expected rows with a course code and L-T-P-C credits")
        if self._is_demo(doc):
            self._demo_ids(doc)
        self.state.doc = doc
        self._rooms_for(doc)
        demo = self._is_demo(doc)
        self.state.sizes = {c.id: self.state.sizes.get(c.id, (demo and self._demo_size(c)) or c.size or cohort_size(c.name))
                            for c in doc.cohorts}
        self.save()
        return doc

    def bootstrap_demo(self) -> None:
        """Build and publish the demo department's semester timetable in the background, once,
        so the pages that read it (clubs, preferences) work from the start, as the weekly one does."""
        doc = self.state.doc
        if self.demo is None or doc is None or doc.source != self.demo.source or self.state.versions or self.job["running"]:
            return

        if not self.state.preferences:  # what the demo's teachers asked for, so the build has wishes to weigh
            from .demo import PREFERENCES

            for source, text in self.demo_preferences if self.demo_preferences is not None else PREFERENCES:
                self.add(text, source)

        def run() -> None:
            self._run("build", "demo department, built at start-up", None)
            v = next((x for x in self.state.versions if not x.timetable.report.hard_violations), None)
            if v is not None:
                self.publish(v.version)

        self.job = {"running": True, "kind": "build", "note": "demo department, built at start-up", "started": time.time(),
                    "error": None, "version": None}
        threading.Thread(target=run, daemon=True).start()

    def _rooms_for(self, doc: OfferingDoc | None) -> None:
        """The demo department's rooms for the demo document, the institute's for any other."""
        self.rooms = (self.demo_rooms or default_rooms()) if self._is_demo(doc) else default_rooms()

    def _is_demo(self, doc: OfferingDoc | None) -> bool:
        """The demo document, or an uploaded one that lists only demo courses (its PDF, ``semester.demo``)."""
        if self.demo is None or doc is None or not doc.courses:
            return False
        return doc.source == self.demo.source or set(doc.courses) <= set(self.demo.courses)

    def _demo_ids(self, doc: OfferingDoc) -> None:
        """A demo PDF's programmes get the demo sections' ids (matched by their courses), so the
        timetables, preferences and bookings made with the built-in demo still line up."""
        ids = {}
        for c in doc.cohorts:
            twin = next((d for d in self.demo.cohorts if set(d.courses) == set(c.courses)), None)
            if twin is not None and twin.id not in ids.values():
                ids[c.id] = twin.id
        for c in doc.cohorts:
            c.id = ids.get(c.id, c.id)
        for o in doc.courses.values():
            o.cohorts = [ids.get(x, x) for x in o.cohorts]

    def _demo_size(self, cohort) -> int | None:
        """A demo section's real size, for the same section read from a PDF (matched by its courses)."""
        if self.demo is None:
            return None
        return next((c.size for c in self.demo.cohorts if set(c.courses) == set(cohort.courses) and c.size), None)

    def load_demo(self) -> OfferingDoc:
        if self.demo is None:
            raise ValueError("no demo department")
        self.state.doc = self.demo.model_copy(deep=True)
        self._rooms_for(self.state.doc)
        self.state.sizes = {c.id: self.state.sizes.get(c.id, c.size or cohort_size(c.name)) for c in self.demo.cohorts}
        self.save()
        return self.state.doc

    def set_sizes(self, sizes: dict[str, int]) -> None:
        for k, v in sizes.items():
            if k in self.state.sizes and 0 < int(v) < 2000:
                self.state.sizes[k] = int(v)
        self.save()

    def needs(self) -> dict[str, list[dict]]:
        """What each course needs, as the solver will place it (for review before building)."""
        if not self.state.doc:
            return {}
        comps, _ = components(self.state.doc, self.state.sizes, self.rooms, self.calendar)
        out: dict[str, list[dict]] = {}
        for c in comps:
            out.setdefault(c.course, []).append({"kind": c.kind, "label": c.label, "rooms": c.rooms,
                                                 "room_type": c.room_type, "options": len(c.options),
                                                 "pair": len(c.options[0].intervals) > 1 if c.options else False})
        return out

    # -- preferences and closures --------------------------------------------------------

    def add(self, text: str, source: str, preview: bool = False) -> dict:
        if not self.state.doc:
            raise ValueError("load the course offerings first")
        parsed = parse(text, self.state.doc, self.rooms, source)
        if parsed.error:
            return {"ok": False, "error": parsed.error}
        if not preview:
            self.state.preferences += parsed.preferences
            self.state.closures += [Closure(room=r, window=w, text=text) for r, w in parsed.closures]
            self.save()
        return {"ok": True, "summary": parsed.summary, "preferences": [p.model_dump() for p in parsed.preferences],
                "closures": [{"room": r, **w.model_dump()} for r, w in parsed.closures]}

    def remove_preference(self, pid: str) -> None:
        self.state.preferences = [p for p in self.state.preferences if p.id != pid]
        self.save()

    # -- builds ----------------------------------------------------------------------------

    def published(self) -> Version | None:
        return next((v for v in reversed(self.state.versions) if v.published), None)

    def _solver(self) -> SemesterSolver:
        return SemesterSolver(self.state.doc, rooms=self.rooms, sizes=self.state.sizes, calendar=self.calendar,
                              preferences=self.state.preferences,
                              room_closures=[(c.room, c.window) for c in self.state.closures],
                              time_limit=self.time_limit)

    def start(self, kind: str, note: str = "") -> None:
        if not self.state.doc:
            raise ValueError("load the course offerings first")
        if self.job.get("running"):
            raise RuntimeError("a build is already running")
        base = self.published() if kind == "change" else None
        if kind == "change" and base is None:
            raise ValueError("publish a timetable before proposing changes to it")
        self.job = {"running": True, "kind": kind, "note": note, "started": time.time(), "error": None,
                    "version": None}
        threading.Thread(target=self._run, args=(kind, note, base), daemon=True).start()

    def _run(self, kind: str, note: str, base: Version | None) -> None:
        try:
            tt = self._solver().solve(base.timetable if base else None)
            if not tt.meetings:
                why = self.why_infeasible() if tt.report.status == "infeasible" else []
                self.job.update(running=False, error=f"no timetable: {tt.report.status}"
                                + (". " + " ".join(why) if why else ""), report=tt.report.model_dump(), causes=why)
                return
            v = Version(version=len(self.state.versions) + 1, created=datetime.now().isoformat(timespec="seconds"),
                        kind=kind, note=note, based_on=base.version if base else None, timetable=tt,
                        diff=diff(base.timetable, tt) if base else [])
            self.state.versions.append(v)
            self.save()
            self.job.update(running=False, version=v.version)
        except Exception as e:  # noqa: BLE001 - surfaced in the UI
            self.job.update(running=False, error=f"{type(e).__name__}: {e}")

    def why_infeasible(self, quick: float = 10.0, most: int = 12) -> list[str]:
        """Why a build found no timetable, in words the office can act on: first the hard preferences that
        leave some class no time at all (read off the options, no solving); else each hard preference tried
        on its own with the closures (a short solve each, at most ``most``); else the closures alone."""
        solver = self._solver()
        hard = [p for p in self.state.preferences if p.hard]
        out = []
        for p in hard:
            for c in solver.comps:
                if c.options and _applies(p, c) and all(_pref_violated(p, o) for o in c.options):
                    out.append(f"“{p.text or p.who}” leaves no time for {c.course} {c.label}; "
                               f"remove it or make it a preference.")
                    break
        if out:
            return out

        def feasible(prefs: list[Preference]) -> bool:
            s = SemesterSolver(self.state.doc, rooms=self.rooms, sizes=self.state.sizes, calendar=self.calendar,
                               preferences=prefs, room_closures=[(c.room, c.window) for c in self.state.closures],
                               time_limit=quick)
            return s.solve().report.status != "infeasible"

        soft = [p for p in self.state.preferences if not p.hard]
        if not feasible(soft):
            return ["Even without the hard preferences there is no timetable: check the room closures and the "
                    "programme sizes against the rooms."]
        for p in hard[:most]:
            if not feasible(soft + [p]):
                out.append(f"“{p.text or p.who}” cannot be met; remove it or make it a preference.")
        return out or [f"The hard preferences ({len(hard)}) cannot all be met together; make some of them preferences."]

    def status(self) -> dict:
        j = dict(self.job)
        if j.get("running"):
            j["elapsed"] = round(time.time() - j["started"], 1)
        return j

    def publish(self, version: int) -> Version:
        v = next((x for x in self.state.versions if x.version == version), None)
        if v is None:
            raise KeyError(version)
        if v.timetable.report.hard_violations:
            raise ValueError("this version breaks hard rules and cannot be published")
        for x in self.state.versions:
            x.published = x.version == version
        self.save()
        return v

    # -- clubs -------------------------------------------------------------------------------

    def desk(self) -> ClubDesk:
        return ClubDesk(self.calendar, self.rooms, date.fromisoformat(self.state.semester_start))

    def club(self, req: ClubRequest, submit: bool, rules: list[dict], today: date | None = None) -> ClubDecision:
        v = self.published()
        if v is None:
            raise ValueError("there is no published semester timetable yet")
        today = today or (self.clock() if self.clock else None)
        d = self.desk().check(req, v.timetable.meetings, self.state.bookings, today, rules)
        if submit:
            self.state.bookings.append(d)
            self.save()
        return d

    def decide(self, booking: str, approve: bool) -> ClubDecision:
        d = next(b for b in self.state.bookings if b.id == booking)
        if d.status != "needs_approval":
            raise ValueError("only requests waiting for the office can be decided")
        d.status = "booked" if approve else "rejected"
        self.save()
        return d


def diff(before: Timetable, after: Timetable) -> list[dict]:
    """Sessions whose time or rooms changed between two versions."""
    def where(tt: Timetable) -> dict[str, tuple[str, list[str]]]:
        out: dict[str, tuple[str, list[str]]] = {}
        for m in tt.meetings:
            t, rooms = out.get(m.component, ("", m.rooms))
            out[m.component] = ((t + " + " if t else "") + f"{m.day} {m.start}", rooms)
        return out

    a, b = where(before), where(after)
    label = {m.component: (m.course, m.name, m.label) for m in after.meetings}
    rows = []
    for comp in sorted(set(a) | set(b)):
        if a.get(comp) != b.get(comp):
            course, name, lab = label.get(comp, (comp, "", ""))
            rows.append({"component": comp, "course": course, "name": name, "label": lab,
                         "before": f"{a[comp][0]} · {', '.join(a[comp][1])}" if comp in a else "—",
                         "after": f"{b[comp][0]} · {', '.join(b[comp][1])}" if comp in b else "—"})
    return rows
