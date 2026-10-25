"""The semester pipeline: offering document -> sessions -> timetable ->
changes -> club bookings. The real offering PDF is parsed in the fast tests;
full builds of it are slow and marked so."""

import base64
import time
from datetime import date
from pathlib import Path

import pytest

from semester.clubs import ClubDesk, ClubRequest
from semester.offerings import elective_pools, parse_offerings, parse_rows, text_rows
from semester.requests import parse
from semester.solver import (
    Meeting,
    Preference,
    SemesterCalendar,
    SemesterSolver,
    Window,
    components,
    default_rooms,
    verify,
)

OFFERINGS = Path("course-offering/CourseOfferings-M26-V7.pdf")
needs_pdf = pytest.mark.skipif(not OFFERINGS.exists(), reason="the course offering PDF is not in the repository")

SMALL = """B.Tech I year I Semester - CSE&CSD
Pr MA MA5.101 Discrete Structures 3-1-0-4 Deepak S + Harsha K
Pr CS CS0.101 Computer Programming 3-1-3-5 Aarav Kulkarni + Rahul Desai
Pr EC EC2.101 Digital Systems and Microcontrollers 3-1-3-5 Meera Joshi
In HS OC2.101 Arts-1 (H1) 2-0-0-2 Saroja T K (Coordinator)
In HS OC1.101 Sports-1 0-2-0-1 Physical Education Centre
B.Tech I year I Semester - ECE&ECD
Pr MA EC5.101 Networks, Signals and Systems 3-1-0-4 Arjun Menon
Pr CS CS0.101 Computer Programming 3-1-3-5 Aarav Kulkarni + Rahul Desai
Pr EC EC2.102 Electronic Workshop-1 (H2) 2-0-3-2 Sameer Patil
Robotics Stream
CS7.503 Mobile Robotics 3-1-0-4 R Vikram Sood
EC4.401 Robotics: Dynamics and Control 3-1-0-4 Rohan Mathew
"""


@pytest.fixture(scope="module")
def small():
    return parse_rows(text_rows(SMALL), "small.txt")


@needs_pdf
def test_real_offering_document():
    doc = parse_offerings(OFFERINGS)
    assert len(doc.courses) >= 140 and len(doc.cohorts) >= 35 and len(doc.pools) >= 15
    cp = doc.courses["CS0.101"]
    assert (cp.L, cp.T, cp.P, cp.C) == (3, 1, 3, 5)
    # names are not asserted (no real people in the code); the structure is what parsing gets right or wrong
    assert len(cp.faculty) == 3 and all(len(f.split()) == 2 for f in cp.faculty)  # wrapped over two lines
    assert len(cp.cohorts) == 6
    assert len(doc.courses["EC2.101"].faculty[-1].split()) == 2  # a three-line cell: the name split over lines is rejoined
    cl = doc.courses["CL1.101"].faculty
    assert len(cl) == 2 and all(len(f.split()) == 2 for f in cl)  # the next row's surname is not stolen
    assert doc.courses["OC2.101"].half == "H1" and doc.courses["HS7.101"].half == "H2"
    assert doc.courses["HS0.203a"].name.startswith("Basics of Ethics") and len(doc.courses["HS0.203a"].cohorts) == 8
    assert not doc.courses["OC1.101"].scheduled and not doc.courses["CS9.402"].scheduled
    pools = {p.name: p for p in doc.pools}
    assert pools["Humanities Electives for UG3 &UG4"].cap == 40
    assert any(p.cap == 150 for n, p in pools.items() if n.startswith("Bouquet"))
    # a programme with only elective slots still has something to show: the pools behind them
    names = {p.id: p.name for p in doc.pools}
    mtech = next(c for c in doc.cohorts if c.name.startswith("M.Tech II year I Semester - CSE"))
    assert not mtech.courses
    links = {slot: [names[i] for i in ids] for slot, ids in elective_pools(doc, mtech)}
    assert links["Bouquet Core"][0].startswith("Bouquet") and links["Area Elective"] == ["CSE/Open Electives"]
    ece = next(c for c in doc.cohorts if c.name == "B.Tech III year I Semester - ECE")
    assert dict(elective_pools(doc, ece))["EC ECE-Honours-1 (Optional)"] == []  # project work, no classes


def test_text_rows_and_needs(small):
    assert set(small.courses) == {"MA5.101", "CS0.101", "EC2.101", "OC2.101", "OC1.101", "EC5.101", "EC2.102",
                                  "CS7.503", "EC4.401"}
    assert small.courses["CS0.101"].cohorts == ["B-TECH-I-YEAR-I-SEMESTER-CSE-CSD", "B-TECH-I-YEAR-I-SEMESTER-ECE-ECD"]
    assert small.pools[0].courses == ["CS7.503", "EC4.401"]
    comps, notes = components(small, {c.id: 180 for c in small.cohorts}, default_rooms(), SemesterCalendar())
    by = {c.id: c for c in comps}
    assert len(by["CS0.101/L"].options[0].intervals) == 2  # 3 lecture hours: a day and the day three later
    assert len(by["CS0.101/L"].rooms) == 2  # 360 students: two parallel sections
    assert by["CS0.101/P.1"].room_type == "computing" and by["EC2.101/P.1"].room_type == "electronics"
    assert len(by["OC2.101/L"].options[0].intervals) == 1 and by["OC2.101/L"].half == "H1"  # 2 hours, H1: once a week
    assert "OC1.101/L" not in by and any("Physical Education" in n for n in notes)
    free = (14 * 60, 18 * 60 + 40)
    assert all(not any(d in ("Wed", "Sat") and a < free[1] and free[0] < b for d, a, b in o.intervals)
               for c in comps for o in c.options)  # the free afternoons stay free


def test_small_build_is_valid_and_honours_hard_preferences(small):
    hard = Preference(id="p1", target="faculty", who="Arjun Menon", mode="avoid", days=["Mon", "Tue"], hard=True)
    soft = Preference(id="p2", target="cohort", who="B-TECH-I-YEAR-I-SEMESTER-CSE-CSD", mode="avoid", start="17:00")
    s = SemesterSolver(small, preferences=[hard, soft], time_limit=20)
    tt = s.solve()
    assert tt.meetings and tt.report.hard_violations == []
    assert all(m.day not in ("Mon", "Tue") for m in tt.meetings if m.course == "EC5.101")
    assert all(p["met"] for p in tt.report.preferences)
    assert tt.report.pool_clashes == 0
    # repair: close the room the CS0.101 lectures use; only what must move moves
    room = next(m.rooms[0] for m in tt.meetings if m.component == "CS0.101/L")
    day = next(m.day for m in tt.meetings if m.component == "CS0.101/L")
    closed = SemesterSolver(small, preferences=[hard, soft], time_limit=20,
                            room_closures=[(room, Window(days=[day], start="00:00", end="23:59"))])
    tt2 = closed.solve(baseline=tt)
    assert tt2.report.hard_violations == []
    assert all(room not in m.rooms for m in tt2.meetings if m.day == day)
    assert tt2.report.moved <= 3


def test_verify_allows_h1_and_h2_to_share_a_slot():
    rooms = default_rooms()
    a = Meeting(component="A/L", course="A", name="A", kind="lecture", label="L", day="Mon", start="08:30",
                end="09:55", rooms=["H101"], faculty=["x"], cohorts=["c"], half="H1")
    b = a.model_copy(update={"component": "B/L", "course": "B", "half": "H2"})
    assert verify([a, b], rooms, {}, SemesterCalendar()) == []
    c = a.model_copy(update={"component": "C/L", "course": "C", "half": "full"})
    assert any("clash" in p for p in verify([a, c], rooms, {}, SemesterCalendar()))


def test_parse_requests(small):
    rooms = default_rooms()
    p = parse("Arjun Menon is unavailable on Fridays", small, rooms)
    assert p.preferences[0].hard and p.preferences[0].days == ["Fri"] and p.preferences[0].target == "faculty"
    p = parse("UG1 CSE students would like no classes after 5 pm on Saturday", small, rooms)
    assert p.preferences[0].who == "B-TECH-I-YEAR-I-SEMESTER-CSE-CSD" and p.preferences[0].start == "17:00"
    assert not p.preferences[0].hard and p.preferences[0].mode == "avoid"
    p = parse("CS0.101 should be in the morning", small, rooms)
    assert p.preferences[0].mode == "prefer" and p.preferences[0].end == "13:05"
    p = parse("Room H105 is closed on Tuesdays", small, rooms)
    assert p.closures[0][0] == "H105" and p.closures[0][1].days == ["Tue"]
    assert parse("make it nicer", small, rooms).error


def test_club_checks():
    cal = SemesterCalendar()
    lec = Meeting(component="X/L", course="X", name="X", kind="lecture", label="Lecture", day="Thu", start="17:10",
                  end="18:40", rooms=["H101"], faculty=[], cohorts=["ug1"], half="full")
    desk = ClubDesk(cal, default_rooms(), date(2026, 7, 27))
    today = date(2026, 9, 28)
    ok = desk.check(ClubRequest(club="Robotics", activity="Build night", date="2026-10-01", start="19:00", end="21:00",
                                attendees=40, cohorts=["ug1"]), [lec], [], today)
    assert ok.status == "booked" and ok.room and ok.week == 10
    early = desk.check(ClubRequest(club="Music", activity="Practice", date="2026-10-01", start="17:30", end="19:00",
                                   attendees=20, cohorts=["ug1"]), [lec], [ok], today)
    assert early.status == "rejected" and not early.checks[0].ok and early.alternatives
    busy = desk.check(ClubRequest(club="Quiz", activity="Quiz", date="2026-10-01", start="19:30", end="20:30",
                                  attendees=20, room=ok.room), [lec], [ok], today)
    assert busy.status == "rejected" and "booked for Robotics" in next(c.detail for c in busy.checks if c.name == "Room")
    big = desk.check(ClubRequest(club="Cultural", activity="Fest", date="2026-10-08", start="18:45", end="21:30",
                                 attendees=260, room="Auditorium"), [lec], [], today)
    assert big.status == "needs_approval"


def test_semester_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from web.app import COORDINATOR, World, create_web_app

    monkeypatch.setenv("NTS_SEMESTER_DIR", str(tmp_path))
    monkeypatch.setenv("NTS_SEMESTER_TIME", "20")

    class Monday12October(date):  # the club booking below is three days ahead, whenever the test runs
        @classmethod
        def today(cls):
            return date(2026, 10, 12)

    monkeypatch.setattr("semester.clubs.date", Monday12October)
    c = TestClient(create_web_app(World(seed_history=False)))
    office, faculty, student = {"X-User": COORDINATOR}, {"X-User": "F-102"}, {"X-User": "ST-G-01"}
    assert c.get("/api/semester", headers=faculty).status_code == 403
    html = "<table>" + "".join(f"<tr><td>{r}</td></tr>" for r in SMALL.strip().splitlines()) + "</table>"
    r = c.post("/api/semester/offerings", json={"name": "offerings.html", "data": base64.b64encode(html.encode()).decode()},
               headers=office).json()
    assert r["courses"] == 9 and r["cohorts"] == 2
    assert c.post("/api/semester/preferences", json={"text": "Arjun Menon is unavailable on Fridays"},
                  headers=faculty).json()["ok"]
    assert c.post("/api/semester/build", json={}, headers=faculty).status_code == 403
    c.post("/api/semester/build", json={"note": "test"}, headers=office)
    for _ in range(120):
        st = c.get("/api/semester/status", headers=office).json()
        if not st["running"]:
            break
        time.sleep(0.5)
    assert st["version"] == 1, st
    tt = c.get("/api/semester/timetable", headers=office).json()
    assert tt["report"]["hard_violations"] == [] and all(m["day"] != "Fri" for m in tt["meetings"] if m["course"] == "EC5.101")
    assert c.post("/api/clubs", json={"request": {"club": "A", "activity": "B", "date": "2026-10-15", "start": "19:00",
                                                  "end": "20:00", "attendees": 20}}, headers=student).status_code == 409
    c.post("/api/semester/versions/1/publish", headers=office)
    d = c.post("/api/clubs", json={"request": {"club": "A", "activity": "B", "date": "2026-10-15", "start": "19:00",
                                               "end": "20:00", "attendees": 20}, "submit": True}, headers=student).json()
    assert d["status"] == "booked"
    assert c.get("/api/clubs", headers=faculty).status_code == 403
    assert len(c.get("/api/clubs", headers=student).json()) == 1


@needs_pdf
@pytest.mark.slow
def test_real_build_has_no_hard_violations():
    doc = parse_offerings(OFFERINGS)
    tt = SemesterSolver(doc, time_limit=60).solve()
    assert tt.meetings and tt.report.hard_violations == [] and tt.report.pool_clashes == 0


def test_demo_department_as_offerings():
    from core.generator import generate_department
    from semester.demo import offerings_from_instance, rooms_from_instance
    from semester.requests import parse

    inst = generate_department(seed=1, n_faculty=8, n_groups=4, courses_per_group=4, n_lecture_rooms=4, n_labs=3)
    doc = offerings_from_instance(inst)
    assert len(doc.courses) == 16 and [c.name for c in doc.cohorts] == ["Section 1", "Section 2", "Section 3", "Section 4"]
    assert doc.courses["CS1.101"].faculty == ["Dr. Menon"] and doc.courses["CS1.101"].P == 2
    rooms = rooms_from_instance(inst)
    assert {r.type for r in rooms} == {"lecture", "computing", "electronics"}
    who = {p.target: p.who for t in ("Dr. Menon prefers not to teach before 10 am", "Section 2 wants Fridays free",
                                     "CS1.101 should be in the morning") for p in parse(t, doc, rooms).preferences}
    assert who == {"faculty": "Dr. Menon", "cohort": "G-02", "course": "CS1.101"}


def test_demo_offerings_pdf_reads_back(tmp_path):
    from core.generator import generate_department
    from semester.demo import offerings_from_instance, offerings_pdf

    inst = generate_department(seed=1, n_faculty=8, n_groups=4, courses_per_group=4, n_lecture_rooms=4, n_labs=3)
    demo = offerings_from_instance(inst)
    pdf = tmp_path / "demo.pdf"
    pdf.write_bytes(offerings_pdf(demo))
    doc = parse_offerings(pdf)
    assert set(doc.courses) == set(demo.courses) and not doc.warnings
    assert all((c.L, c.T, c.P, c.faculty) == (demo.courses[k].L, demo.courses[k].T, demo.courses[k].P, demo.courses[k].faculty)
               for k, c in doc.courses.items())
    assert [c.name.rsplit(" ", 2)[-2:] for c in doc.cohorts] == [["Section", str(i)] for i in range(1, 5)]
