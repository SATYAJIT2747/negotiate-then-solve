"""A large demo campus as a course offering document: four years of B.Tech in
CSE and ECE, twelve sections, about 45 courses, regular and visiting faculty.

    uv run python -m semester.campus        # writes course-offering/CampusCourseOfferings-M26.pdf

Upload the PDF (or pick "Campus" when creating a world) and the web app builds
the people, sections, rooms and weekly timetable from it (``semester.department``).
The shape plants conflicts worth showing:

* first-year courses are taught to two sections at once, and third-year
  Economics to three: big classes that only the halls fit;
* final-year electives are shared by CSE and ECE, so both programmes' weeks
  have to leave the same hours free;
* visiting faculty teach electives and come in on few days (their requests
  in the demo history say which);
* one lab has routers and several courses want it.

All names are invented.
"""

from __future__ import annotations

from collections import Counter

from .offerings import Cohort, Offering, OfferingDoc

SECTIONS = [  # (id, name, size)
    ("Y1-A", "B.Tech I year CSE+ECE - Section A", 66), ("Y1-B", "B.Tech I year CSE+ECE - Section B", 64),
    ("Y1-C", "B.Tech I year CSE+ECE - Section C", 65), ("Y1-D", "B.Tech I year CSE+ECE - Section D", 63),
    ("Y2-CSE-A", "B.Tech II year CSE - Section A", 60), ("Y2-CSE-B", "B.Tech II year CSE - Section B", 58),
    ("Y2-ECE", "B.Tech II year ECE", 60),
    ("Y3-CSE-A", "B.Tech III year CSE - Section A", 56), ("Y3-CSE-B", "B.Tech III year CSE - Section B", 55),
    ("Y3-ECE", "B.Tech III year ECE", 57),
    ("Y4-CSE", "B.Tech IV year CSE", 52), ("Y4-ECE", "B.Tech IV year ECE", 48),
]

# (code, name, L, T, P, sections, area of the teacher, visiting?)
COURSES: list[tuple[str, str, int, int, int, list[str], str, bool]] = []
for s, pair in (("a", ["Y1-A", "Y1-B"]), ("b", ["Y1-C", "Y1-D"])):
    COURSES += [
        (f"MA1.101{s}", "Calculus", 3, 1, 0, pair, "MA", False),
        (f"SC1.101{s}", "Engineering Physics", 3, 0, 2, pair, "SC", False),
        (f"CS0.101{s}", "Computer Programming", 3, 0, 3, pair, "CS", False),
        (f"EC0.101{s}", "Digital Systems", 3, 0, 2, pair, "EC", False),
        (f"HS1.101{s}", "Communication Skills", 2, 0, 0, pair, "HS", False),
    ]
for s, sec in (("a", "Y2-CSE-A"), ("b", "Y2-CSE-B")):
    COURSES += [
        (f"CS1.201{s}", "Data Structures", 3, 1, 2, [sec], "CS", False),
        (f"CS2.201{s}", "Computer Organisation", 3, 0, 0, [sec], "CS", False),
        (f"MA1.201{s}", "Discrete Mathematics", 3, 1, 0, [sec], "MA", False),
        (f"CS1.202{s}", "Object-Oriented Programming", 2, 0, 2, [sec], "CS", False),
    ]
COURSES += [
    ("MA1.202", "Probability and Statistics", 3, 0, 0, ["Y2-CSE-A", "Y2-CSE-B"], "MA", False),
    ("EC1.201", "Signals and Systems", 3, 1, 0, ["Y2-ECE"], "EC", False),
    ("EC2.202", "Electronic Circuits", 3, 0, 3, ["Y2-ECE"], "EC", False),
    ("EC2.203", "Network Theory", 3, 1, 0, ["Y2-ECE"], "EC", False),
    ("MA1.203", "Linear Algebra", 3, 0, 0, ["Y2-ECE"], "MA", False),
    ("CS1.203", "Programming for Engineers", 2, 0, 2, ["Y2-ECE"], "CS", False),
]
for s, sec in (("a", "Y3-CSE-A"), ("b", "Y3-CSE-B")):
    COURSES += [
        (f"CS3.301{s}", "Operating Systems", 3, 0, 2, [sec], "CS", False),
        (f"CS3.302{s}", "Computer Networks", 3, 0, 2, [sec], "CS", False),
        (f"CS3.303{s}", "Database Systems", 3, 0, 0, [sec], "CS", False),
        (f"CS3.304{s}", "Theory of Computation", 3, 1, 0, [sec], "CS", False),
    ]
COURSES += [
    ("HS3.301", "Economics", 2, 0, 0, ["Y3-CSE-A", "Y3-CSE-B", "Y3-ECE"], "HS", False),
    ("EC3.301", "Communication Systems", 3, 0, 2, ["Y3-ECE"], "EC", False),
    ("EC3.302", "VLSI Design", 3, 0, 2, ["Y3-ECE"], "EC", False),
    ("EC3.303", "Control Systems", 3, 1, 0, ["Y3-ECE"], "EC", False),
    ("EC3.304", "Embedded Systems", 2, 0, 3, ["Y3-ECE"], "EC", False),
    ("CS4.401", "Machine Learning", 3, 0, 2, ["Y4-CSE"], "CS", False),
    ("CS4.402", "Distributed Systems", 3, 0, 0, ["Y4-CSE"], "CS", False),
    ("CS4.403", "Information Security", 3, 0, 0, ["Y4-CSE"], "CS", False),
    ("EC4.401", "Wireless Communication", 3, 0, 0, ["Y4-ECE"], "EC", False),
    ("EC4.402", "Digital Signal Processing", 3, 0, 2, ["Y4-ECE"], "EC", False),
    ("CS4.410", "Cloud Computing", 3, 0, 0, ["Y4-CSE", "Y4-ECE"], "CS", True),
    ("EC4.410", "Internet of Things", 2, 0, 2, ["Y4-CSE", "Y4-ECE"], "EC", True),
    ("HS4.410", "Entrepreneurship", 2, 0, 0, ["Y4-CSE", "Y4-ECE"], "HS", True),
    ("CS4.411", "Natural Language Processing", 3, 0, 0, ["Y4-CSE"], "CS", True),
]

TEACHERS = {  # invented surnames, by area
    "CS": ["Rao", "Menon", "Iyer", "Khan", "Pillai", "Sharma", "Gupta", "Nair", "Reddy", "Banerjee",
           "Joshi", "Kulkarni", "Patel", "Shah"],
    "EC": ["Das", "Mukherjee", "Singh", "Verma", "Chatterjee", "Bose", "Kapoor", "Mehta", "Desai"],
    "MA": ["Naidu", "Hegde", "Kumar"],
    "SC": ["Agarwal", "Saxena"],
    "HS": ["Mishra", "Tiwari"],
}
VISITORS = {"CS": ["Fernandes", "Thomas"], "EC": ["Sinha"], "HS": ["Bhatt"]}


# what the campus's teachers and class reps asked of its semester timetable, added before the start-up build
# (``SemesterPlanner.bootstrap_demo``): hard and soft, one course only, "only after", a guest's days, a day with
# no classes anyway (Saturday), sections' own wishes. Dr. Fernandes's days cannot all be kept, so the build
# shows one wish it had to give up.
PREFERENCES = [
    ("Dr. Naidu", "Dr. Naidu is unavailable on Fridays because of HoD meetings."),
    ("Dr. Menon", "Dr. Menon prefers not to teach before 10 am."),
    ("Dr. Iyer", "Dr. Iyer can teach Data Structures only after 11:30 on Mondays."),
    ("Dr. Das", "Digital Systems should be in the morning."),
    ("Dr. Pillai", "Dr. Pillai has a research group meeting every Thursday 2-4 pm."),
    ("Dr. Sharma", "Dr. Sharma: Mondays are bad for me."),
    ("Dr. Fernandes", "Dr. Fernandes can only teach on Tuesdays and Thursdays."),
    ("Dr. Kulkarni", "Dr. Kulkarni would like Wednesday afternoons free."),
    ("Dr. Singh", "Dr. Singh would like Saturdays off."),
    ("Class rep, I year Section A", "B.Tech I year CSE+ECE - Section A would like no classes after 4 pm on Friday."),
    ("Class rep, III year ECE", "B.Tech III year ECE wants Tuesday afternoons free."),
    ("Class rep, IV year CSE", "B.Tech IV year CSE would rather not have classes at 8:30 on Monday."),
]


def campus_offerings() -> OfferingDoc:
    load: Counter[str] = Counter()
    courses: dict[str, Offering] = {}
    visitor_turn: Counter[str] = Counter()
    for code, name, L, T, P, secs, area, visiting in COURSES:
        if visiting:
            pool = VISITORS[area]
            who = f"Dr. {pool[visitor_turn[area] % len(pool)]} (Visiting)"
            visitor_turn[area] += 1
        else:
            who = "Dr. " + min(TEACHERS[area], key=lambda n: (load[n], TEACHERS[area].index(n)))
            load[who.removeprefix("Dr. ")] += L + T + P
        courses[code] = Offering(code=code, name=name, L=L, T=T, P=P, C=L + T + P // 2, faculty=[who],
                                 cohorts=list(secs), category=area)
    cohorts = [Cohort(id=i, name=n, size=size, courses=[c for c, o in courses.items() if i in o.cohorts])
               for i, n, size in SECTIONS]
    return OfferingDoc(source="Campus (demo): B.Tech CSE and ECE", semester="Monsoon 2026", courses=courses,
                       cohorts=cohorts)


def main() -> None:
    import argparse
    from pathlib import Path

    from .demo import offerings_pdf

    ap = argparse.ArgumentParser(description="Write the demo campus as a course offering PDF.")
    ap.add_argument("out", type=Path, nargs="?", default=Path("course-offering/CampusCourseOfferings-M26.pdf"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(offerings_pdf(campus_offerings(), title="B.Tech Course Offerings, Monsoon 2026"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
