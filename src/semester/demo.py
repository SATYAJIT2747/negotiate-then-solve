"""The demo department as a course offering document, so the semester planner
works on the same courses, teachers, sections and rooms as the rest of the
web app (instead of the institute's real offering PDF, whose faculty are not
the demo personas).

Each demo course becomes one offering: a code in the institute's format
(``CS1.101``; ``EC`` when its practical needs electronics, which gives it an
electronics lab), L-T-P from the hours of its lectures, tutorials and
practicals, its teacher's name as the faculty, and the sections that attend
as cohorts. Lecture rooms and labs keep their ids and capacities.
"""

from __future__ import annotations

from collections import defaultdict

from core.instance import Instance

from .offerings import CODE, Cohort, Offering, OfferingDoc
from .solver import Room


def course_code(instance: Instance, course: str) -> str:
    """``C-011`` (section 1, course 1) -> ``CS1.101``; ``EC`` if a practical needs electronics.
    A course already named in the institute's format (``CS3.301a``) keeps its code."""
    if CODE.fullmatch(course):
        return course
    digits = "".join(ch for ch in course if ch.isdigit())
    group, k = int(digits[:-1] or 0), int(digits[-1])
    electronics = any(s.course == course and "electronics" in s.equipment for s in instance.sessions)
    return f"{'EC' if electronics else 'CS'}{group}.{100 + k:03d}"


def offerings_from_instance(instance: Instance, semester: str = "") -> OfferingDoc:
    hours: dict[str, dict[str, int]] = defaultdict(lambda: {"lecture": 0, "tutorial": 0, "practical": 0})
    teachers: dict[str, list[str]] = defaultdict(list)
    sections: dict[str, list[str]] = defaultdict(list)
    for s in instance.sessions:
        hours[s.course][s.kind.value] += s.duration
        name = instance.faculty_by_id[s.faculty].name if s.faculty in instance.faculty_by_id else s.faculty
        if name not in teachers[s.course]:
            teachers[s.course].append(name)
        for g in s.groups:
            if g not in sections[s.course]:
                sections[s.course].append(g)
    courses = {}
    for course in sorted(hours):
        code = course_code(instance, course)
        h = hours[course]
        courses[code] = Offering(code=code, name=instance.course_title(course), L=h["lecture"], T=h["tutorial"],
                                 P=h["practical"], C=h["lecture"] + h["tutorial"] + h["practical"] // 2,
                                 faculty=teachers[course], cohorts=sections[course], category=code[:2],
                                 note=f"demo course {course}")
    cohorts = [Cohort(id=g.id, name=g.name, size=g.size,
                      courses=[c for c, o in courses.items() if g.id in o.cohorts]) for g in instance.groups]
    source = instance.name if "demo" in instance.name.lower() else f"{instance.name} (demo department)"
    return OfferingDoc(source=source, semester=semester, courses=courses,
                       cohorts=cohorts)


def rooms_from_instance(instance: Instance) -> list[Room]:
    def kind(r) -> str:
        if r.type.value != "lab":
            return "lecture"
        return next((k for k in ("electronics", "science", "design") if k in r.equipment), "computing")

    return [Room(id=r.id, capacity=r.capacity, type=kind(r)) for r in instance.rooms]


# ---------------------------------------------------------------------------
# The same offerings as a PDF, laid out like the institute's document
# ---------------------------------------------------------------------------

# what the demo department's teachers and class representatives told the office before the semester was
# built (sender, as they said it): hard and soft, one course only, "only after", a programme's own wish.
# Dr. Gupta has not sent one, so the "not heard from" list has someone in it.
PREFERENCES = [
    ("Dr. Menon", "Dr. Menon prefers not to teach before 10 am."),
    ("Dr. Rao", "Dr. Rao is unavailable on Fridays because of HoD meetings."),
    ("Dr. Iyer", "Dr. Iyer can teach Compiler Design only after 11:30 on Mondays."),
    ("Dr. Khan", "Dr. Khan would like Saturdays off."),
    ("Dr. Das", "Digital Logic should be in the morning."),
    ("Dr. Pillai", "Dr. Pillai has a research group meeting every Thursday 2-4 pm."),
    ("Dr. Sharma", "Dr. Sharma: Mondays are bad for me."),
    ("Class rep, Section 1", "Section 1 students would like no classes after 4 pm on Friday."),
    ("Class rep, Section 3", "Section 3 wants Tuesday afternoons free."),
    ("Class rep, Section 4", "Section 4 would rather not have classes at 8:30 on Monday."),
]

YEAR_HEADING = "B.Tech II year I Semester - CSE"


def _esc(text: str) -> str:
    """Escape a PDF string literal: backslash and brackets."""
    return text.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")


def offerings_pdf(doc: OfferingDoc, title: str = "Course Offerings, Monsoon 2026") -> bytes:
    """A programme heading per section ("B.Tech II year I Semester - CSE Section 1") and one row per
    course (CD, code, name, L-T-P-C, faculty), as ``semester.offerings`` reads it; A4 pages, as many
    as the rows need. Written directly (text and ruled lines), so no PDF library is needed."""
    cols = [40, 72, 142, 380, 444, 555]  # CD | code | course name | L-T-P-C | faculty | right edge
    pages: list[list[str]] = []
    ops: list[str] = []
    row_h, bottom = 16.0, 60.0
    state = {"y": 0.0, "top": 0.0}

    def text(x: float, y: float, s: str, bold: bool = False, size: float = 9.0) -> None:
        ops.append(f"BT /{'F2' if bold else 'F1'} {size} Tf {x:.1f} {y:.1f} Td ({_esc(s)}) Tj ET")

    def hline(y: float) -> None:
        ops.append(f"{cols[0]} {y:.1f} m {cols[-1]} {y:.1f} l S")

    def close_page() -> None:
        hline(state["y"] + 12)
        for x in cols:
            ops.append(f"{x} {state['top']:.1f} m {x} {state['y'] + 12:.1f} l S")
        text(cols[0], 30, f"Page {len(pages) + 1}. Generated from a demo department of the Negotiate, Then Solve "
                          "web app; all names are invented.", size=7.5)
        pages.append(ops[:])
        ops.clear()

    def open_page() -> None:
        y = 800.0
        if not pages:
            text(cols[0], y, f"{doc.source}: {title}", bold=True, size=13)
            y -= 16
            text(cols[0], y, "L-T-P-C: lecture, tutorial and practical hours per week, and credits. "
                             "Pr: programme core.", size=8.5)
            y -= 22
        state["top"] = y + 12
        ops.append("0.85 0.85 0.85 RG 0.6 w")
        hline(state["top"])
        for x, head in zip(cols, ["CD", "Code", "Course name", "L-T-P-C", "Faculty"], strict=False):
            text(x + 3, y, head, bold=True)
        state["y"] = y - row_h

    def room_for(rows: int) -> None:
        if state["y"] - rows * row_h < bottom:
            close_page()
            open_page()

    open_page()
    for cohort in doc.cohorts:
        room_for(2)  # a heading never ends a page alone
        y = state["y"]
        hline(y + 12)
        ops.append(f"0.93 0.93 0.93 rg {cols[0]} {y - 4:.1f} {cols[-1] - cols[0]} {row_h} re f 0 0 0 rg")
        heading = cohort.name if cohort.name.upper().startswith(("B.TECH", "M.TECH")) else f"{YEAR_HEADING} {cohort.name}"
        if cohort.size and "students)" not in heading:
            heading += f" ({cohort.size} students)"  # the reader takes the size from here (solver.cohort_size)
        text(cols[0] + 3, y, heading, bold=True)
        state["y"] -= row_h
        for code in cohort.courses:
            room_for(1)
            c, y = doc.courses[code], state["y"]
            hline(y + 12)
            for x, cell in zip(cols, ["Pr", c.code, c.name, f"{c.L}-{c.T}-{c.P}-{c.C}", " + ".join(c.faculty)],
                               strict=False):
                text(x + 3, y, cell)
            state["y"] -= row_h
    close_page()

    fonts = 3 + 2 * len(pages)  # object numbers: catalog 1, pages 2, then a page and its contents per page
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>",
                            b"<< /Type /Pages /Kids [" + " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages))).encode()
                            + b"] /Count " + str(len(pages)).encode() + b" >>"]
    for i, page in enumerate(pages):
        stream = "\n".join(page).encode("latin-1")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {4 + 2 * i} 0 R "
                        f"/Resources << /Font << /F1 {fonts} 0 R /F2 {fonts + 1} 0 R >> >> >>").encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def main() -> None:
    import argparse
    from pathlib import Path

    from core.generator import generate_department

    ap = argparse.ArgumentParser(description="Write the web app's demo department as a course offering PDF.")
    ap.add_argument("out", type=Path, nargs="?", default=Path("course-offering/DemoCourseOfferings-M26.pdf"))
    args = ap.parse_args()
    inst = generate_department(seed=1, n_faculty=8, n_groups=4, courses_per_group=4, n_lecture_rooms=4, n_labs=3)
    inst.name = "CSE department (demo)"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(offerings_pdf(offerings_from_instance(inst)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
