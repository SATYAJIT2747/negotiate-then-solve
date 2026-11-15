"""The examination regulations, applied as a rule rather than left to a model (``data/policies/exam-regulations.pdf``).

    ruling = exam_ruling([7], {"EXAM-3.1": text, "EXAM-3.2": text})   # -> forward to the examination cell
    ruling = exam_ruling([8], rules)                                   # -> refuse: an examination week

Two rules of that document decide every change that touches the weeks around the examinations:

    EXAM-3.1  no regular classes are held in the examination weeks (8 and 16): a change in them is refused
    EXAM-3.2  no change may take effect in weeks 7, 8, 15 or 16 without the Controller of Examinations' written
              approval: a change in weeks 7 or 15 is forwarded to the examination cell

The week numbers are read from the rules' own text, so another calendar's regulations work unchanged, and nothing
applies to a department whose corpus has no EXAM-3.2 (the synthetic handbook alone). The request pipeline
(``pipeline.orchestrator``) calls it (the agentic branch's ``nts.exams`` is the same code, for the agent's
``nts.tools.end.submit``), so the model only has to read which weeks a request is for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

EXAMS, AROUND, EXTRA = "EXAM-3.1", "EXAM-3.2", "EXAM-3.5"
MAX_WEEKS = 4  # a change for more weeks than this is a standing one ("from now on", "the rest of the semester")
OFFICE = "the examination cell"
_WEEKS = re.compile(r"\bweeks?\s+((?:\d+)(?:\s*(?:,|or|and)\s*\d+)*)", re.IGNORECASE)


def weeks_in(text: str) -> set[int]:
    """The week numbers a rule names: "weeks 7, 8, 15 or 16", "week 8 ... week 16"."""
    return {int(n) for m in _WEEKS.finditer(text) for n in re.findall(r"\d+", m[1])}


@dataclass(frozen=True)
class ExamRuling:
    action: Literal["refuse", "forward"]
    weeks: list[int]  # the weeks of the request the rule is about
    cited: list[str] = field(default_factory=list)
    message: str = ""  # for the sender

    @property
    def reason(self) -> str:
        return f"{self.cited[0]}: week {', '.join(map(str, self.weeks))}"


def _weeks(ws: list[int]) -> str:
    return f"week {ws[0]}" if len(ws) == 1 else "weeks " + ", ".join(map(str, ws[:-1])) + f" and {ws[-1]}"


def exam_ruling(weeks: Iterable[int | None], rules: Mapping[str, str], *, extra: bool = False) -> ExamRuling | None:
    """What the examination regulations say about a change in ``weeks`` (``rules``: rule id -> text, the corpus in
    force). ``extra``: the change is an extra class (EXAM-3.5 allows revision classes in weeks 7 and 15 outside the
    timetable, still with the Controller's approval). None: the regulations are not loaded, or no week is near an
    examination."""
    if AROUND not in rules:
        return None
    asked = {w for w in weeks if w is not None}
    if len(asked) > MAX_WEEKS:  # a standing change: in force around the examinations too, where no class is held
        return None
    near = sorted(asked & weeks_in(rules[AROUND]))
    if not near:
        return None
    exam = sorted(set(near) & weeks_in(rules.get(EXAMS, "")))
    if exam:
        return ExamRuling("refuse", exam, [EXAMS, AROUND],
                          f"{_weeks(exam).capitalize()} {'is an examination week' if len(exam) == 1 else 'are examination weeks'}: "
                          "no regular classes are held then "
                          f"(§3.1 Examination weeks [{EXAMS}]), so the timetable cannot be changed for it.")
    cited = [AROUND] + ([EXTRA] if extra and EXTRA in rules else [])
    what = "An extra class" if extra else "A change"
    return ExamRuling("forward", near, cited,
                      f"{what} in {_weeks(near)} needs the written approval of the Controller of Examinations "
                      f"(§3.2 Timetable changes around examinations [{AROUND}]), so your request has been forwarded "
                      f"to {OFFICE}." + (" Revision classes then must be outside the regular timetable, at most two "
                                         f"hours per course, announced three working days ahead [{EXTRA}]."
                                         if EXTRA in cited else ""))
