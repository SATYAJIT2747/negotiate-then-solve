"""Requests that combine what the intent vocabulary says one piece at a time: a suite of hand-written messages
(``data/suites/compositional.jsonl``), scored by ``evaluation.suites`` like the generated ones.

    uv run python -m evaluation.compositional        # writes the suite
    uv run python -m evaluation.suites --parser local --only compositional ...

Each message needs one thing the vocabulary handles badly: times of day that differ by day, an exception to a
period, alternatives, a first and second choice, a relation between two classes, a limit other than hours in a
row, a fallback, a parity of weeks. Schema 6 says every one of them (the compositional request language
design: ``except_weeks`` and ``parity``, ``limit``, ``relate``, choices between items and a class named as
another item's); until schema 5 the gold was a question for nine of them. A reader that turns "two classes a
day" into "two hours in a row" is wrong without saying so, which ``silent_wrong`` counts.

All are from F-107 of the suites' department (``data/synthetic-cse-s0.json``), who teaches Database Systems
(three lectures and a practical) and Cloud Computing (three lectures), in week 14 of 16. Their classes sit as
``PLACED`` puts them: two Database Systems lectures on Monday, one Friday class at 9 am, a Thursday 4 pm lecture.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/suites/compositional.jsonl")
SENDER, WEEK, TODAY = "F-107", 14, "2026-10-28"
DB, CLOUD = "Database Systems", "Cloud Computing"
# session -> (day, start slot: 0 is 9 am, room)
PLACED = {"C-014-L1": ("Mon", 0, "R-101"), "C-014-L2": ("Mon", 2, "R-101"), "C-014-L3": ("Wed", 0, "R-102"),
          "C-014-P": ("Mon", 5, "L-1"), "C-106-L1": ("Wed", 1, "R-103"), "C-106-L2": ("Fri", 0, "R-101"),
          "C-106-L3": ("Thu", 7, "R-104")}


def _compile(*intents: dict) -> dict:
    return {"action": "compile", "intents": list(intents)}


# (kind: what the message needs, message, gold reading)
CASES: list[tuple[str, str, dict]] = [
    ("union_of_windows", "No classes on Monday mornings or Friday afternoons, please.",
     _compile({"kind": "avoid", "time": {"days": ["Mon"], "part": "morning"}},
              {"kind": "avoid", "time": {"days": ["Fri"], "part": "afternoon"}})),
    ("except_weeks", "I can't teach on Wednesdays from now on, except in week 16.",
     _compile({"kind": "unavailable", "time": {"days": ["Wed"]},
               "period": {"relative": "rest_of_semester", "except_weeks": [16]}})),
    ("or_over_days", f"Could you move my {DB} practical to Tuesday or Thursday afternoon?",
     _compile({"kind": "move", "which": {"course": DB, "kind": "practical"},
               "to": {"days": ["Tue", "Thu"], "part": "afternoon"}})),
    ("or_over_windows", f"Could you move my {DB} practical to Tuesday morning or Thursday afternoon?",
     _compile({"kind": "move", "id": "a", "which": {"course": DB, "kind": "practical"},
               "to": {"days": ["Tue"], "part": "morning"}},
              {"kind": "move", "id": "b", "which": {"same": "a"}, "to": {"days": ["Thu"], "part": "afternoon"}},
              {"kind": "any_of", "of": ["a", "b"]})),
    ("ranked", "Ideally move my Friday 9 am lecture to Thursday morning; if that doesn't work, Tuesday morning "
               "is fine.",
     _compile({"kind": "move", "id": "a", "which": {"kind": "lecture", "day": "Fri", "time": "09:00"},
               "to": {"days": ["Thu"], "part": "morning"}},
              {"kind": "move", "id": "b", "which": {"same": "a"}, "to": {"days": ["Tue"], "part": "morning"}},
              {"kind": "ranked", "of": ["a", "b"]})),
    ("relation_after", f"The {DB} practical should be on the same day as one of its lectures, right after it.",
     _compile({"kind": "relate", "which": {"course": DB, "kind": "practical"}, "relation": "right_after",
               "other": {"course": DB, "kind": "lecture"}, "any": True})),
    ("relation_different_day", f"Please don't put two {DB} lectures on the same day.",
     _compile({"kind": "relate", "which": {"course": DB, "kind": "lecture"}, "relation": "different_day"})),
    ("limit_per_day", "No more than two classes a day for me, please.",
     _compile({"kind": "limit", "measure": "classes_per_day", "n": 2})),
    ("limit_days_per_week", "I'd like to teach on at most three days a week.",
     _compile({"kind": "limit", "measure": "days_per_week", "n": 3})),
    ("limit_gap", "Leave at least an hour free between any two of my classes.",
     _compile({"kind": "limit", "measure": "gap", "n": 1})),
    ("fallback", "Move my Friday lecture in week 15 to Thursday; if that isn't possible, cancel it.",
     _compile({"kind": "move", "id": "a", "which": {"kind": "lecture", "day": "Fri"}, "to": {"days": ["Thu"]},
               "period": {"weeks": [15]}},
              {"kind": "cancel", "id": "b", "which": {"same": "a"}, "period": {"weeks": [15]}},
              {"kind": "fallback", "of": ["a", "b"]})),
    ("day_per_week", "I'm away on Tuesday in week 15 and on Thursday in week 16.",
     _compile({"kind": "unavailable", "time": {"days": ["Tue"]}, "period": {"weeks": [15]}},
              {"kind": "unavailable", "time": {"days": ["Thu"]}, "period": {"weeks": [16]}})),
    ("parity", "On even weeks I can't teach on Friday afternoons.",
     _compile({"kind": "unavailable", "time": {"days": ["Fri"], "part": "afternoon"}, "period": {"parity": "even"}})),
    ("binding_relation", f"Move my {CLOUD} Thursday 4 pm lecture to the morning, and keep it on a different day "
                         f"from my other {CLOUD} lectures.",
     _compile({"kind": "move", "id": "a", "which": {"course": CLOUD, "kind": "lecture", "day": "Thu", "time": "16:00"},
               "to": {"part": "morning"}},
              {"kind": "relate", "which": {"same": "a"}, "relation": "different_day",
               "other": {"course": CLOUD, "kind": "lecture"}})),
]


def rows(instance_path: Path = Path("data/synthetic-cse-s0.json")) -> list[dict]:
    from core.instance import Instance
    from core.schemas import Channel, Request, Role
    from language.intent_compiler import format_row
    from language.intents import SCHEMA_VERSION

    instance = Instance.model_validate_json(instance_path.read_text(encoding="utf-8"))
    assert {s.id for s in instance.sessions if s.faculty == SENDER} == set(PLACED), "the department changed"
    timetable = {sid: {"day": d, "slot": q, "room": r} for sid, (d, q, r) in PLACED.items()}
    out = []
    for i, (kind, text, gold) in enumerate(CASES, 1):
        rid = f"I-compositional-{i:02d}"
        request = Request(id=rid, channel=Channel.PORTAL, sender_id=SENDER, role=Role.FACULTY, raw_text=text,
                          received_at=f"{TODAY}T09:00:00")
        row = {"id": rid, "source": "hand", "kind": kind, "schema": SCHEMA_VERSION, "paraphrased": False,
               "messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""},
                            {"role": "assistant", "content": json.dumps(gold)}],
               "request": request.model_dump(mode="json"), "current_week": WEEK, "today": TODAY,
               "timetable": timetable, "origin": "evaluation.compositional"}
        out.append(format_row(instance, row))  # the current prompt and context, the gold written with names
    return out


def main() -> None:
    written = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written), encoding="utf-8")
    print(f"wrote {len(written)} rows to {OUT}")


if __name__ == "__main__":
    main()
