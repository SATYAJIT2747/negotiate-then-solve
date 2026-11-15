"""A message of several parts about one class is not split when a part is unclear: "my Computer Programming
practical needs the routers, and it has to be on Tuesday afternoon", from Dr. Rao, who teaches it to Sections A
and B, was split into a room change for both sections (solved, waiting for approval) and a move asking which
section, so the routers went ahead without the move and the answer reached only the move."""

import os

import pytest

from language.intents import Avoid, ClassRef, IntentOutput, Move, NeedsRoom, TimeSpec
from test_intent_routing import IntentStub

TEXT = "My Computer Programming practical needs the routers, and it has to be on Tuesday afternoon."


@pytest.fixture(scope="module")
def campus(tmp_path_factory):
    os.environ.setdefault("NTS_RETRIEVER", "bm25")
    from semester.campus import campus_offerings
    from semester.department import instance_from_offerings
    from web.world import World

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NTS_SEMESTER_TIME", "20")
        inst = instance_from_offerings(campus_offerings(), {}, name="Campus (demo)")
        w = World(parser_mode="offline", seed_history=False, deadline=5, instance=inst, campus=True,
                  semester_dir=tmp_path_factory.mktemp("semester"))
    w.orch.system_one = None
    w.orch.parser = IntentStub(w.instance, w.orch.parser)
    w.orch.current_week = lambda: 5
    return w


def submit(w, text, intents):
    w.orch.parser.known[text] = IntentOutput(action="compile", intents=intents)
    return w.orch.submit(w.intake.from_portal("F-103", text))


PRACTICAL = ClassRef(course="CS0.101a", kind="practical")


def test_an_unclear_part_about_a_class_another_part_names_keeps_the_message_whole(campus):
    case = submit(campus, TEXT, [NeedsRoom(which=PRACTICAL, equipment=["routers"]),
                                 Move(which=PRACTICAL, to=TimeSpec(days=["Tue"], part="afternoon"), hard=True)])
    assert not case.parts and "+split" not in case.route, (case.route, case.reply)
    assert case.status.value == "clarification_requested", (case.status, case.reply)
    assert "Section A" in case.reply and "Section B" in case.reply, case.reply
    assert case.proposal is None  # nothing went ahead before the answer


def test_an_unclear_part_about_another_class_is_still_split_off(campus):
    """The rest of the message goes ahead when the unclear part is about something else."""
    text = "Keep my Friday mornings free; also move my Computer Programming practical to Tuesday afternoon."
    case = submit(campus, text, [Avoid(time=TimeSpec(days=["Fri"], part="morning")),
                                 Move(which=PRACTICAL, to=TimeSpec(days=["Tue"], part="afternoon"))])
    assert "+split" in case.route and len(case.parts) == 2, (case.route, case.reply)
    assert campus.orch.cases[case.parts[1]].status.value == "clarification_requested"
