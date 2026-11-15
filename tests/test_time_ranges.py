"""A class longer than an hour says when it ends, so a move that pushes another class out is plain to see: a
practical moved to Wednesday 2 pm runs to 4 pm, over the 3 pm practical it displaced."""

from core.instance import Instance
from core.schemas import Placement
from agents.explainer import placement_text
from language.corpus import hour_range


def test_hour_range():
    assert hour_range(5, 2) == "2–4 pm"
    assert hour_range(2, 2) == "11 am–1 pm"
    assert hour_range(5, 1) == "2 pm"
    assert hour_range(2, 2, " to ") == "11 am to 1 pm"


def test_a_practical_says_when_it_ends_and_a_lecture_when_it_starts():
    from web.world import World

    inst: Instance = World(parser_mode="offline", seed_history=False).instance
    two = next(s for s in inst.sessions if s.duration == 2)
    one = next(s for s in inst.sessions if s.duration == 1)
    room = inst.rooms[0]
    at = Placement(day="Wed", slot=5, room=room.id)
    assert placement_text(inst, at, two.id) == f"Wednesday from 2 to 4 pm in {room.name}"
    assert placement_text(inst, at, one.id) == f"Wednesday at 2 pm in {room.name}"
    assert placement_text(inst, at) == f"Wednesday at 2 pm in {room.name}"
