from core.generator import generate_department
from core.graph import add_change, affected_stakeholders, build_graph
from core.schemas import Role


def test_the_teacher_of_a_moved_class_is_told_whatever_their_role():
    inst = generate_department(seed=0)
    hods = {f.id for f in inst.faculty if f.role == Role.HOD}
    s = next(x for x in inst.sessions if x.faculty in hods)  # a class an HoD teaches
    g = build_graph(inst)
    add_change(g, "V2", [s.id])
    people = affected_stakeholders(g, "V2")
    assert s.faculty in people
    assert {f"ST-{grp}" for grp in s.groups} <= people
    assert not (hods - {s.faculty}) & people  # authority over the class is not teaching it
