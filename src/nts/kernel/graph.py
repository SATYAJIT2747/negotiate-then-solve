"""Stakeholder-constraint knowledge graph (proposal Section 8.8, Figure 5).

Nodes are stakeholders, constraints, resources (sessions, rooms, groups),
requests/rules and timetable changes. Edges carry the relation:

owns, references, has_authority_over, reports_to, derived_from,
conflicts_with (from MUS results), affects, and a ``balance`` attribute on
stakeholders (fairness credit from the ledger).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx

from .instance import Instance
from .schemas import Constraint, Placement, Role


def build_graph(instance: Instance, constraints: Iterable[Constraint] = (),
                credits: Mapping[str, float] | None = None) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for f in instance.faculty:
        g.add_node(f.id, kind="stakeholder", role=f.role.value, name=f.name,
                   balance=(credits or {}).get(f.id, 0.0))
        if f.reports_to:
            g.add_edge(f.id, f.reports_to, rel="reports_to")
    for grp in instance.groups:
        g.add_node(grp.id, kind="resource", type="group")
        g.add_node(f"ST-{grp.id}", kind="stakeholder", role=Role.STUDENT.value)
        g.add_edge(f"ST-{grp.id}", grp.id, rel="member_of")
    for r in instance.rooms:
        g.add_node(r.id, kind="resource", type="room")
    g.add_node("S-LAB", kind="stakeholder", role=Role.LAB_INCHARGE.value)
    for r in instance.rooms:
        if r.type.value == "lab":
            g.add_edge("S-LAB", r.id, rel="has_authority_over")
    hods = [f.id for f in instance.faculty if f.role == Role.HOD]
    for s in instance.sessions:
        g.add_node(s.id, kind="resource", type="session")
        g.add_edge(s.faculty, s.id, rel="has_authority_over")
        for h in hods:
            g.add_edge(h, s.id, rel="has_authority_over")
        for grp in s.groups:
            g.add_edge(s.id, grp, rel="attended_by")
    for c in constraints:
        add_constraint(g, c)
    return g


def add_constraint(g: nx.MultiDiGraph, c: Constraint) -> None:
    g.add_node(c.id, kind="constraint", tier=int(c.tier), type=c.type.value)
    if c.owner:
        g.add_edge(c.owner, c.id, rel="owns")
    for target in (c.scope.session, c.scope.room, c.scope.group, c.scope.faculty):
        if target:
            g.add_edge(c.id, target, rel="references")
    src = c.source.request or c.source.rule
    if src:
        g.add_node(src, kind="request" if c.source.request else "rule")
        g.add_edge(c.id, src, rel="derived_from")


def add_conflict(g: nx.MultiDiGraph, mus: Iterable[str]) -> None:
    mus = list(mus)
    for a in mus:
        for b in mus:
            if a != b:
                g.add_edge(a, b, rel="conflicts_with")


def add_change(g: nx.MultiDiGraph, change_id: str, moved: Iterable[str]) -> None:
    g.add_node(change_id, kind="change")
    for sid in moved:
        g.add_edge(change_id, sid, rel="affects")


def _out(g: nx.MultiDiGraph, node: str, rel: str) -> list[str]:
    return [v for _, v, d in g.out_edges(node, data=True) if d.get("rel") == rel]


def _in(g: nx.MultiDiGraph, node: str, rel: str) -> list[str]:
    return [u for u, _, d in g.in_edges(node, data=True) if d.get("rel") == rel]


def owner_of(g: nx.MultiDiGraph, cid: str) -> str | None:
    owners = _in(g, cid, "owns")
    return owners[0] if owners else None


def has_authority(g: nx.MultiDiGraph, stakeholder: str, resource: str) -> bool:
    return resource in _out(g, stakeholder, "has_authority_over")


def escalation_chain(g: nx.MultiDiGraph, stakeholder: str) -> list[str]:
    chain, seen = [], {stakeholder}
    node = stakeholder
    while (up := _out(g, node, "reports_to")) and up[0] not in seen:
        chain.append(up[0])
        seen.add(up[0])
        node = up[0]
    return chain


def conflicts(g: nx.MultiDiGraph, cid: str) -> list[str]:
    return sorted(set(_out(g, cid, "conflicts_with")))


def affected_stakeholders(g: nx.MultiDiGraph, change_id: str) -> set[str]:
    """Who to notify: faculty teaching the moved sessions and their sections."""
    people: set[str] = set()
    for sid in _out(g, change_id, "affects"):
        people |= {u for u in _in(g, sid, "has_authority_over") if g.nodes[u].get("role") == Role.FACULTY.value}
        for grp in _out(g, sid, "attended_by"):
            people |= set(_in(g, grp, "member_of"))
    return people


def moved_sessions(before: Mapping[str, Placement], after: Mapping[str, Placement]) -> list[str]:
    return sorted(s for s in set(before) | set(after) if before.get(s) != after.get(s))
