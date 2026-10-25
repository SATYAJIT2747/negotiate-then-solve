"""Evaluate tools: ask the kernel what the draft would do.

``check_draft`` re-solves every timetable the draft's new rules touch, as a repair of the timetable
that runs there now (moving as few classes as it can), and says in words what moves, who else it
affects, which wishes are not met, or, when it cannot be done, which rules clash. A clashing rule
another teacher set is numbered, so the agent can ask them to waive it (``make_offer``). The timetables it
found are kept (``ctx.scratch["check"]``) for ``submit``, which accepts them only for the exact draft
version they were solved for.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from nts.kernel.schemas import Placement
from nts.kernel.solver import SolverTimeout, TimetableSolver
from nts.state import TimetableSet
from nts.state.store import touched

from . import words
from .read import _at
from .registry import Context, Refusal, Scope, tool
from .talk import number_theirs, waived

TIME_LIMIT = 20.0
MAX_LISTED = 15  # moves listed one by one; past that, a count


@dataclass
class Checked:
    version: str  # the draft version it was solved for
    ok: bool
    sets: list[TimetableSet]


class CheckDraft(BaseModel):
    pass


@tool(Scope.EVALUATE)
def check_draft(ctx: Context, args: CheckDraft) -> dict:
    """Ask the solver what the draft does: whether it can be done, which classes move, who else is affected,
    which wishes cannot be met, or which rules clash. Call it after changing the draft and before submitting."""
    if ctx.draft is None:
        raise Refusal("no_draft", "this episode has no draft")
    base, after = ctx.world, ctx.draft.world
    added = [c for cid, c in after.constraints.items() if cid not in base.constraints]
    removed = [c for cid, c in base.constraints.items() if cid not in after.constraints]
    if not added and not removed:
        raise Refusal("empty_draft", "the draft has no changes yet")
    ours = {c.id: n for n, ids in ctx.scratch.get("changes", {}).items() for c in added if c.id in ids}
    # removing or waiving a rule never breaks a timetable
    weeks = set().union(*(touched(after, c) for c in added if not waived(c.id)))
    if not weeks:
        ctx.scratch["check"] = Checked(after.version, True, [])
        return {"ok": True, "applies_to": "no timetable changes", "classes_moved": 0, "moves": [],
                "others_affected": [], "not_met": []}

    groups: dict[tuple, list[int | None]] = {}
    for w in weeks:
        key = (frozenset(c.id for c in after.active(w)), w if w in base.timetables else None)
        groups.setdefault(key, []).append(w)

    sets: list[TimetableSet] = []
    moved: dict[str, tuple[Placement, Placement]] = {}
    unmet: set[str] = set()
    for ws in groups.values():
        w = min(ws, key=lambda x: 0 if x is None else x)
        now = base.timetable(w)
        assert now is not None
        solver = TimetableSolver(after.instance, after.constraints.values(), week=w, baseline=now.assignment,
                                 skip=now.cancelled, time_limit=TIME_LIMIT, presolve=False)
        result = solver.solve()
        if result.status == "unknown":
            ctx.scratch["check"] = Checked(after.version, False, [])
            return {"ok": False, "why": "the solver ran out of time; try a smaller change"}
        if not result.ok:
            ctx.scratch["check"] = Checked(after.version, False, [])
            return _clash(ctx, solver, ours, ws)
        assert result.assignment is not None
        sets += [TimetableSet(week=x, assignment=result.assignment, cancelled=sorted(now.cancelled)) for x in ws]
        for sid, p in result.assignment.items():
            if now.assignment.get(sid) != p and sid not in moved:
                moved[sid] = (now.assignment[sid], p)
        unmet |= set(result.soft_violations) & set(ours)  # the wishes in this draft that the solver could not meet

    ctx.scratch["check"] = Checked(after.version, True, sets)
    inst = after.instance
    rows = [{"class": _at(inst, inst.session_by_id[sid], a), "now": _at(inst, inst.session_by_id[sid], b)}
            for sid, (a, b) in sorted(moved.items())]
    me = inst.faculty_by_id[ctx.actor].name if ctx.actor in inst.faculty_by_id else None
    others = sorted({r["class"]["teacher"] for r in rows} - {me}) + sorted({g for r in rows for g in r["class"]["sections"]})
    out = {"ok": True, "applies_to": _span(ws for ws in groups.values()), "classes_moved": len(rows),
           "moves": rows[:MAX_LISTED], "others_affected": others,
           "not_met": [words.constraint(inst, after.constraints[c]) for c in sorted(unmet)]}
    if len(rows) > MAX_LISTED:
        out["more_moves"] = len(rows) - MAX_LISTED
    return out


def _clash(ctx: Context, solver: TimetableSolver, ours: dict[str, int], ws: list[int | None]) -> dict:
    try:
        core = solver.core(solver.hard_ids) or []
    except SolverTimeout:
        core = []
    after = ctx.state
    inst = after.instance
    blocked = [] if None in ws else sorted(w for w in ws if w is not None)  # [] : the whole term

    def by(cid: str) -> dict:
        if cid in ours:
            return {"your_change": ours[cid]}
        owner = after.constraints[cid].owner
        if owner in inst.faculty_by_id and owner != ctx.actor:
            given = sum(e.stakeholder == owner for e in after.concessions)
            return {"set_by": inst.faculty_by_id[owner].name, "number": number_theirs(ctx, cid, blocked),
                    "has_given_way": f"{given} time{'s' * (given != 1)} this term"}
        return {"set_by": "already agreed"}

    rules = [{"rule": words.constraint(inst, after.constraints[c]), **by(c)} for c in core]
    return {"ok": False, "applies_to": _span([ws]),
            "why": "these rules cannot all hold at once" if rules else "the draft cannot be done",
            "clashing": rules,
            "next": "retract or soften (firm=false) one of your changes, ask the owner of a numbered rule to waive it "
                    "(make_offer), or decline or escalate the request"}


def _span(groups) -> str:
    ws = sorted({w for g in groups for w in g}, key=lambda x: 0 if x is None else x)
    if None in ws:
        return "the whole term" + (f" (and weeks {words.joined([str(w) for w in ws if w])})" if len(ws) > 1 else "")
    return f"week {ws[0]}" if len(ws) == 1 else f"weeks {words.joined([str(w) for w in ws])}"


EVALUATE_TOOLS = [check_draft]
