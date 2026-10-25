"""Who has given way, and how evenly: the concession ledger of a department, from its event log.

    rows, spread = standing(store.world)     # per teacher: times, burden; the Gini, max and CV of the burden

Each time a teacher agrees to waive one of their rules for someone else's request (``make_offer``), the
draft records a ``Conceded`` entry; once committed it is part of the log, so the ledger is the history
itself. The credit is the kernel's: the rule's importance (its tier and how it was justified) times
the size of the concession (waived for some weeks: partial; dropped: full).
"""

from __future__ import annotations

from nts.kernel.ledger import FULL, PARTIAL, ConcessionLedger, importance
from nts.kernel.schemas import ConcessionEntry, Constraint
from nts.state import World

TERM = "term"  # one store is one term; later terms would be labelled for the ledger's decay


def entry(c: Constraint, to: str, weeks: list[int]) -> ConcessionEntry:
    assert c.owner is not None
    return ConcessionEntry(stakeholder=c.owner, constraint_id=c.id, conceded_to=to, semester=TERM,
                           week=weeks[0] if len(weeks) == 1 else None,
                           credit=round(importance(c) * (PARTIAL if weeks else FULL), 4))


def ledger(world: World) -> ConcessionLedger:
    return ConcessionLedger(world.concessions)


def given_way(world: World, teacher: str) -> int:
    return sum(e.stakeholder == teacher for e in world.concessions)


def standing(world: World) -> tuple[list[dict], dict[str, float]]:
    inst = world.instance
    led = ledger(world)
    times, burden = led.times(), led.counts()
    people = [f.id for f in inst.faculty]
    rows = [{"teacher": inst.faculty_by_id[f].name, "given_way": times.get(f, 0), "burden": round(burden.get(f, 0.0), 3),
             "for": sorted({inst.faculty_by_id[e.conceded_to].name if e.conceded_to in inst.faculty_by_id else "the office"
                            for e in world.concessions if e.stakeholder == f})}
            for f in people]
    spread = {k: round(v, 3) for k, v in led.dispersion(people).items()}
    return sorted(rows, key=lambda r: (-r["burden"], r["teacher"])), spread
