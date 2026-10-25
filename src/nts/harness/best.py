"""Best of n: run one request several times (the model sampling) and keep the episode most runs agree on.

    out = best_of(harness, 3, agent, "move my Monday lecture to Friday", store, actor="F-103", week=5)
    out.status, out.episode_id      # the chosen run; every run is an episode in the telemetry

The choice, in order:
  1. the ending most runs reached (submitted, waiting, answered, escalated, declined); runs that broke
     (budget, stuck, model error) count only if every run broke
  2. among those, the draft most of them wrote (the same rules, whatever their ids)
  3. the cheapest, by what the kernel says the draft does: other teachers whose classes move, rules
     others were asked to waive, wishes left unmet, classes moved; then fewer refused calls and tokens

Each run gets a "selection" span on its trace: the group, whether it was chosen, the votes and its
cost. A run not chosen that wrote a different draft is labelled bad for training (``Episode.label``),
which pairs it with the chosen one when the chosen one is judged good (``nts.data.outcome_pairs``).

Talk: the runs answer asks from their own copy of ``people``; with no people, a run that asks waits
like any episode, and only the chosen one's ask reaches anyone.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass

from nts.people import People
from nts.semester import unmet
from nts.state import Conceded, ConstraintAdded, ConstraintRemoved, Store, TimetableSet
from nts.telemetry import new_id

from .agent import AgentConfig
from .loop import Harness, Outcome

ENDINGS = ("submitted", "waiting", "answered", "escalated", "declined")  # ties go to the earlier one
BROKEN = ("budget_exhausted", "stuck", "model_error")


@dataclass(frozen=True)
class Cost:
    others: int = 0  # other teachers with a class moved
    waivers: int = 0  # rules others agreed to waive
    unmet: int = 0  # wishes broken in the weeks it changes
    moved: int = 0  # classes moved
    refused: int = 0
    tokens: int = 0

    def key(self) -> tuple:
        return (3 * self.others + 2 * self.waivers + self.unmet + self.moved, self.refused, self.tokens)


def cost(out: Outcome) -> Cost:
    """What the episode's draft does, measured on the draft itself (its timetables are the checked ones)."""
    draft = out.draft
    base, after = draft.base, draft.world
    inst = after.instance
    moved: set[str] = set()
    broken = 0
    for ch in draft.changes:
        if not isinstance(ch, TimetableSet):
            continue
        now = base.timetable(ch.week)
        moved |= {sid for sid, p in ch.assignment.items() if now is not None and now.assignment.get(sid) != p}
        broken += len(unmet(inst, ch.assignment, after.constraints.values(), ch.week))
    others = {inst.session_by_id[sid].faculty for sid in moved} - {draft.actor}
    return Cost(others=len(others), waivers=sum(isinstance(c, Conceded) for c in draft.changes), unmet=broken,
                moved=len(moved), refused=out.refused, tokens=out.usage.prompt_tokens + out.usage.completion_tokens)


def draft_key(out: Outcome) -> str:
    """The rules a draft adds and removes, without the ids an episode gives them: equal for the same draft."""
    if out.status == "waiting" and out.pending is not None:
        return json.dumps(["ask", out.pending.kind, out.pending.to, out.pending.rule])
    rules = []
    for ch in out.draft.changes:
        if isinstance(ch, ConstraintAdded):
            c = ch.constraint.model_dump(mode="json", exclude={"id", "source"})
            rules.append(json.dumps(["add", c], sort_keys=True))
        elif isinstance(ch, ConstraintRemoved):
            rules.append(json.dumps(["remove", ch.id]))
    return json.dumps(sorted(rules))


def choose(outs: list[Outcome]) -> tuple[int, dict]:
    """The index of the run to keep, and why (the votes)."""
    ok = [i for i, o in enumerate(outs) if o.status not in BROKEN] or list(range(len(outs)))
    endings = Counter(outs[i].status for i in ok)
    rank = {s: n for n, s in enumerate(ENDINGS)}
    ending = max(endings, key=lambda s: (endings[s], -rank.get(s, len(ENDINGS))))
    same = [i for i in ok if outs[i].status == ending]
    drafts = Counter(draft_key(outs[i]) for i in same)
    costs = {i: cost(outs[i]) for i in same}
    top = max(drafts.values())
    best = min((i for i in same if drafts[draft_key(outs[i])] == top), key=lambda i: (costs[i].key(), i))
    return best, {"endings": dict(endings), "drafts": sorted(drafts.values(), reverse=True),
                  "agree": drafts[draft_key(outs[best])], "of": len(outs)}


def best_of(harness: Harness, n: int, agent: AgentConfig, request: str, store: Store, actor: str,
            week: int | None = None, *, people: People | None = None, episode_id: str | None = None) -> Outcome:
    if n <= 1:
        return harness.run(agent, request, store, actor, week, episode_id=episode_id, people=people)
    group = episode_id or new_id("ep-")
    outs = [harness.run(agent, request, store, actor, week, episode_id=group if k == 0 else f"{group}.{k + 1}",
                        people=copy.deepcopy(people)) for k in range(n)]
    best, votes = choose(outs)
    chosen_key = draft_key(outs[best])
    for k, o in enumerate(outs):
        c = cost(o)
        with harness.tracer.span("selection", f"run {k + 1} of {n}", trace_id=o.episode_id, group=group,
                                 chosen=k == best, chosen_id=outs[best].episode_id, same_draft=draft_key(o) == chosen_key,
                                 status=o.status, cost={**c.__dict__, "score": c.key()[0]}, **votes):
            pass
    return outs[best]
