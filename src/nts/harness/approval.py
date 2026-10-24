"""The human step after ``submit``: approve (commit) or reject a submitted draft, recorded on the episode.

    approve(store, outcome, tracer, by="office")    # the new World, or None if the kernel refused it
    reject(outcome, tracer, by="office", note="Not in exam week.")

Each decision is an "approval" span in the episode's trace, so it labels the episode for training:
approved and committed, refused by the kernel at commit, or rejected by a person (with the note).
"""

from __future__ import annotations

from nts.state import InvalidChange, StaleDraft, Store, World
from nts.telemetry import Tracer

from .loop import Outcome


def approve(store: Store, outcome: Outcome, tracer: Tracer, *, by: str = "office") -> World | None:
    if outcome.status != "submitted":
        raise ValueError(f"episode {outcome.episode_id} was not submitted ({outcome.status})")
    with tracer.span("approval", by, trace_id=outcome.episode_id, by=by, base_version=outcome.draft.base.version) as span:
        try:
            world = store.commit(outcome.draft)
        except (InvalidChange, StaleDraft) as e:
            span.attrs.update(approved=False, committed=False, problems=getattr(e, "problems", [str(e)]),
                              reason=type(e).__name__)
            return None
        span.attrs.update(approved=True, committed=True, version=world.version)
        return world


def reject(outcome: Outcome, tracer: Tracer, *, by: str = "office", note: str = "") -> None:
    with tracer.span("approval", by, trace_id=outcome.episode_id, by=by, base_version=outcome.draft.base.version,
                     approved=False, committed=False, note=note):
        pass
