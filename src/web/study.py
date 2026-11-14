"""Endpoints for the human study (``evaluation.study``): the team's labels for
judge validation and the anonymised faculty pilot.

A participant is known only by a code (``X-Study``), issued by the
coordinator. Labelling needs no sign-in to the portal; a pilot participant's
live session runs in the portal as the demo faculty member their code names,
and its records carry both the code and that persona.

The study belongs to the demo department, whichever world the browser is in: its
endpoints read the demo world (persona names, the practice clash), and a pilot's
live session is refused in any other world, so no live record comes from a
department its persona is not part of.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from evaluation.study import LIVE_STEPS, REPLY_LABELS, RULE_LABELS, SURVEY, TASKS, analyse, steps_done

from .wording import plain
from .worlds import DEMO, current_world

DEFAULT_PERSONA = "F-102"  # owns a practical with no special equipment: the practice clash needs one

CONSENT = [
    "Taking part is voluntary. You may stop at any time without giving a reason, and ask for your answers to be removed.",
    "You are known only by your participant code. We store no name, e-mail or other identifying detail.",
    "We store your ratings and labels, what you type to the assistant and in the practice session, your answers to the closing questions, and when you gave them.",
    "Please do not type anything personal (health, family or other private reasons) in the practice session.",
    "The answers are used only to evaluate this project, and are reported in aggregate.",
]


class ParticipantsBody(BaseModel):
    kind: Literal["team", "pilot"]
    n: int = Field(1, ge=1, le=20)
    persona: str | None = None


CLEAR_PHRASE = "delete all study data"


class ClearBody(BaseModel):
    confirm: str  # must be CLEAR_PHRASE, typed by the coordinator


class RevokeBody(BaseModel):
    delete_answers: bool = False  # a withdrawal: erase their labels and live records too


class LabelBody(BaseModel):
    task: str
    item: str
    value: dict


class LiveBody(BaseModel):
    kind: Literal["understood", "reply", "rating", "answer", "survey"]
    step: str | None = None  # the live-session step a check belongs to (evaluation.study.LIVE_STEPS)
    n: int | None = None  # the chat message an answer check is about
    case: str | None = None
    item: str | None = None  # inbox message id
    text: str | None = None
    reading: str | None = None
    confirmed: bool | None = None
    value: str | dict | None = None
    comment: str | None = Field(None, max_length=1000)


def _blind(task: str, it: dict) -> dict:
    """What a rater sees: never the configuration or the machine label."""
    if task == "claims":
        return {"id": it["id"], "claim": it["claim"], "facts": it["facts"]}
    if task == "replies":
        return {"id": it["id"], "message": plain(it["message"]), "reply": it["reply"]}
    if task == "rules":
        return {"id": it["id"], "query": it["query"], "answer": it["answer"], "rules": it["rules"]}
    return {"id": it["id"], "text": plain(it["text"])}  # as the portal shows it to the recipient


def _check(task: str, value: dict) -> dict:
    if task == "claims":
        if not isinstance(value.get("supported"), bool):
            raise HTTPException(422, "claims need supported: true or false")
        return {"supported": value["supported"]}
    if task == "rules":
        if value.get("label") not in RULE_LABELS:
            raise HTTPException(422, f"label must be one of {list(RULE_LABELS)}")
        return {"label": value["label"]}
    if task == "replies":
        if value.get("label") not in REPLY_LABELS:
            raise HTTPException(422, f"label must be one of {list(REPLY_LABELS)}")
        return {"label": value["label"]}
    try:
        c, a = int(value["clarity"]), int(value["acceptability"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, "ratings need clarity and acceptability (1-5)") from None
    if not (1 <= c <= 5 and 1 <= a <= 5):
        raise HTTPException(422, "ratings are 1 to 5")
    out = {"clarity": c, "acceptability": a}
    if value.get("comment"):
        out["comment"] = str(value["comment"])[:1000]
    return out


def register(app: FastAPI, demo: Callable, user: Callable, coordinator: Callable) -> None:
    """``demo`` gives the demo world: the study's people and practice clash live there."""
    W = demo  # noqa: N806 - read as the other modules' W(), but always the demo world

    def participant(x_study: str = Header(...)) -> dict:
        p = W().study.get(x_study)
        if p is None:
            raise HTTPException(401, "unknown participant code")
        return p

    def consented(p: dict = Depends(participant)) -> dict:
        if not p["consented"]:
            raise HTTPException(403, "consent first")
        return p

    def session(p: dict) -> dict:
        w = W()
        persona = p.get("persona")
        return p | {"persona_name": w.name_of(persona) if persona else None,
                    "tasks": w.study.tasks_for(p["kind"]),
                    "progress": w.study.progress(p["code"], p["kind"]) if w.study.items() else {},
                    "items_ready": w.study.items() is not None, "consent_text": CONSENT,
                    "reply_labels": list(REPLY_LABELS), "rule_labels": list(RULE_LABELS),
                    "live_steps": [{"step": st, "title": t, "text": x} for st, _, t, x in LIVE_STEPS],
                    "steps_done": sorted(steps_done(w.study.live_rows(), p["code"])),
                    "survey": [{"key": k, "text": t} for k, t in SURVEY]}

    @app.get("/api/study/me")
    def me(p: dict = Depends(participant)) -> dict:
        return session(p)

    @app.post("/api/study/consent")
    def consent(p: dict = Depends(participant)) -> dict:
        return session(W().study.consent(p["code"]))

    @app.get("/api/study/next/{task}")
    def next_item(task: str, p: dict = Depends(consented)) -> dict:
        w = W()
        if task not in w.study.tasks_for(p["kind"]):
            raise HTTPException(403, "this task is not part of your session")
        done = w.study.labels().get((task, p["code"]), {})
        queue = w.study.queue(p["code"], p["kind"], task)
        todo = [i for i in queue if i["id"] not in done]
        return {"item": _blind(task, todo[0]) if todo else None, "done": len(queue) - len(todo), "total": len(queue)}

    @app.post("/api/study/labels")
    def label(body: LabelBody, p: dict = Depends(consented)) -> dict:
        w = W()
        if body.task not in TASKS or body.task not in w.study.tasks_for(p["kind"]):
            raise HTTPException(403, "this task is not part of your session")
        if body.item not in {i["id"] for i in w.study.queue(p["code"], p["kind"], body.task)}:
            raise HTTPException(404, "no such item in your session")
        w.study.label(p["code"], body.task, body.item, _check(body.task, body.value))
        return {"ok": True}

    def pilot_in_portal(p: dict = Depends(consented), u: dict = Depends(user)) -> tuple[dict, dict]:
        if current_world.get() != DEMO:  # the same id can be someone else in another department
            raise HTTPException(409, "the study's live session runs in the demo department; switch to it first")
        if p["kind"] != "pilot" or u["id"] != p["persona"]:
            raise HTTPException(403, "sign in as your study persona first")
        return p, u

    @app.post("/api/study/practice")
    def practice(pu: tuple = Depends(pilot_in_portal)) -> dict:
        p, u = pu
        w = W()
        try:
            got = w.practice_clash(u["id"])
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        w.study.live(p["code"], u["id"], "practice", world=DEMO, **got)
        return got

    @app.post("/api/study/live")
    def live(body: LiveBody, pu: tuple = Depends(pilot_in_portal)) -> dict:
        p, u = pu
        if body.kind == "rating":
            body.value = _check("ratings", body.value if isinstance(body.value, dict) else {})
        if body.kind == "answer":
            if body.step not in ("timetable", "rules") or body.value not in ("yes", "partly", "no"):
                raise HTTPException(422, "an answer check needs step timetable or rules, and value yes, partly or no")
            chat = W().store.chat(u["id"], 0)  # the question it answered: the person's message before it
            body.text = body.text or next((m.get("text") for m in reversed(chat)
                                           if m["n"] < (body.n or 0) and m["kind"] == "text"), None)
        if body.kind == "survey":
            v = body.value if isinstance(body.value, dict) else {}
            got = {k: v.get(k) for k, _ in SURVEY}
            if not all(isinstance(x, int) and 1 <= x <= 5 for x in got.values()):
                raise HTTPException(422, "every closing statement needs a rating from 1 to 5")
            body.value = got
        W().study.live(p["code"], u["id"], body.kind, world=DEMO, **body.model_dump(exclude={"kind"}, exclude_none=True))
        return {"ok": True}

    # -- coordinator ----------------------------------------------------------------------

    @app.get("/api/study/admin")
    def admin(u: dict = Depends(coordinator)) -> dict:
        w = W()
        items = w.study.items()
        ps = w.study.participants()
        return {
            "items": {t: len(items.get(t, [])) for t in TASKS} | {"pilot_ratings": sum(i["pilot"] for i in items["ratings"]),
                                                          "built": items.get("built"), "source": items.get("source")}
            if items else None,
            "participants": [{"code": c, **v, "persona_name": w.name_of(v["persona"]) if v.get("persona") else None,
                              "progress": w.study.progress(c, v["kind"]) if items else {}}
                             for c, v in sorted(ps.items())],
            "analysis": analyse(w.study) if items else None,
            "reply_parser": w.models.get("replies"),
        }

    @app.post("/api/study/participants")
    def add(body: ParticipantsBody, u: dict = Depends(coordinator)) -> dict:
        w = W()
        persona = body.persona or DEFAULT_PERSONA
        if body.kind == "pilot" and persona not in w.instance.faculty_by_id:
            raise HTTPException(422, f"{persona} is not a faculty member of this department")
        return {"codes": w.study.add_participants(body.kind, body.n, persona)}

    @app.post("/api/study/participants/{code}/revoke")
    def revoke(code: str, body: RevokeBody, u: dict = Depends(coordinator)) -> dict:
        try:
            return W().study.revoke(code, delete_answers=body.delete_answers)
        except KeyError:
            raise HTTPException(404, "no such participant") from None

    @app.post("/api/study/participants/{code}/restore")
    def restore(code: str, u: dict = Depends(coordinator)) -> dict:
        try:
            return W().study.restore(code)
        except KeyError:
            raise HTTPException(404, "no such participant") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.post("/api/study/clear")
    def clear(body: ClearBody, u: dict = Depends(coordinator)) -> dict:
        if body.confirm.strip().lower() != CLEAR_PHRASE:
            raise HTTPException(422, f'type "{CLEAR_PHRASE}" to confirm')
        return {"removed": W().study.clear()}

    @app.get("/api/study/export")
    def export(u: dict = Depends(coordinator)) -> dict:
        w = W()
        return {"participants": w.study.participants(),
                "labels": [{"task": t, "code": c, "item": i, **v} for (t, c), vals in w.study.labels().items()
                           for i, v in vals.items()],
                "live": w.study.live_rows()}
