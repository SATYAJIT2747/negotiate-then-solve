"""Requests in other words: the scenario templates rewritten the way people write, keeping every fact.

    suite = rewrite(build_suite(world, seed=0), "heldout", GeminiChat(temperature=1.0))

The templates are few, so a model trained on them may learn the templates as much as the task. Each
request is rewritten by a model in one of a style set's styles (picked by a hash of the request, so a
request always gets the same style), and kept only if every fact survives: the days (three letters
are enough: "Tue"), the hours, the weeks ("week 6", "next week"), the names of courses, teachers and
rooms, and "if possible" when the request was a wish (and never when it was not). A rewrite that loses one falls back to the
template. The two style sets are disjoint: "train" for collecting training episodes, "heldout" for
evaluating on wording the model never trained on.

Rewrites are cached by style and original request (``runs/eval/rewrites-<style>.json``), so a suite
rewrites the same way on every run, and the model is called only for requests it has not seen.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

from nts.models import Model
from nts.protocol import Message
from nts.state import World
from nts.tools.names import FULL_DAY

from .scenarios import Scenario

STYLES = {
    "train": [
        "a quick chat message: lower case, short forms like 'wk', 'tmrw', 'Tues', 'pls'",
        "a polite email: a greeting, the request, a thank-you and a sign-off",
        "terse, as few words as possible, like a text message",
        "chatty, with a short aside about something unrelated (students, the weather, a meeting)",
        "hurried, with one or two small typos (never in a name, day, hour or number)",
    ],
    "heldout": [
        "written by a non-native English speaker: slightly awkward grammar and word order",
        "very indirect and apologetic, the request comes only at the end",
        "as a short bulleted list",
        "with day abbreviations (Mon, Wed, Thu) and hours as 'o'clock' where natural",
        "starting with why (the students, a clash, a trip) before saying what is wanted",
    ],
}
SOFT = re.compile(r"if (at all )?possible|prefer|rather|would be nice|ideally|if it can|not a must|if you can", re.IGNORECASE)
BATCH = 12

PROMPT = """You rewrite requests sent to a university timetable office so they read like real people wrote them.
Each request gets a style. Keep every fact exactly: the same days, the same hours (as numbers, e.g. "2 pm",
"2pm" or "2 o'clock"), the same weeks ("week 6", "this week", "next week"), the same names of courses,
teachers and rooms, "only that week" or "from now on" if said, and "if possible" (or "I'd prefer") if the
request is a wish rather than a must; never make a must sound optional ("if possible"). Do not add facts, do not answer the request.
Return only a JSON list of {"id": ..., "text": ...}, one per request, in the same order."""


def style_of(request: str, styles: list[str]) -> str:
    return styles[int(hashlib.sha256(request.encode()).hexdigest(), 16) % len(styles)]


def facts(request: str, world: World) -> list[str]:
    """Patterns (regular expressions) a rewrite must match: one per fact the request states."""
    inst = world.instance
    text = request.lower()
    out = []
    for day in inst.calendar.days:
        if re.search(rf"\b{FULL_DAY[day].lower()}", text):
            out.append(rf"\b{day.lower()}")
    for h, ap in re.findall(r"\b(1[0-2]|[1-9])(?::00)?\s*([ap])\.?m\b", text):
        out.append(rf"\b{h}(?!\d)" +("|noon|midday" if h == "12" and ap == "p" else ""))
    for w in re.findall(r"\bweek (\d+)", text):
        out.append(rf"\b(week|wk)\.? ?{w}\b")
    for rel in ("next week", "this week"):
        if rel in text:
            out.append(rf"\b{rel.split()[0]}\b")
    names = [*inst.course_titles.values(), *(f.name for f in inst.faculty), *(r.name for r in inst.rooms)]
    for n in names:
        if n.lower() in text:
            out.append(re.escape(n.lower()).replace(r"dr\.\ ", r"(dr\.? )?"))
    if SOFT.search(request):
        out.append(SOFT.pattern)
    return out


def keeps(rewritten: str, patterns: list[str]) -> list[str]:
    """The patterns a rewrite does not match (empty: it kept every fact)."""
    return [p for p in patterns if not re.search(p, rewritten, re.IGNORECASE)]


def rewrite(suite: list[Scenario], style: str, model: Model, world: World, *, cache: Path | None = None,
            log: bool = False, strict: bool = False) -> list[Scenario]:
    """``strict``: raise if a batch could not be rewritten (the model failed), instead of keeping its templates."""
    styles = STYLES[style]
    cache = cache or Path(f"runs/eval/rewrites-{style}.json")
    known: dict[str, str] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = list(dict.fromkeys(s.request for s in suite if s.request not in known))
    failed = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        items = [{"id": k, "style": style_of(r, styles), "request": r} for k, r in enumerate(batch)]
        try:
            reply = model.chat([Message(role="system", content=PROMPT),
                                Message(role="user", content=json.dumps(items, ensure_ascii=False))], [])
            got = {int(x["id"]): str(x["text"]) for x in json.loads(_json(reply.text))}
        except Exception as e:  # noqa: BLE001 -- a failed batch keeps its templates, and is tried again next time
            if log:
                print(f"rewrite: batch {i // BATCH + 1} failed: {type(e).__name__}: {e}")
            failed += len(batch)
            continue
        for k, original in enumerate(batch):
            new = got.get(k, "").strip()
            lost = keeps(new, facts(original, world)) if new else ["no text"]
            if new and SOFT.search(new) and not SOFT.search(original):
                lost.append("a must that now reads as a wish")
            known[original] = original if lost else new  # a rewrite that lost a fact: the template stays
            if log and lost:
                print(f"rewrite kept the template ({', '.join(lost)}): {original}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    if strict and failed:
        raise RuntimeError(f"{failed} of {len(todo)} requests could not be rewritten (the rewrites so far are cached)")
    return [dataclasses.replace(s, request=known.get(s.request, s.request)) for s in suite]


def _json(text: str) -> str:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m is None:
        raise ValueError("no JSON list in the reply")
    return m.group(0)
