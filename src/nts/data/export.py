"""Training rows from labelled episodes, in the chat format with tools that Qwen's template renders.

    python -m nts export --out data/agent-v1 --models gemini

SFT rows are whole good episodes: ``{"messages": [...], "tools": [...]}``, ready for
``tokenizer.apply_chat_template(messages, tools=tools)``, trained on every assistant turn. By default a
turn whose calls were all refused for a fixable reason (a bad name, bad arguments, a repeat) is left
out with its results, so the model learns the clean path; ``keep_errors`` keeps them, teaching recovery.

DPO pairs come from those refusals: the prompt is the conversation before the refused turn, "rejected"
the refused turn, "chosen" the turn that fixed it (same tool, accepted). The fix was written after
seeing the error, so the pair is an approximation of what the model should have done first.

Rows are split by a hash of the request text, so one request's rows never sit on both sides.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from nts.protocol import Message

from .episodes import Episode

FIXABLE = {"unknown_name", "ambiguous_name", "bad_arguments", "bad_json", "unknown_tool", "repeated_call",
           "unknown_day", "unknown_hour", "unknown_week", "too_many_calls", "not_checked"}


def chat(m: Message) -> dict:
    if m.role == "assistant":
        out: dict = {"role": "assistant", "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in m.tool_calls]
        return out
    if m.role == "tool":
        return {"role": "tool", "name": m.name, "content": m.content}
    return {"role": m.role, "content": m.content}


def _refused(m: Message) -> str | None:
    """The error code of a tool message, or None if it succeeded."""
    try:
        body = json.loads(m.content)
    except json.JSONDecodeError:
        return None
    return body["error"].get("code") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None


def conversation(ep: Episode) -> list[Message]:
    """The whole episode as the model last saw it, plus its last reply."""
    turns = ep.turns()
    if not turns:
        return []
    sent, reply = turns[-1]
    return [*sent, reply.as_message()]


def without_errors(messages: list[Message]) -> list[Message]:
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            results = messages[i + 1: i + 1 + len(m.tool_calls)]
            codes = [_refused(r) for r in results]
            if results and all(c in FIXABLE for c in codes):
                i += 1 + len(results)
                continue
        out.append(m)
        i += 1
    return out


def sft_row(ep: Episode, *, keep_errors: bool = False) -> dict | None:
    msgs = conversation(ep)
    if not msgs:
        return None
    if not keep_errors:
        msgs = without_errors(msgs)
    return {"messages": [chat(m) for m in msgs], "tools": ep.tools,
            "meta": {"episode": ep.id, "model": ep.model, **{k: v for k, v in ep.label().items() if k != "model"}}}


def dpo_pairs(ep: Episode) -> list[dict]:
    msgs = conversation(ep)
    pairs = []
    for i, m in enumerate(msgs):
        if m.role != "assistant" or not m.tool_calls:
            continue
        results = msgs[i + 1: i + 1 + len(m.tool_calls)]
        if not results or not all(_refused(r) in FIXABLE for r in results):
            continue
        j = i + 1 + len(results)
        fix = msgs[j] if j < len(msgs) else None
        if fix is None or fix.role != "assistant" or not fix.tool_calls:
            continue
        fixed = msgs[j + 1: j + 1 + len(fix.tool_calls)]
        if {c.name for c in fix.tool_calls} & {c.name for c in m.tool_calls} and all(_refused(r) is None for r in fixed):
            pairs.append({"prompt": [chat(x) for x in msgs[:i]], "rejected": chat(m), "chosen": chat(fix),
                          "tools": ep.tools, "meta": {"episode": ep.id, "error": _refused(results[0])}})
    return pairs


def split_of(ep: Episode, val_share: int = 10) -> str:
    request = ep.episode.attrs.get("request", "")
    return "val" if int(hashlib.sha256(request.encode()).hexdigest(), 16) % val_share == 0 else "train"


def export(eps: Iterable[Episode], out: Path, *, models: list[str] | None = None, include_unjudged: bool = False,
           keep_errors: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    files = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("sft-train", "sft-val", "dpo")}
    counts = {"episodes": 0, "good": 0, "bad": 0, "unjudged": 0, "skipped_model": 0,
              "sft-train": 0, "sft-val": 0, "dpo": 0}
    seen: set[str] = set()
    try:
        for ep in eps:
            counts["episodes"] += 1
            if models and not any(m in ep.model for m in models):
                counts["skipped_model"] += 1
                continue
            good = ep.label()["good"]
            counts["good" if good else "unjudged" if good is None else "bad"] += 1
            for pair in dpo_pairs(ep):
                files["dpo"].write(json.dumps(pair, ensure_ascii=False) + "\n")
                counts["dpo"] += 1
            if not (good or (good is None and include_unjudged)):
                continue
            row = sft_row(ep, keep_errors=keep_errors)
            if row is None:
                continue
            key = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            name = f"sft-{split_of(ep)}"
            files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[name] += 1
    finally:
        for f in files.values():
            f.close()
    (out / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return counts
