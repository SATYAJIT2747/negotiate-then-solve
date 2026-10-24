"""Gemini through google-genai, with function calling.

    model = GeminiChat()      # the keys in GEMINI_API_KEY (one or more) and GEMINI_MODEL from the environment

Gemini 3 models sign their function calls (``thought_signature``) and want the signed parts back on
later turns. The reply's parts are kept in ``AssistantTurn.meta["gemini_parts"]`` (they travel with
the transcript and the telemetry) and sent back as they came. With several keys, calls rotate across
them, each kept under its requests-per-minute limit (``GEMINI_RPM``, 15 on the free tier).
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nts.protocol import AssistantTurn, Message, ToolCall, Usage

DEFAULT_MODEL = "gemini-3.5-flash-lite"
RETRY_CODES = (429, 500, 503)
MAX_RATE_LIMITED = 5  # per key and call: per-minute limits hit this often mean something else is wrong
_ids = itertools.count(1)


@dataclass
class _Key:
    """One API key: its client, when it may be used next, and whether its daily quota is spent."""

    client: Any
    label: str  # "key 2 of 3": what errors and logs say instead of the key
    ready_at: float = 0.0
    spent: bool = False


def api_keys() -> list[str]:
    """The keys in GEMINI_API_KEYS / GEMINI_API_KEY (separated by commas, semicolons or white space) and in
    GEMINI_API_KEY_1, _2, ..., in that order, without repeats."""
    out: list[str] = []
    for name in ("GEMINI_API_KEYS", "GEMINI_API_KEY"):
        out += re.split(r"[,;\s]+", os.environ.get(name, ""))
    numbered = sorted((int(m.group(1)), v) for k, v in os.environ.items() if (m := re.fullmatch(r"GEMINI_API_KEY_(\d+)", k)))
    out += [v for _, v in numbered]
    return list(dict.fromkeys(k.strip() for k in out if k.strip()))


class GeminiChat:
    """Every call goes to the key that is ready soonest; each key is spaced to its own requests-per-minute
    limit, so n keys allow n times the rate. A per-minute rate limit rests that key for the delay the
    error asks for; a spent daily quota takes the key out until the process ends; the call moves on to
    another key either way. An overloaded server is retried after a growing wait."""

    def __init__(self, model: str | None = None, *, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048, retries: int = 4, rpm: float | None = None, client: Any = None,
                 clients: Sequence[Any] | None = None) -> None:
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.name = self.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        rpm = rpm or float(os.environ.get("GEMINI_RPM") or 15)
        self.min_interval = 60.0 / rpm
        given = list(clients or ([client] if client is not None else []))
        self._given, self._api_key = given, api_key
        self._keys: list[_Key] | None = None

    @property
    def keys(self) -> list[_Key]:
        if self._keys is None:
            clients = self._given
            if not clients:
                from google import genai

                keys = [self._api_key] if self._api_key else api_keys()
                if not keys:
                    raise RuntimeError("no GEMINI_API_KEY")
                clients = [genai.Client(api_key=k) for k in keys]
            self._keys = [_Key(c, f"key {i + 1} of {len(clients)}") for i, c in enumerate(clients)]
        return self._keys

    def _next_key(self) -> _Key:
        live = [k for k in self.keys if not k.spent]
        if not live:
            raise RuntimeError(f"{self.model}: the daily quota of all {len(self.keys)} key(s) is spent")
        key = min(live, key=lambda k: k.ready_at)
        if (wait := key.ready_at - time.monotonic()) > 0:
            time.sleep(wait)
        key.ready_at = time.monotonic() + self.min_interval
        return key

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> AssistantTurn:
        from google.genai import errors, types

        system, contents = to_gemini(messages)
        config = types.GenerateContentConfig(
            system_instruction=system or None, temperature=self.temperature, max_output_tokens=self.max_tokens,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["function"]["name"], description=t["function"]["description"],
                                          parameters_json_schema=t["function"]["parameters"]) for t in tools])]
            if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        t0 = time.perf_counter()
        overloaded = limited = 0
        while True:
            key = self._next_key()
            try:
                resp = key.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except errors.APIError as e:
                if e.code == 429 and per_day(e):
                    key.spent = True
                    continue
                if e.code == 429 and limited < MAX_RATE_LIMITED * len(self.keys):
                    limited += 1
                    key.ready_at = time.monotonic() + (retry_delay(e) or 60.0)
                    continue
                if e.code not in RETRY_CODES or overloaded == self.retries:
                    raise
                overloaded += 1
                time.sleep(min(60, 2 ** overloaded))
        return from_gemini(resp, self.model, round((time.perf_counter() - t0) * 1000, 1))


def per_day(e: Exception) -> bool:
    """Whether a rate-limit error is the daily quota (not the per-minute one)."""
    return bool(re.search(r"PerDay|per day|daily", str(e), re.IGNORECASE))


def retry_delay(e: Exception) -> float | None:
    """The wait a rate-limit error asks for ("Please retry in 26.38s", or a RetryInfo retryDelay "26s"), plus a second."""
    m = re.search(r"retry in ([\d.]+)s", str(e), re.IGNORECASE) or re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", str(e))
    return float(m.group(1)) + 1 if m else None


def to_gemini(messages: Sequence[Message]) -> tuple[str, list]:
    """(system instruction, contents); consecutive messages of one role go in one content, so the results of
    calls made in one turn answer together."""
    from google.genai import types

    system: list[str] = []
    contents: list[types.Content] = []

    def put(role: str, parts: list) -> None:
        if contents and contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=list(parts)))

    for i, m in enumerate(messages):
        if m.role == "system" and i == 0:
            system.append(m.content)
        elif m.role in ("system", "user"):
            put("user", [types.Part(text=m.content)])
        elif m.role == "assistant":
            if m.meta and m.meta.get("gemini_parts"):
                put("model", [types.Part.model_validate_json(p) for p in m.meta["gemini_parts"]])
            else:
                put("model", ([types.Part(text=m.content)] if m.content else [])
                    + [types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
                       for c in m.tool_calls])
        else:
            try:
                body = json.loads(m.content)
            except json.JSONDecodeError:
                body = m.content
            put("user", [types.Part(function_response=types.FunctionResponse(
                id=m.call_id, name=m.name, response=body if isinstance(body, dict) and "error" in body else {"output": body}))])
    return "\n\n".join(system), contents


def from_gemini(resp: Any, model: str, ms: float) -> AssistantTurn:
    cand = (resp.candidates or [None])[0]
    parts = list(cand.content.parts or []) if cand is not None and cand.content is not None else []
    if not parts:
        reason = getattr(cand, "finish_reason", None)
        raise RuntimeError(f"{model} sent an empty reply (finish_reason={reason})")
    text = "".join(p.text for p in parts if p.text and not p.thought)
    calls = [ToolCall(id=p.function_call.id or f"gcall-{next(_ids)}", name=p.function_call.name or "",
                      arguments=dict(p.function_call.args or {})) for p in parts if p.function_call]
    if any(p.function_call and not p.function_call.id for p in parts):  # keep our ids in the parts sent back
        for p, c in zip([p for p in parts if p.function_call], calls, strict=True):
            p.function_call.id = c.id
    um = resp.usage_metadata
    usage = Usage(prompt_tokens=(um.prompt_token_count or 0) if um else 0,
                  completion_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0)
    return AssistantTurn(text=text, tool_calls=calls, model=model, ms=ms, usage=usage,
                         raw=json.dumps([p.model_dump(mode="json", exclude_none=True) for p in parts], ensure_ascii=False),
                         meta={"gemini_parts": [p.model_dump_json(exclude_none=True) for p in parts]})
