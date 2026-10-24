"""Gemini through google-genai, with function calling.

    model = GeminiChat()                      # GEMINI_API_KEY and GEMINI_MODEL from the environment

Gemini 3 models sign their function calls (``thought_signature``) and want the signed parts back on
later turns. The reply's parts are kept in ``AssistantTurn.meta["gemini_parts"]`` (they travel with
the transcript and the telemetry) and sent back as they came. A rate limit or an overloaded server is
retried a few times with a growing wait.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Sequence
from typing import Any

from nts.protocol import AssistantTurn, Message, ToolCall, Usage

DEFAULT_MODEL = "gemini-3.5-flash-lite"
RETRY_CODES = (429, 500, 503)
_ids = itertools.count(1)


class GeminiChat:
    def __init__(self, model: str | None = None, *, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048, retries: int = 4, client: Any = None) -> None:
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.name = self.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = client
        self._key = api_key

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            key = self._key or os.environ.get("GEMINI_API_KEY", "").split(",")[0].strip()
            if not key:
                raise RuntimeError("no GEMINI_API_KEY")
            self._client = genai.Client(api_key=key)
        return self._client

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
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except errors.APIError as e:
                if e.code not in RETRY_CODES or attempt == self.retries:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))
        return from_gemini(resp, self.model, round((time.perf_counter() - t0) * 1000, 1))


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
