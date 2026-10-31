"""Models behind an OpenAI-style /v1/chat/completions endpoint with tool calls: llama.cpp's llama-server.

    model = OpenAIChat("http://10.1.25.191:8080/v1", adapter="nts-agent-v1")

llama-server needs ``--jinja`` to accept tools (it then renders them with the model's own chat
template, so the format the model is served in is the one it is trained in). With ``adapter``, the
server's LoRA adapters (``/lora-adapters``) are switched per request: that one at scale 1, the rest 0.
Arguments that are not valid JSON come back as one ``UNPARSED`` argument holding the text, which the
registry refuses as ``bad_json``, so the model sees its mistake instead of the episode failing.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from nts.protocol import UNPARSED, AssistantTurn, Message, ToolCall, Usage

_ids = itertools.count(1)


CONTEXT_FULL = re.compile(r"context size|exceeds? the available context|context length", re.IGNORECASE)


class ModelUnavailable(RuntimeError):
    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient  # busy or unreachable: the same request may work a little later

    @property
    def context_full(self) -> bool:
        return bool(CONTEXT_FULL.search(str(self)))


class OpenAIChat:
    def __init__(self, base_url: str, model: str | None = None, *, adapter: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 120.0, thinking: bool = False,
                 retries: int = 2, backoff: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.adapter = adapter
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking = thinking
        self.retries = retries
        self.backoff = backoff
        self._lora: list[dict] | None = None
        self.name = f"{Path(model).name if model else 'served'}{'+' + adapter if adapter else ''}"

    def _post(self, url: str, payload: dict | None = None) -> dict:
        """One request; a busy or unreachable server is tried again ``retries`` times, waiting longer each time.
        A full context is not (the same prompt would not fit again): the harness trims it and asks again."""
        for attempt in range(self.retries + 1):
            try:
                return self._post_once(url, payload)
            except ModelUnavailable as e:
                if attempt == self.retries or not e.transient:
                    raise
                time.sleep(self.backoff * 2 ** attempt)
        raise AssertionError("unreachable")

    def _post_once(self, url: str, payload: dict | None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode(errors="replace")
            hint = " (start llama-server with --jinja for tool calls)" if "tool" in body.lower() else ""
            busy = e.code in (429, 502, 503, 504) or (e.code == 500 and not CONTEXT_FULL.search(body))
            raise ModelUnavailable(f"{url}: HTTP {e.code}: {body}{hint}", transient=busy) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ModelUnavailable(f"{url} not reachable: {e}", transient=True) from e

    def _adapters(self) -> list[dict] | None:
        if self.adapter is None or self._lora is not None:
            return self._lora
        root = self.base_url.removesuffix("/v1")
        loaded = self._post(f"{root}/lora-adapters")
        want = self.adapter.removesuffix(".gguf")
        hits = [a["id"] for a in loaded if Path(a.get("path", "")).name.removesuffix(".gguf") == want]
        if not hits:
            names = [Path(a.get("path", "")).name for a in loaded]
            raise ModelUnavailable(f"adapter {self.adapter!r} is not loaded; the server has {names}")
        self._lora = [{"id": a["id"], "scale": 1.0 if a["id"] == hits[0] else 0.0} for a in loaded]
        return self._lora

    def chat(self, messages: Sequence[Message], tools: Sequence[dict]) -> AssistantTurn:
        payload: dict = {"messages": [to_openai(m) for m in messages], "temperature": self.temperature,
                         "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if tools:
            payload["tools"] = list(tools)
        if self.model:
            payload["model"] = self.model
        if (lora := self._adapters()) is not None:
            payload["lora"] = lora
        t0 = time.perf_counter()
        resp = self._post(f"{self.base_url}/chat/completions", payload)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelUnavailable(f"unexpected response: {str(resp)[:300]}") from e
        usage = resp.get("usage") or {}
        return AssistantTurn(text=msg.get("content") or "", tool_calls=[from_openai(c) for c in msg.get("tool_calls") or []],
                             raw=json.dumps(msg, ensure_ascii=False), model=resp.get("model") or self.name, ms=ms,
                             usage=Usage(prompt_tokens=usage.get("prompt_tokens", 0),
                                         completion_tokens=usage.get("completion_tokens", 0)))


def to_openai(m: Message) -> dict:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.call_id, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        return {"role": "assistant", "content": m.content or None,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                               for c in m.tool_calls]}
    return {"role": m.role, "content": m.content}


def from_openai(c: dict) -> ToolCall:
    fn = c.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        args = raw
    else:
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = {UNPARSED: raw}
        if not isinstance(args, dict):
            args = {UNPARSED: raw}
    return ToolCall(id=c.get("id") or f"call-{next(_ids)}", name=fn.get("name", ""), arguments=args)
