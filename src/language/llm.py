"""Gemini API client with multi-key load balancing and pooling (proposal Section 14).

* Structured output: responses are parsed into a Pydantic model.
* Disk cache keyed by a hash of (model, prompts, schema, temperature), so
  reruns of an experiment never repeat a call.
* API Key Pooling: supports multiple API keys via ``GEMINI_API_KEYS`` or
  ``GEMINI_API_KEY`` (comma-delimited) or numbered variables ``GEMINI_API_KEY_1``,
  ``GEMINI_API_KEY_2``, etc. Requests are balanced round-robin across ready keys,
  multiplying effective throughput and daily budget.
* Per-key rate limiting, pacing, and daily quota tracking, with automatic
  failover when an individual key reaches its daily budget or rate limit.

The keys come from ``GEMINI_API_KEYS`` / ``GEMINI_API_KEY`` (environment or a ``.env`` file).
Model ids come from ``GEMINI_MODEL`` / ``GEMINI_LITE_MODEL`` so they can
follow Google's current Flash releases without code changes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypeVar

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Pinned model ids (not "-latest" aliases, which can be repointed or
# overloaded) so experiment runs are reproducible. Each model has its own
# free-tier quota, so the roles use different models:
#   agent     - System Two parser and negotiator
#   simulator - stakeholder simulators and the LLM judge (bulk calls)
#   flash     - larger-model comparison set (only 20 requests/day)
MODELS = {
    "agent": "gemini-3.5-flash-lite",
    "simulator": "gemini-3.1-flash-lite",
    "flash": "gemini-3.8-flash",
}
_ENV = {"agent": "GEMINI_MODEL", "simulator": "GEMINI_SIM_MODEL", "flash": "GEMINI_FLASH_MODEL"}

# Free-tier (RPM, RPD) per model, from the AI Studio rate-limit page (Sep 2026).
# GEMINI_RPM / GEMINI_RPD override these.
FREE_TIER = {
    "gemini-3.5-flash-lite": (15, 500),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-3.8-flash": (5, 20),
    "gemini-3.6-flash": (5, 20),
    "gemini-3.5-flash": (5, 20),
    "gemini-2.5-flash": (5, 20),
}
_FALLBACK_LIMITS = (5, 20)  # unknown model: assume the tightest tier


def default_thinking(model: str) -> int | str:
    """Flash accepts thinking_budget=0; Flash-Lite only accepts a thinking level."""
    return "low" if "lite" in model else 0


def default_model(role: str = "agent") -> str:
    load_dotenv(find_dotenv(usecwd=True))
    return os.environ.get(_ENV[role], MODELS[role])


def mask_key(key: str) -> str:
    """Mask an API key for safe logging, showing only the tail."""
    if not key:
        return "<empty>"
    clean = key.strip()
    if len(clean) <= 8:
        return f"...{clean[-4:]}" if len(clean) >= 4 else "..."
    return f"...{clean[-6:]}"


def resolve_api_keys(
    api_key: str | None = None,
    api_keys: list[str] | str | None = None,
) -> list[str]:
    """Extract and deduplicate API keys from explicit parameters and environment variables.

    Sources checked:
    1. ``api_keys`` argument (list of strings or delimited string)
    2. ``api_key`` argument (single string or delimited string)
    3. ``GEMINI_API_KEYS`` environment variable (comma, semicolon, or newline separated)
    4. ``GEMINI_API_KEY`` environment variable (single or delimited)
    5. Numbered environment variables: ``GEMINI_API_KEY_1``, ``GEMINI_API_KEY_2``, ...
    """
    raw_keys: list[str] = []

    def _split_and_add(val: str | None) -> None:
        if not val:
            return
        parts = re.split(r"[,;\n\r]+", val)
        for p in parts:
            cleaned = p.strip().strip("'\"")
            if cleaned:
                raw_keys.append(cleaned)

    explicit_provided = (api_keys is not None) or (api_key is not None)

    # 1. Explicit api_keys argument
    if api_keys is not None:
        if isinstance(api_keys, str):
            _split_and_add(api_keys)
        elif isinstance(api_keys, (list, tuple, set)):
            for k in api_keys:
                if isinstance(k, str):
                    _split_and_add(k)

    # 2. Explicit api_key argument
    if api_key is not None:
        _split_and_add(api_key)

    # 3. Environment variables (only if no explicit argument was provided)
    if not explicit_provided:
        _split_and_add(os.environ.get("GEMINI_API_KEYS"))
        _split_and_add(os.environ.get("GEMINI_API_KEY"))

        numbered: list[tuple[int, str]] = []
        for env_k, env_v in os.environ.items():
            m = re.match(r"^GEMINI_API_KEY_(\d+)$", env_k, re.IGNORECASE)
            if m and env_v.strip():
                numbered.append((int(m.group(1)), env_v.strip()))
        for _, val in sorted(numbered, key=lambda x: x[0]):
            _split_and_add(val)

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for k in raw_keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    return deduped


@dataclass
class KeyStats:
    key_id: str
    calls: int = 0
    retries: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    daily_used: int = 0
    exhausted_today: bool = False
    disabled: bool = False


@dataclass
class PooledKey:
    index: int
    key: str
    client: Any
    key_id: str
    key_hash: str
    calls: int = 0
    retries: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    daily_used: int = 0
    exhausted_today: bool = False
    last_call_time: float = 0.0
    disabled: bool = False
    disable_reason: str = ""


class ApiKeyPool:
    """Thread-safe pool of Gemini API keys with round-robin load balancing,
    per-key pacing, per-key daily quota tracking, and automatic failover."""

    def __init__(
        self,
        keys: list[str],
        model: str,
        cache_dir: Path,
        rpm: float,
        rpd: int,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not keys:
            raise LLMError("ApiKeyPool requires at least one API key")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rpm = rpm
        self.rpd = rpd
        self.min_interval = 60.0 / (rpm * 0.9)  # 10% headroom
        self._lock = threading.Lock()
        self._round_robin_idx = 0

        if client_factory is None:
            from google import genai
            client_factory = lambda k: genai.Client(api_key=k)
        self._client_factory = client_factory

        self.keys: list[PooledKey] = []
        for i, k in enumerate(keys, start=1):
            khash = hashlib.sha256(k.encode("utf-8")).hexdigest()[:8]
            kid = f"key-{i} ({mask_key(k)})"
            self.keys.append(
                PooledKey(
                    index=i,
                    key=k,
                    client=self._client_factory(k),
                    key_id=kid,
                    key_hash=khash,
                )
            )

    @property
    def primary_client(self) -> Any:
        return self.keys[0].client

    @property
    def effective_min_interval(self) -> float:
        active = [k for k in self.keys if not k.exhausted_today and not k.disabled]
        n = len(active) or 1
        return self.min_interval / n

    @property
    def effective_rpd(self) -> int:
        active = [k for k in self.keys if not k.disabled]
        return self.rpd * len(active)

    def has_available_keys(self) -> bool:
        today_str = date.today().isoformat()
        with self._lock:
            for k in self.keys:
                if k.disabled or k.exhausted_today:
                    continue
                qpath = self.cache_dir / f"_quota-{self.model}-{today_str}-{k.key_hash}.txt"
                if qpath.exists():
                    with contextlib.suppress(OSError, ValueError):
                        used = int(qpath.read_text(encoding="utf-8"))
                        if used >= self.rpd:
                            k.exhausted_today = True
                            k.daily_used = used
                            continue
                return True
        return False

    def select_and_reserve(self) -> tuple[PooledKey, float]:
        """Atomically select the best available key and reserve its time slot.

        Returns (selected_key, wait_seconds).
        The caller MUST sleep for `wait_seconds` (if > 0) outside the lock,
        then invoke the API.
        """
        today_str = date.today().isoformat()
        now = time.time()
        with self._lock:
            candidates: list[PooledKey] = []
            for k in self.keys:
                if k.disabled:
                    continue
                if not k.exhausted_today:
                    qpath = self.cache_dir / f"_quota-{self.model}-{today_str}-{k.key_hash}.txt"
                    if qpath.exists():
                        with contextlib.suppress(OSError, ValueError):
                            used = int(qpath.read_text(encoding="utf-8"))
                            if used >= self.rpd:
                                k.exhausted_today = True
                                k.daily_used = used
                                continue
                            k.daily_used = used
                    candidates.append(k)

            if not candidates:
                raise DailyQuotaReached(
                    f"{self.model}: all {len(self.keys)} API key(s) reached their daily budget of {self.rpd} (GEMINI_RPD)"
                )

            best_key: PooledKey | None = None
            best_ready_time = float("inf")
            ready_candidates: list[PooledKey] = []

            for k in candidates:
                stamp_path = self.cache_dir / f"_last-{self.model}-{k.key_hash}.txt"
                disk_last = 0.0
                if stamp_path.exists():
                    with contextlib.suppress(OSError, ValueError):
                        disk_last = float(stamp_path.read_text(encoding="utf-8"))
                last_call = max(k.last_call_time, disk_last)
                ready_at = last_call + self.min_interval
                if ready_at <= now:
                    ready_candidates.append(k)
                if ready_at < best_ready_time:
                    best_ready_time = ready_at
                    best_key = k

            if ready_candidates:
                # Round-robin among ready keys, breaking ties with least calls
                selected = min(
                    ready_candidates,
                    key=lambda k: (k.calls, (k.index - 1 - self._round_robin_idx) % len(self.keys)),
                )
                wait_seconds = 0.0
                scheduled_time = now
            else:
                assert best_key is not None
                selected = best_key
                wait_seconds = max(0.0, best_ready_time - now)
                scheduled_time = best_ready_time

            self._round_robin_idx = (selected.index) % len(self.keys)
            selected.last_call_time = scheduled_time

            # Record reservations on disk for cross-process coordination
            with contextlib.suppress(OSError, ValueError):
                stamp_path = self.cache_dir / f"_last-{self.model}-{selected.key_hash}.txt"
                stamp_path.write_text(repr(scheduled_time), encoding="utf-8")

                qpath = self.cache_dir / f"_quota-{self.model}-{today_str}-{selected.key_hash}.txt"
                used = int(qpath.read_text(encoding="utf-8")) if qpath.exists() else 0
                used += 1
                qpath.write_text(str(used), encoding="utf-8")
                selected.daily_used = used

            return selected, wait_seconds

    def mark_exhausted(self, key: PooledKey, reason: str = "") -> None:
        today_str = date.today().isoformat()
        with self._lock:
            key.exhausted_today = True
            key.disable_reason = reason
            with contextlib.suppress(OSError, ValueError):
                qpath = self.cache_dir / f"_quota-{self.model}-{today_str}-{key.key_hash}.txt"
                qpath.write_text(str(self.rpd), encoding="utf-8")
                key.daily_used = self.rpd

    def mark_rate_limited(self, key: PooledKey, penalty_seconds: float = 5.0) -> None:
        now = time.time()
        with self._lock:
            key.last_call_time = max(key.last_call_time, now) + penalty_seconds
            with contextlib.suppress(OSError, ValueError):
                stamp_path = self.cache_dir / f"_last-{self.model}-{key.key_hash}.txt"
                stamp_path.write_text(repr(key.last_call_time), encoding="utf-8")

    def mark_disabled(self, key: PooledKey, reason: str = "") -> None:
        with self._lock:
            key.disabled = True
            key.disable_reason = reason

    def record_success(self, key: PooledKey, prompt_tokens: int, output_tokens: int) -> None:
        with self._lock:
            key.calls += 1
            key.prompt_tokens += prompt_tokens
            key.output_tokens += output_tokens

    def get_status(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "index": k.index,
                    "key_id": k.key_id,
                    "calls": k.calls,
                    "retries": k.retries,
                    "errors": k.errors,
                    "prompt_tokens": k.prompt_tokens,
                    "output_tokens": k.output_tokens,
                    "daily_used": k.daily_used,
                    "exhausted_today": k.exhausted_today,
                    "disabled": k.disabled,
                    "disable_reason": k.disable_reason,
                }
                for k in self.keys
            ]


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    last_retry_reason: str = ""
    key_distribution: dict[str, int] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class InvalidOutput(LLMError):
    """The model answered, but not with an instance of the schema (cut off at
    the token limit, say). Callers may treat it as a failed call and retry, or
    repair ``text`` (the answer as given) when it is a near miss."""

    def __init__(self, message: str, text: str | None = None) -> None:
        super().__init__(message)
        self.text = text


class DailyQuotaReached(LLMError):
    """This run would exceed the per-day request budget. Cached responses
    mean a rerun tomorrow resumes where this one stopped."""


class GeminiClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        cache_dir: Path | str = "runs/llm_cache",
        rpm: float | None = None,
        rpd: int | None = None,
        max_retries: int = 6,
        thinking: int | str | None = None,
        api_key: str | None = None,
        api_keys: list[str] | str | None = None,
        _client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        load_dotenv(find_dotenv(usecwd=True))
        keys = resolve_api_keys(api_key=api_key, api_keys=api_keys)
        if not keys:
            raise LLMError("GEMINI_API_KEY is not set (environment or .env)")

        self.model = model or default_model()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tier_rpm, tier_rpd = FREE_TIER.get(self.model, _FALLBACK_LIMITS)
        self.rpm = rpm or float(os.environ.get("GEMINI_RPM", tier_rpm))
        self.rpd = rpd or int(os.environ.get("GEMINI_RPD", tier_rpd))
        self.min_interval = 60.0 / (self.rpm * 0.9)  # 10% headroom per key
        self.max_retries = max_retries
        self.thinking = thinking if thinking is not None else default_thinking(self.model)
        self.usage = Usage()
        self.latencies: list[float] = []

        self.pool = ApiKeyPool(
            keys=keys,
            model=self.model,
            cache_dir=self.cache_dir,
            rpm=self.rpm,
            rpd=self.rpd,
            client_factory=_client_factory,
        )
        self._lock = threading.Lock()

    @property
    def _client(self) -> Any:
        """Backward compatibility property returning the primary client."""
        return self.pool.primary_client

    @property
    def keys(self) -> list[str]:
        return [k.key for k in self.pool.keys]

    @property
    def effective_min_interval(self) -> float:
        return self.pool.effective_min_interval

    @property
    def effective_rpd(self) -> int:
        return self.pool.effective_rpd

    def pool_status(self) -> list[dict]:
        """Return real-time usage and health status for each key in the pool."""
        return self.pool.get_status()

    def generate(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        key = hashlib.sha256(
            json.dumps([self.model, str(self.thinking), system, prompt, schema_json, temperature]).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if use_cache and path.exists():
            with self._lock:
                self.usage.cache_hits += 1
            return schema.model_validate_json(json.loads(path.read_text(encoding="utf-8"))["text"])

        text, usage = self._call(system, prompt, schema, temperature)
        try:
            result = schema.model_validate_json(text)
        except ValidationError as e:
            raise InvalidOutput(f"invalid {schema.__name__} ({len(text)} chars)", text) from e
        path.write_text(
            json.dumps({"model": self.model, "text": text, "usage": usage}, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    def _call(self, system: str, prompt: str, schema: type[BaseModel], temperature: float) -> tuple[str, dict]:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if isinstance(self.thinking, int):
            config.thinking_config = types.ThinkingConfig(thinking_budget=self.thinking)
        elif isinstance(self.thinking, str):
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking)

        delay = 5.0
        for attempt in range(self.max_retries + 1):
            pooled_key, wait_seconds = self.pool.select_and_reserve()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            started = time.monotonic()
            try:
                resp = pooled_key.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                latency = time.monotonic() - started
            except errors.APIError as e:
                details_str = json.dumps(e.details or {}) if getattr(e, "details", None) is not None else ""
                err_text = f"{e.message or ''} {e!s} {details_str}".lower()

                is_daily = e.code == 429 and ("perday" in err_text or "daily" in err_text)
                if is_daily:
                    self.pool.mark_exhausted(pooled_key, reason=str(e.message or str(e)))
                    if self.pool.has_available_keys():
                        continue
                    raise DailyQuotaReached(
                        f"{self.model}: all {len(self.pool.keys)} API key(s) exhausted daily quota"
                    ) from e

                # 403 is always a key/project problem (e.g. "project has been denied access")
                is_bad_key = e.code == 403 or (e.code == 400 and any(
                    phrase in err_text
                    for phrase in ["api key not valid", "api_key_invalid", "unregistered", "forbidden", "bad request"]
                ))
                if is_bad_key:
                    self.pool.mark_disabled(pooled_key, reason=str(e.message or str(e)))
                    if self.pool.has_available_keys():
                        continue
                    err_msg = e.message or str(e)
                    raise LLMError(f"All available API keys failed. Last error on {pooled_key.key_id}: {err_msg}") from e

                retryable = e.code == 429 or (e.code is not None and e.code >= 500)
                if not retryable or attempt == self.max_retries:
                    with self._lock:
                        pooled_key.errors += 1
                    err_msg = e.message or str(e)
                    raise LLMError(f"Gemini call failed ({e.code}) on {pooled_key.key_id}: {err_msg}") from e

                with self._lock:
                    self.usage.retries += 1
                    self.usage.last_retry_reason = f"{e.code} on {pooled_key.key_id}: {str(e.message or str(e))[:300]}"
                    pooled_key.retries += 1
                    if e.code == 429:
                        self.pool.mark_rate_limited(pooled_key, penalty_seconds=delay)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue

            meta = resp.usage_metadata
            usage = {
                "prompt_tokens": (meta.prompt_token_count or 0) if meta else 0,
                "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
                "key_id": pooled_key.key_id,
            }
            self.pool.record_success(pooled_key, usage["prompt_tokens"], usage["output_tokens"])
            with self._lock:
                self.usage.calls += 1
                self.usage.prompt_tokens += usage["prompt_tokens"]
                self.usage.output_tokens += usage["output_tokens"]
                self.usage.key_distribution[pooled_key.key_id] = (
                    self.usage.key_distribution.get(pooled_key.key_id, 0) + 1
                )
                self.latencies.append(latency)

            if not resp.text:
                raise LLMError(f"empty response (finish reason: {resp.candidates and resp.candidates[0].finish_reason})")
            return resp.text, usage

        raise AssertionError("unreachable")

    def _pace(self) -> None:
        """Legacy helper maintained for backward compatibility."""

    def _count_today(self) -> None:
        """Legacy helper maintained for backward compatibility."""
