"""The production ModelClient, targeting deepseek-v4-flash.

The provider offers a JSON-object output mode but no strict schema
enforcement (ADR-0007), so this module owns validation and retry: every
response is checked against the requested Pydantic model, retried at most
twice, and then failed loudly by raising ModelClientError, which
screening.core turns into an Unresolved Candidate. No repair call is ever
made - a retry repeats the exact same prompt rather than feeding the bad
output back for correction.

The HTTP call itself sits behind CompletionTransport, an injected boundary,
so the retry-and-validation layer here is testable without network access.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from screening.model_client import ModelClientError, RunMetrics, T

MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
MAX_ATTEMPTS = 3  # one attempt plus two retries
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"


@dataclass(frozen=True)
class RawCompletion:
    """What the transport boundary hands back: the provider's raw message
    content - None if it returned an empty response - plus the cache token
    counts it reported for the call.
    """

    content: str | None
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0


class CompletionTransport(Protocol):
    def __call__(self, prompt: str) -> RawCompletion: ...


def build_live_transport(
    api_key: str, *, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0
) -> CompletionTransport:
    """The real HTTP transport, against DeepSeek's OpenAI-compatible
    chat-completions endpoint with JSON-object output mode.
    """

    def transport(prompt: str) -> RawCompletion:
        body = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())

        content = payload["choices"][0]["message"].get("content")
        usage = payload.get("usage", {})
        return RawCompletion(
            content=content if content else None,
            cache_hit_tokens=usage.get("prompt_cache_hit_tokens") or 0,
            cache_miss_tokens=usage.get("prompt_cache_miss_tokens") or 0,
        )

    return transport


class DeepSeekModelClient:
    """A ModelClient backed by deepseek-v4-flash. `metrics` accumulates
    parse-failure and cache-token counters across every call made through
    this instance, for the caller to fold into the run record.
    """

    def __init__(self, transport: CompletionTransport, *, max_attempts: int = MAX_ATTEMPTS) -> None:
        self._transport = transport
        self._max_attempts = max_attempts
        self.metrics = RunMetrics()

    def complete(self, prompt: str, response_model: type[T]) -> T:
        last_error: ModelClientError | None = None

        for _ in range(self._max_attempts):
            self.metrics.total_attempts += 1
            try:
                raw = self._transport(prompt)
            except Exception as exc:
                # A network failure or an unexpected provider envelope shape
                # (e.g. a missing "choices" key) is not a parse failure - it
                # never produced content to parse - but per ADR-0001/ADR-0007
                # it must degrade the same way: retried, then failed loudly
                # via ModelClientError rather than crashing the run.
                self.metrics.transport_failures += 1
                last_error = ModelClientError(f"provider transport failed: {exc}")
                continue
            self.metrics.cache_hit_tokens += raw.cache_hit_tokens
            self.metrics.cache_miss_tokens += raw.cache_miss_tokens

            if not raw.content:
                self.metrics.parse_failures += 1
                self.metrics.empty_failures += 1
                last_error = ModelClientError("provider returned an empty response")
                continue

            try:
                data = json.loads(raw.content)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                self.metrics.parse_failures += 1
                self.metrics.malformed_failures += 1
                last_error = ModelClientError(f"provider returned malformed output: {exc}")
                continue

        assert last_error is not None
        raise last_error


class MissingApiKey(Exception):
    """Raised when `api_key_env_var` is not set but a live DeepSeekModelClient
    was requested.
    """


def build_deepseek_client(
    *, usage: str, api_key_env_var: str = API_KEY_ENV_VAR
) -> DeepSeekModelClient:
    """Reads `api_key_env_var` from the environment and builds a
    DeepSeekModelClient over the live transport - the single place every
    entry point (the CLI, the web demo, and the evaluation scripts)
    constructs the real provider, so changing provider or env var is one
    edit. `usage` names what the caller was trying to do (e.g. "to run with
    --live"), appended to the MissingApiKey message so each caller's error
    stays specific without duplicating the check.
    """
    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise MissingApiKey(f"{api_key_env_var} must be set {usage}")
    return DeepSeekModelClient(build_live_transport(api_key))
