"""DeepSeekModelClient owns validation and retry over the raw provider
transport (ADR-0007). These tests inject a fake CompletionTransport so the
retry-and-validation layer is exercised without network access; a separate
test below exercises build_live_transport's request/response shaping against
a monkeypatched urlopen, also without network access.
"""

from __future__ import annotations

import json

import pytest

from screening.core import run_screening
from screening.deepseek_client import (
    DEFAULT_BASE_URL,
    MAX_ATTEMPTS,
    MODEL,
    DeepSeekModelClient,
    RawCompletion,
    build_live_transport,
)
from screening.domain import Candidate, Qualified, Requirement, RequirementSet, Resume, Role, Unresolved
from screening.model_client import ModelClientError, RequirementVerdictResponse, ScreeningResponse

_VALID_PAYLOAD = json.dumps(
    {
        "verdicts": [
            {"requirement_id": "req-python", "met": True, "justification": "Cites Python"},
        ]
    }
)

_VALID_RANKING_PAYLOAD = json.dumps(
    {
        "dimensions": [
            {"dimension": "role_relevance", "rating": "strong", "justification": "Cites Python backend work"},
            {"dimension": "skill_depth", "rating": "moderate", "justification": "Several years listed"},
            {"dimension": "impact_evidence", "rating": "minimal", "justification": "No metrics cited"},
        ]
    }
)


class FakeTransport:
    """Records every prompt it is called with and returns scripted
    RawCompletion responses in order. A scripted Exception is raised instead
    of returned, standing in for a network failure or a malformed provider
    envelope.
    """

    def __init__(self, responses: list[RawCompletion | Exception]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> RawCompletion:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeTransport has no more scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_a_valid_response_on_the_first_attempt_is_returned_and_recorded():
    transport = FakeTransport([RawCompletion(content=_VALID_PAYLOAD, cache_hit_tokens=10, cache_miss_tokens=5)])
    client = DeepSeekModelClient(transport)

    response = client.complete("prompt", ScreeningResponse)

    assert response.verdicts == [
        RequirementVerdictResponse(requirement_id="req-python", met=True, justification="Cites Python")
    ]
    assert len(transport.prompts) == 1
    assert client.metrics.total_attempts == 1
    assert client.metrics.parse_failures == 0
    assert client.metrics.cache_hit_tokens == 10
    assert client.metrics.cache_miss_tokens == 5


def test_a_malformed_response_is_retried_and_a_later_valid_response_succeeds():
    transport = FakeTransport(
        [
            RawCompletion(content="not json"),
            RawCompletion(content=_VALID_PAYLOAD),
        ]
    )
    client = DeepSeekModelClient(transport)

    response = client.complete("prompt", ScreeningResponse)

    assert response.verdicts[0].requirement_id == "req-python"
    assert len(transport.prompts) == 2
    assert client.metrics.total_attempts == 2
    assert client.metrics.parse_failures == 1
    assert client.metrics.malformed_failures == 1
    assert client.metrics.empty_failures == 0


def test_a_response_that_never_validates_is_retried_at_most_twice_then_fails_loudly():
    transport = FakeTransport(
        [
            RawCompletion(content="not json"),
            RawCompletion(content="still not json"),
            RawCompletion(content="never json"),
        ]
    )
    client = DeepSeekModelClient(transport)

    with pytest.raises(ModelClientError):
        client.complete("prompt", ScreeningResponse)

    assert len(transport.prompts) == MAX_ATTEMPTS == 3
    assert client.metrics.total_attempts == 3
    assert client.metrics.parse_failures == 3


def test_no_repair_call_is_made_the_retry_prompt_is_byte_identical():
    transport = FakeTransport(
        [
            RawCompletion(content="not json"),
            RawCompletion(content=_VALID_PAYLOAD),
        ]
    )
    client = DeepSeekModelClient(transport)

    client.complete("the exact same prompt", ScreeningResponse)

    assert transport.prompts == ["the exact same prompt", "the exact same prompt"]


def test_a_transient_transport_failure_is_retried_rather_than_crashing_the_run():
    transport = FakeTransport(
        [
            ConnectionError("connection reset"),
            RawCompletion(content=_VALID_PAYLOAD),
        ]
    )
    client = DeepSeekModelClient(transport)

    response = client.complete("prompt", ScreeningResponse)

    assert response.verdicts[0].requirement_id == "req-python"
    assert client.metrics.transport_failures == 1
    assert client.metrics.parse_failures == 0


def test_a_transport_failure_on_every_attempt_fails_loudly_not_with_a_crash():
    transport = FakeTransport(
        [
            ConnectionError("connection reset"),
            TimeoutError("timed out"),
            KeyError("choices"),
        ]
    )
    client = DeepSeekModelClient(transport)

    with pytest.raises(ModelClientError):
        client.complete("prompt", ScreeningResponse)

    assert client.metrics.transport_failures == 3


def test_an_empty_response_is_a_distinct_failure_kind_from_a_malformed_one():
    transport = FakeTransport(
        [
            RawCompletion(content=None),
            RawCompletion(content=""),
            RawCompletion(content="not json"),
        ]
    )
    client = DeepSeekModelClient(transport)

    with pytest.raises(ModelClientError):
        client.complete("prompt", ScreeningResponse)

    assert client.metrics.empty_failures == 2
    assert client.metrics.malformed_failures == 1
    assert client.metrics.parse_failures == 3


def test_a_response_that_fails_schema_validation_counts_as_a_malformed_failure():
    invalid_shape = json.dumps({"verdicts": [{"requirement_id": "req-python"}]})
    transport = FakeTransport([RawCompletion(content=invalid_shape)])
    client = DeepSeekModelClient(transport, max_attempts=1)

    with pytest.raises(ModelClientError):
        client.complete("prompt", ScreeningResponse)

    assert client.metrics.malformed_failures == 1


def test_metrics_accumulate_cache_tokens_across_multiple_calls_on_one_client():
    transport = FakeTransport(
        [
            RawCompletion(content=_VALID_PAYLOAD, cache_hit_tokens=100, cache_miss_tokens=20),
            RawCompletion(content=_VALID_PAYLOAD, cache_hit_tokens=150, cache_miss_tokens=0),
        ]
    )
    client = DeepSeekModelClient(transport)

    client.complete("prompt one", ScreeningResponse)
    client.complete("prompt two", ScreeningResponse)

    assert client.metrics.cache_hit_tokens == 250
    assert client.metrics.cache_miss_tokens == 20


def test_a_screening_run_completes_end_to_end_through_deepseek_v4_flash():
    role = Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(
            requirements=(Requirement(id="req-python", text="5+ years of Python"),),
            approved=True,
        ),
    )
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice's resume")),
        Candidate(id="bob", resume=Resume(text="Bob's resume")),
    ]
    unresolved_payload = json.dumps({"verdicts": [{"requirement_id": "req-python"}]})
    transport = FakeTransport(
        [
            RawCompletion(content=_VALID_PAYLOAD),
            RawCompletion(content=unresolved_payload),
            RawCompletion(content=unresolved_payload),
            RawCompletion(content=unresolved_payload),
            RawCompletion(content=_VALID_RANKING_PAYLOAD),
        ]
    )
    client = DeepSeekModelClient(transport)

    shortlist = run_screening(role, candidates, client)

    outcomes = {entry.candidate_id: entry.outcome for entry in shortlist.entries}
    assert isinstance(outcomes["alice"], Qualified)
    assert outcomes["alice"].fit is not None
    assert isinstance(outcomes["bob"], Unresolved)
    assert client.metrics.total_attempts == 5


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_live_transport_sends_model_and_json_mode_and_parses_usage(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        return _FakeHttpResponse(
            {
                "choices": [{"message": {"content": _VALID_PAYLOAD}}],
                "usage": {"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 3},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = build_live_transport("secret-key")

    result = transport("a rendered prompt")

    assert captured["url"] == DEFAULT_BASE_URL
    assert captured["body"]["model"] == MODEL
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["messages"] == [{"role": "user", "content": "a rendered prompt"}]
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert result.content == _VALID_PAYLOAD
    assert result.cache_hit_tokens == 7
    assert result.cache_miss_tokens == 3


def test_live_transport_reports_an_empty_provider_response_as_none(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeHttpResponse({"choices": [{"message": {"content": ""}}], "usage": {}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = build_live_transport("secret-key")

    result = transport("a rendered prompt")

    assert result.content is None


def test_live_transport_treats_a_null_cache_token_count_as_zero(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeHttpResponse(
            {
                "choices": [{"message": {"content": _VALID_PAYLOAD}}],
                "usage": {"prompt_cache_hit_tokens": None, "prompt_cache_miss_tokens": None},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transport = build_live_transport("secret-key")

    result = transport("a rendered prompt")

    assert result.cache_hit_tokens == 0
    assert result.cache_miss_tokens == 0
