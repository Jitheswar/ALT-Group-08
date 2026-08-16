"""The core's only injected dependency: a prompt and an expected schema in,
a validated object out.

The production implementation (ticket 02) targets deepseek-v4-flash and owns
its own validation and retry; this module only defines the seam and the
schema Screening calls through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelClientError(Exception):
    """Raised when a model client cannot produce a valid object for a prompt."""


class ModelClient(Protocol):
    def complete(self, prompt: str, response_model: type[T]) -> T: ...


@dataclass
class RunMetrics:
    """Reliability and cost counters a ModelClient may accumulate across a
    run, read afterwards for the run record (ADR-0007). A client that has
    nothing to report - the scripted double, for instance - simply has no
    `metrics` attribute; callers read it with `getattr(client, "metrics",
    RunMetrics())`.
    """

    total_attempts: int = 0
    parse_failures: int = 0
    empty_failures: int = 0
    malformed_failures: int = 0
    transport_failures: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.total_attempts if self.total_attempts else 0.0


class RequirementVerdictResponse(BaseModel):
    requirement_id: str
    met: bool
    justification: str


class ScreeningResponse(BaseModel):
    verdicts: list[RequirementVerdictResponse]


class ProposedRequirementResponse(BaseModel):
    text: str


class ExtractionResponse(BaseModel):
    requirements: list[ProposedRequirementResponse]
