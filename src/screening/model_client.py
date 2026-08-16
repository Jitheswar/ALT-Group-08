"""The core's only injected dependency: a prompt and an expected schema in,
a validated object out.

The production implementation (ticket 02) targets deepseek-v4-flash and owns
its own validation and retry; this module only defines the seam and the
schema Screening calls through it.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelClientError(Exception):
    """Raised when a model client cannot produce a valid object for a prompt."""


class ModelClient(Protocol):
    def complete(self, prompt: str, response_model: type[T]) -> T: ...


class RequirementVerdictResponse(BaseModel):
    requirement_id: str
    met: bool
    justification: str


class ScreeningResponse(BaseModel):
    verdicts: list[RequirementVerdictResponse]
