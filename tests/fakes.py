"""The model client seam's test double: a scripted fake that returns canned
responses in call order and records every call it receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedCall:
    prompt: str
    response_model: type


@dataclass
class RecordingFakeModelClient:
    responses: list[Any]
    calls: list[RecordedCall] = field(default_factory=list)

    def complete(self, prompt: str, response_model: type) -> Any:
        self.calls.append(RecordedCall(prompt=prompt, response_model=response_model))
        if not self.responses:
            raise AssertionError("RecordingFakeModelClient has no more scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
