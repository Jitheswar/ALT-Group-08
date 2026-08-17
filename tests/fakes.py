"""The model client seam's test double: a scripted fake that returns canned
responses in call order and records every call it receives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from screening.model_client import ComparativeResponse, ComparativeWinner, RunMetrics


@dataclass
class RecordedCall:
    prompt: str
    response_model: type


@dataclass
class RecordingFakeModelClient:
    responses: list[Any]
    calls: list[RecordedCall] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def complete(self, prompt: str, response_model: type) -> Any:
        self.calls.append(RecordedCall(prompt=prompt, response_model=response_model))
        if not self.responses:
            raise AssertionError("RecordingFakeModelClient has no more scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


_STRENGTH = re.compile(r"STRENGTH:(\d+)")


@dataclass
class StrengthBasedComparativeFakeModelClient:
    """Like RecordingFakeModelClient - Screening and Ranking responses are
    scripted and popped in call order - except ComparativeResponse calls
    are judged from a "STRENGTH:<n>" marker embedded in each Candidate's
    resume text rather than scripted by position.

    The comparative pass's underlying sort is free to compare whichever
    pairs it likes in whatever order, so a test asserting on the *result*
    of the comparative pass (rather than on a specific call it makes) needs
    a comparator that is a pure function of its two inputs - exactly what a
    real comparator is - rather than one indexed by call order.
    """

    responses: list[Any]
    calls: list[RecordedCall] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def complete(self, prompt: str, response_model: type) -> Any:
        self.calls.append(RecordedCall(prompt=prompt, response_model=response_model))
        if response_model is ComparativeResponse:
            return self._judge(prompt)
        if not self.responses:
            raise AssertionError(
                "StrengthBasedComparativeFakeModelClient has no more scripted responses"
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def _judge(self, prompt: str) -> ComparativeResponse:
        _, _, after_a = prompt.partition("Candidate A:\n")
        resume_a_text, _, resume_b_text = after_a.partition("\n\nCandidate B:\n")
        winner: ComparativeWinner = (
            "a" if _strength(resume_a_text) >= _strength(resume_b_text) else "b"
        )
        return ComparativeResponse(winner=winner, justification=f"Candidate {winner} is stronger")


def _strength(resume_text: str) -> int:
    match = _STRENGTH.search(resume_text)
    if not match:
        raise AssertionError("resume text is missing a STRENGTH:<n> marker for this fake")
    return int(match.group(1))
