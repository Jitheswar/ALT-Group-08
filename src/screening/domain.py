"""Domain types shared by every seam in the core.

Vocabulary follows CONTEXT.md: Role, Job Description, Requirement,
Requirement Set, Candidate, Resume, Screening, Qualified/Unresolved
Candidate, Shortlist, Justification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, TypeVar


@dataclass(frozen=True)
class JobDescription:
    text: str


@dataclass(frozen=True)
class Requirement:
    id: str
    text: str


@dataclass(frozen=True)
class RequirementSet:
    requirements: tuple[Requirement, ...]
    approved: bool = False


def approve_requirement_set(requirements: Sequence[Requirement]) -> RequirementSet:
    """The single place a Requirement Set becomes approved. Extraction never
    calls this - it only proposes (screening.extraction.extract_requirements
    returns a plain tuple of Requirement, never a RequirementSet) - so a
    Requirement Set can only reach Screening after an explicit Recruiter
    approval step, per ADR-0004.
    """
    return RequirementSet(requirements=tuple(requirements), approved=True)


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    requirement_set: RequirementSet


@dataclass(frozen=True)
class Resume:
    text: str


@dataclass(frozen=True)
class Candidate:
    id: str
    resume: Resume


@dataclass(frozen=True)
class RequirementVerdict:
    requirement_id: str
    met: bool
    justification: str


@dataclass(frozen=True)
class FitDimension:
    name: str
    rating: str
    justification: str


@dataclass(frozen=True)
class Fit:
    """The graded, non-binary judgement of how well a Qualified Candidate
    suits a Role: one rating per rubric dimension, never a single scalar -
    a scalar score clusters too tightly across Candidates to rank with.
    """

    dimensions: tuple[FitDimension, ...]


@dataclass(frozen=True)
class Qualified:
    """A Candidate who met every Requirement of the Role. `fit` is attached
    by Ranking, which runs after Screening, so it starts as None; a
    Qualified Candidate Ranking failed to judge keeps fit=None rather than
    being dropped or turned into an Unresolved Candidate - that outcome is
    reserved for Screening producing no valid verdict.
    """

    verdicts: tuple[RequirementVerdict, ...]
    fit: Fit | None = None


@dataclass(frozen=True)
class Disqualified:
    """A Candidate who missed one or more Requirements."""

    verdicts: tuple[RequirementVerdict, ...]
    missed: tuple[Requirement, ...]


@dataclass(frozen=True)
class Unresolved:
    """A Candidate whose Screening produced no valid verdict."""

    reason: str


ScreeningOutcome = Qualified | Disqualified | Unresolved


@dataclass(frozen=True)
class ShortlistEntry:
    candidate_id: str
    outcome: ScreeningOutcome


@dataclass(frozen=True)
class Shortlist:
    entries: tuple[ShortlistEntry, ...]


R = TypeVar("R")


def match_outcome(
    outcome: ScreeningOutcome,
    *,
    qualified: Callable[[Qualified], R],
    disqualified: Callable[[Disqualified], R],
    unresolved: Callable[[Unresolved], R],
) -> R:
    """The single dispatch point over ScreeningOutcome's variants, so a new
    variant fails loudly here rather than silently in every consumer.
    """
    if isinstance(outcome, Qualified):
        return qualified(outcome)
    if isinstance(outcome, Disqualified):
        return disqualified(outcome)
    if isinstance(outcome, Unresolved):
        return unresolved(outcome)
    raise TypeError(f"Unknown ScreeningOutcome: {outcome!r}")
