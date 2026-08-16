"""Rubric-based Ranking: attaches a structured per-dimension Fit judgement
to every Qualified Candidate, on the Redacted Resume only (ADR-0005), and
orders the Qualified group by it.

Ranking never calls the model client for a Candidate who failed Screening
(ADR-0002): the loop below only ever redacts and scores entries whose
outcome is Qualified, so a Disqualified or Unresolved Candidate's Resume -
redacted or not - is never part of a Ranking prompt.

This ticket covers rubric-only Ranking. The comparative pass over the top
band that reorders it further is a later ticket.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from screening.domain import (
    Candidate,
    Fit,
    FitDimension,
    Qualified,
    Role,
    Shortlist,
    ShortlistEntry,
)
from screening.model_client import ModelClient, ModelClientError, RankingResponse
from screening.redaction import redact_resume

# Fixed rather than model-chosen, so every Ranking response is checked
# against a known set - the same discipline Screening applies to
# Requirement ids.
DIMENSIONS: tuple[str, ...] = ("role_relevance", "skill_depth", "impact_evidence")

_RATING_LEVELS = ("minimal", "moderate", "strong", "exceptional")
_RATING_WEIGHT = {level: weight for weight, level in enumerate(_RATING_LEVELS)}
_DIMENSION_ORDER = {name: index for index, name in enumerate(DIMENSIONS)}


def build_ranking_prompt(role: Role, redacted_resume_text: str) -> str:
    dimension_lines = "\n".join(f"- {dimension}" for dimension in DIMENSIONS)
    return (
        "You are judging a Qualified Candidate's Fit for a Role.\n"
        "Rate every dimension below as one of minimal, moderate, strong, "
        "exceptional, citing the supporting text for each rating.\n\n"
        f"Role: {role.title}\n\n"
        f"Dimensions:\n{dimension_lines}\n\n"
        f"Redacted Resume:\n{redacted_resume_text}\n"
    )


def fit_weight(fit: Fit) -> int:
    """Total ordinal weight across dimensions - higher sorts first. A
    rubric-only ordering, coarser than the comparative pass a later ticket
    layers over the top band.
    """
    return sum(_RATING_WEIGHT[dimension.rating] for dimension in fit.dimensions)


def rank_shortlist(
    role: Role,
    candidates: Sequence[Candidate],
    shortlist: Shortlist,
    model_client: ModelClient,
) -> Shortlist:
    """Reorders a Screening Shortlist's Qualified entries by rubric-based
    Fit, best first. Disqualified and Unresolved entries pass through
    unchanged and are appended after, in the order Screening produced them,
    so every submitted Candidate still appears exactly once.
    """
    resume_by_id = {candidate.id: candidate.resume for candidate in candidates}

    ranked: list[tuple[ShortlistEntry, int]] = []
    unranked: list[ShortlistEntry] = []
    passthrough: list[ShortlistEntry] = []

    for entry in shortlist.entries:
        if not isinstance(entry.outcome, Qualified):
            passthrough.append(entry)
            continue

        redacted_resume = redact_resume(resume_by_id[entry.candidate_id])
        prompt = build_ranking_prompt(role, redacted_resume.text)
        try:
            response = model_client.complete(prompt, RankingResponse)
        except ModelClientError:
            unranked.append(entry)
            continue

        response_names = {d.dimension for d in response.dimensions}
        if response_names != set(DIMENSIONS) or len(response.dimensions) != len(DIMENSIONS):
            unranked.append(entry)
            continue

        fit = Fit(
            dimensions=tuple(
                FitDimension(name=d.dimension, rating=d.rating, justification=d.justification)
                for d in sorted(response.dimensions, key=lambda d: _DIMENSION_ORDER[d.dimension])
            )
        )
        ranked_entry = replace(entry, outcome=replace(entry.outcome, fit=fit))
        ranked.append((ranked_entry, fit_weight(fit)))

    ranked.sort(key=lambda pair: pair[1], reverse=True)

    return Shortlist(
        entries=tuple(entry for entry, _ in ranked) + tuple(unranked) + tuple(passthrough)
    )
