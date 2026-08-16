"""Rubric-based Ranking: attaches a structured per-dimension Fit judgement
to every Qualified Candidate, on the Redacted Resume only (ADR-0005), and
orders the Qualified group by it.

A second, comparative pass then reorders only the top band of that rubric
ordering - the part a Recruiter actually reads and the part rank metrics
are computed over - by judging Candidates against each other rather than
independently. It is a stable sort (`sorted` via `functools.cmp_to_key`)
whose comparator is a model call, so it makes on the order of n log n
calls over the band rather than the full batch. A comparator failure is
treated as a tie, which a stable sort resolves by keeping that pair in
rubric order - but, as with any comparison sort, only a subset of all
pairs are ever directly compared, so a tie (or an inconsistent judgement
elsewhere in the band) can still influence where other elements land.

Ranking never calls the model client for a Candidate who failed Screening
(ADR-0002): the loop below only ever redacts and scores entries whose
outcome is Qualified, so a Disqualified or Unresolved Candidate's Resume -
redacted or not - is never part of a Ranking or comparative prompt. The
comparative pass reuses the same redacted text computed for the rubric
pass, so there remains no code path by which either pass sees an
unredacted Resume (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cmp_to_key

from screening.domain import (
    Candidate,
    Fit,
    FitDimension,
    Qualified,
    Role,
    Shortlist,
    ShortlistEntry,
)
from screening.model_client import (
    ComparativeResponse,
    ModelClient,
    ModelClientError,
    RankingResponse,
)
from screening.redaction import redact_resume

# Fixed rather than model-chosen, so every Ranking response is checked
# against a known set - the same discipline Screening applies to
# Requirement ids.
DIMENSIONS: tuple[str, ...] = ("role_relevance", "skill_depth", "impact_evidence")

_RATING_LEVELS = ("minimal", "moderate", "strong", "exceptional")
_RATING_WEIGHT = {level: weight for weight, level in enumerate(_RATING_LEVELS)}
_DIMENSION_ORDER = {name: index for index, name in enumerate(DIMENSIONS)}

# "Roughly twenty Candidates" per the spec: the top of the Shortlist a
# Recruiter actually reads, and the part rank metrics are computed over.
DEFAULT_TOP_BAND_SIZE = 20


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
    rubric-only ordering, coarser than the comparative pass layered over
    the top band below.
    """
    return sum(_RATING_WEIGHT[dimension.rating] for dimension in fit.dimensions)


def build_comparative_prompt(role: Role, resume_a_text: str, resume_b_text: str) -> str:
    return (
        "You are comparing two Qualified Candidates' Fit for a Role.\n"
        "Decide which Candidate is the stronger overall Fit.\n\n"
        f"Role: {role.title}\n\n"
        f"Candidate A:\n{resume_a_text}\n\n"
        f"Candidate B:\n{resume_b_text}\n"
    )


def rank_shortlist(
    role: Role,
    candidates: Sequence[Candidate],
    shortlist: Shortlist,
    model_client: ModelClient,
    top_band_size: int = DEFAULT_TOP_BAND_SIZE,
) -> Shortlist:
    """Reorders a Screening Shortlist's Qualified entries by rubric-based
    Fit, best first, then reorders the top `top_band_size` of those further
    by a comparative pass. Rubric-unranked entries are appended next, in
    the order Screening produced them, followed by the Disqualified and
    Unresolved entries, also in the order Screening produced them - so
    every submitted Candidate still appears exactly once.
    """
    resume_by_id = {candidate.id: candidate.resume for candidate in candidates}
    redacted_text_by_id: dict[str, str] = {}

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
        redacted_text_by_id[entry.candidate_id] = redacted_resume.text

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    ranked_entries = [entry for entry, _ in ranked]

    def compare(a: ShortlistEntry, b: ShortlistEntry) -> int:
        """-1 if `a` is the stronger Fit, 1 if `b` is, 0 (a tie) if the
        comparative call failed - `sorted` resolves a tie by keeping the
        two entries in their current (rubric) relative order.
        """
        prompt = build_comparative_prompt(
            role, redacted_text_by_id[a.candidate_id], redacted_text_by_id[b.candidate_id]
        )
        try:
            response = model_client.complete(prompt, ComparativeResponse)
        except ModelClientError:
            return 0
        return -1 if response.winner == "a" else 1

    band = sorted(ranked_entries[:top_band_size], key=cmp_to_key(compare))
    tail = ranked_entries[top_band_size:]

    return Shortlist(
        entries=tuple(band) + tuple(tail) + tuple(unranked) + tuple(passthrough)
    )
