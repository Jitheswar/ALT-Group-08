"""Proxy Relevance construction (ADR-0003, CONTEXT.md): a corpus Resume is
judged relevant to an Evaluation Role's category if it shares that
category, and not relevant otherwise. This is a proxy for a Recruiter's
judgement, not a substitute for it - the name is never shortened to
"relevance" or "labels" anywhere it is surfaced, so nobody mistakes it for
ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusResume:
    """One row of the resume-atlas corpus, reduced to what the sweep
    (ticket 07) needs: an id to key a Screening Run's Shortlist entries
    back to this row, the category Proxy Relevance is constructed from,
    and the Resume text itself.
    """

    candidate_id: str
    category: str
    text: str


def construct_proxy_relevance(
    role_category: str, resumes: Sequence[CorpusResume]
) -> dict[str, bool]:
    """Proxy Relevance for a batch of corpus Resumes against one
    Evaluation Role's category: a Resume sharing the Role's category is
    relevant, every other Resume is not (ADR-0003).
    """
    return {resume.candidate_id: resume.category == role_category for resume in resumes}
