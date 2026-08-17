"""The core's public entry point: a Role and a batch of Candidates in, a
Shortlist out. No knowledge of HTTP or the filesystem.
"""

from __future__ import annotations

from collections.abc import Sequence

from screening.domain import (
    Candidate,
    Disqualified,
    Qualified,
    RequirementVerdict,
    Role,
    ScreeningOutcome,
    Shortlist,
    ShortlistEntry,
    Unresolved,
)
from screening.model_client import ModelClient, ModelClientError, ScreeningResponse
from screening.ranking import DEFAULT_TOP_BAND_SIZE, rank_shortlist

# One attempt plus two retries, mirroring ADR-0007's retry-twice-then-fail-
# loudly policy. A response that fails schema validation is already retried
# inside the ModelClient itself; this budget is for a response that *passes*
# validation but doesn't cover the Requirement Set it was asked about - a
# failure mode the ModelClient has no way to see, so core.py retries it.
_MAX_SCREENING_ATTEMPTS = 3


class RequirementSetNotApproved(Exception):
    pass


def run_screening(
    role: Role,
    candidates: Sequence[Candidate],
    model_client: ModelClient,
    top_band_size: int = DEFAULT_TOP_BAND_SIZE,
) -> Shortlist:
    if not role.requirement_set.approved:
        raise RequirementSetNotApproved(
            f"Requirement Set for role {role.id!r} is not approved"
        )

    entries = tuple(
        ShortlistEntry(
            candidate_id=candidate.id,
            outcome=_screen_candidate(role, candidate, model_client),
        )
        for candidate in candidates
    )
    screened = Shortlist(entries=entries)
    return rank_shortlist(role, candidates, screened, model_client, top_band_size=top_band_size)


def build_screening_prompt(role: Role, candidate: Candidate) -> str:
    """Assemble a Screening prompt, stable part first: instructions, then the
    Role, then the Requirement Set, then the Resume.
    """
    requirement_lines = "\n".join(
        f"- ({req.id}) {req.text}" for req in role.requirement_set.requirements
    )
    return (
        "You are screening a Candidate against a Role's Requirement Set.\n"
        "Return a verdict for every Requirement, citing the supporting text.\n"
        "Respond with a JSON object of exactly this shape, and no other "
        "fields:\n"
        '{"verdicts": [{"requirement_id": "<id>", "met": true or false, '
        '"justification": "<supporting text>"}, ...]}\n\n'
        f"Role: {role.title}\n\n"
        f"Requirement Set:\n{requirement_lines}\n\n"
        f"Resume:\n{candidate.resume.text}\n"
    )


def _screen_candidate(
    role: Role, candidate: Candidate, model_client: ModelClient
) -> ScreeningOutcome:
    prompt = build_screening_prompt(role, candidate)
    requirement_ids = [req.id for req in role.requirement_set.requirements]

    for _ in range(_MAX_SCREENING_ATTEMPTS):
        try:
            response = model_client.complete(prompt, ScreeningResponse)
        except ModelClientError:
            return Unresolved(reason="Screening produced no valid verdict")

        response_ids = [v.requirement_id for v in response.verdicts]
        if sorted(response_ids) == sorted(requirement_ids):
            break
    else:
        return Unresolved(
            reason="Screening response did not cover every Requirement exactly once"
        )

    verdicts = tuple(
        RequirementVerdict(
            requirement_id=v.requirement_id, met=v.met, justification=v.justification
        )
        for v in response.verdicts
    )
    verdict_by_requirement = {v.requirement_id: v for v in verdicts}
    missed = tuple(
        req
        for req in role.requirement_set.requirements
        if not verdict_by_requirement[req.id].met
    )
    if missed:
        return Disqualified(verdicts=verdicts, missed=missed)
    return Qualified(verdicts=verdicts)
