"""Tests through the core's public entry point for rubric-based Ranking:
run_screening(role, candidates, model_client) -> Shortlist, with Ranking now
wired in after Screening. Per the testing decisions in the spec, these
assert on the Shortlist that comes back and on the calls the model client
received - the seam that makes ADR-0002 and ADR-0005 testable rather than
merely intended.
"""

from __future__ import annotations

from screening.core import run_screening
from screening.domain import (
    Candidate,
    Disqualified,
    Fit,
    Qualified,
    Requirement,
    RequirementSet,
    Resume,
    Role,
    Unresolved,
)
from screening.model_client import (
    FitDimensionResponse,
    FitRating,
    ModelClientError,
    RankingResponse,
    RequirementVerdictResponse,
    ScreeningResponse,
)
from screening.ranking import DIMENSIONS
from tests.fakes import RecordingFakeModelClient

REQ_PYTHON = Requirement(id="req-python", text="5+ years of Python")


def _role() -> Role:
    return Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(requirements=(REQ_PYTHON,), approved=True),
    )


def _qualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python", met=True, justification="Cites 8 years of Python"
            )
        ]
    )


def _disqualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python", met=False, justification="No Python listed"
            )
        ]
    )


def _fit(rating: FitRating) -> RankingResponse:
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(
                dimension=dimension, rating=rating, justification=f"Evidences {dimension}"
            )
            for dimension in DIMENSIONS
        ]
    )


def test_qualified_candidates_are_ordered_by_fit_best_first():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nSenior backend engineer.")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nJunior backend engineer.")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), _qualified(), _fit("minimal"), _fit("exceptional")]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert [entry.candidate_id for entry in shortlist.entries] == ["bob", "alice"]


def test_qualified_candidates_carry_a_structured_per_dimension_fit_not_a_bare_scalar():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    fit = entry.outcome.fit
    assert isinstance(fit, Fit)
    assert len(fit.dimensions) >= 2
    for dimension in fit.dimensions:
        assert dimension.justification


def test_disqualified_and_unresolved_candidates_still_appear_alongside_ranked_qualified():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer.")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nNo Python here.")),
        Candidate(id="cara", resume=Resume(text="Cara Lee\nBackend engineer.")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified(),
            _disqualified(),
            ModelClientError("provider returned malformed output"),
            _fit("strong"),
        ]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert {entry.candidate_id for entry in shortlist.entries} == {"alice", "bob", "cara"}
    outcomes = {entry.candidate_id: entry.outcome for entry in shortlist.entries}
    assert isinstance(outcomes["alice"], Qualified)
    assert isinstance(outcomes["bob"], Disqualified)
    assert isinstance(outcomes["cara"], Unresolved)


def test_no_ranking_call_ever_receives_a_candidate_who_failed_screening():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer.")),
        Candidate(
            id="bob",
            resume=Resume(text="Bob Roe\nNo Python here, distinctively unique bob-only text."),
        ),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), _disqualified(), _fit("strong")]
    )

    run_screening(role, candidates, model_client)

    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]
    assert len(ranking_calls) == 1
    assert "distinctively unique bob-only text" not in ranking_calls[0].prompt


def test_no_ranking_call_ever_receives_an_unredacted_resume():
    role = _role()
    candidates = [
        Candidate(
            id="alice",
            resume=Resume(text="Alice Doe\nBackend engineer with 8 years of Python."),
        )
    ]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    run_screening(role, candidates, model_client)

    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]
    [call] = ranking_calls
    assert "Alice" not in call.prompt
    assert "Doe" not in call.prompt
    assert "[REDACTED-NAME]" in call.prompt


def test_screening_still_receives_the_unredacted_resume():
    role = _role()
    candidates = [
        Candidate(
            id="alice",
            resume=Resume(text="Alice Doe\nBackend engineer with 8 years of Python."),
        )
    ]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    run_screening(role, candidates, model_client)

    screening_calls = [c for c in model_client.calls if c.response_model is ScreeningResponse]
    [call] = screening_calls
    assert "Alice Doe" in call.prompt


def test_a_qualified_candidate_whose_ranking_call_fails_is_not_dropped():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), ModelClientError("provider returned malformed output")]
    )

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    assert entry.outcome.fit is None


def test_a_ranking_response_missing_a_dimension_leaves_the_candidate_unranked_not_dropped():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    incomplete = RankingResponse(
        dimensions=[
            FitDimensionResponse(
                dimension=DIMENSIONS[0], rating="strong", justification="Evidences it"
            )
        ]
    )
    model_client = RecordingFakeModelClient(responses=[_qualified(), incomplete])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    assert entry.outcome.fit is None
