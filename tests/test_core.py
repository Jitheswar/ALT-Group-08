"""Tests through the core's public entry point: run_screening(role,
candidates, model_client) -> Shortlist. Per the testing decisions in the
spec, these assert on the Shortlist that comes back and on the calls the
model client received - never on intermediate structure.

Since ticket 04, run_screening also drives rubric-based Ranking over every
Qualified Candidate, so any test that produces a Qualified outcome must
script a matching RankingResponse after the Screening responses - the model
client sees Screening calls for the whole batch first, then Ranking calls
for the Qualified subset, in shortlist order. Since ticket 05, a comparative
pass runs after that whenever two or more Qualified Candidates land in the
top band (the default band comfortably covers every batch in this file), so
a test with exactly two Qualified outcomes must also script one matching
ComparativeResponse - `sorted` makes exactly one comparison for a band of
two, regardless of its internal implementation. A test whose point is
rubric-stage behaviour specifically, rather than the comparative pass,
instead passes top_band_size=0 to opt the comparative pass out entirely.
Ranking-specific behaviour (ordering by Fit, the comparative pass,
redaction, the ADR-0002/ADR-0005 seam guarantees) lives in
tests/test_ranking.py; these tests only script it minimally to keep
run_screening's own invariants passing.
"""

from __future__ import annotations

import pytest

from screening.core import RequirementSetNotApproved, run_screening
from screening.domain import (
    Candidate,
    Disqualified,
    Qualified,
    Requirement,
    RequirementSet,
    Resume,
    Role,
    Unresolved,
)
from screening.model_client import (
    ComparativeResponse,
    ComparativeWinner,
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
REQ_DEGREE = Requirement(id="req-degree", text="Bachelor's degree in Computer Science")


def _role(approved: bool = True) -> Role:
    return Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(
            requirements=(REQ_PYTHON, REQ_DEGREE), approved=approved
        ),
    )


def _qualified_response() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python",
                met=True,
                justification='Resume cites "8 years of Python" under Experience',
            ),
            RequirementVerdictResponse(
                requirement_id="req-degree",
                met=True,
                justification='Resume cites "B.Sc. Computer Science" under Education',
            ),
        ]
    )


def _disqualified_response() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python",
                met=True,
                justification='Resume cites "6 years of Python" under Experience',
            ),
            RequirementVerdictResponse(
                requirement_id="req-degree",
                met=False,
                justification="No degree listed under Education",
            ),
        ]
    )


def _fit_response(rating: FitRating = "strong") -> RankingResponse:
    """A RankingResponse covering exactly DIMENSIONS - what a Qualified
    Candidate needs scripted after their Screening response for the model
    client's response queue to line up.
    """
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(
                dimension=dimension,
                rating=rating,
                justification=f"Resume evidences {dimension.replace('_', ' ')}",
            )
            for dimension in DIMENSIONS
        ]
    )


def _comparative_response(winner: ComparativeWinner = "a") -> ComparativeResponse:
    return ComparativeResponse(winner=winner)


def test_every_submitted_candidate_appears_exactly_once_whatever_happened():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice's resume")),
        Candidate(id="bob", resume=Resume(text="Bob's resume")),
        Candidate(id="cara", resume=Resume(text="Cara's resume")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified_response(),
            _disqualified_response(),
            ModelClientError("provider returned malformed output"),
            _fit_response(),
        ]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert [entry.candidate_id for entry in shortlist.entries] == ["alice", "bob", "cara"]
    outcomes = {entry.candidate_id: entry.outcome for entry in shortlist.entries}
    assert isinstance(outcomes["alice"], Qualified)
    assert isinstance(outcomes["bob"], Disqualified)
    assert isinstance(outcomes["cara"], Unresolved)


def test_disqualified_candidate_is_marked_with_the_requirement_missed():
    role = _role()
    candidates = [Candidate(id="bob", resume=Resume(text="Bob's resume"))]
    model_client = RecordingFakeModelClient(responses=[_disqualified_response()])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Disqualified)
    assert entry.outcome.missed == (REQ_DEGREE,)


def test_screening_outcomes_carry_a_justification_citing_supporting_text():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice's resume"))]
    model_client = RecordingFakeModelClient(responses=[_qualified_response(), _fit_response()])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    for verdict in entry.outcome.verdicts:
        assert verdict.justification
        assert "Resume cites" in verdict.justification


def test_unapproved_requirement_set_is_refused_before_any_model_call():
    role = _role(approved=False)
    candidates = [Candidate(id="alice", resume=Resume(text="Alice's resume"))]
    model_client = RecordingFakeModelClient(responses=[_qualified_response()])

    with pytest.raises(RequirementSetNotApproved):
        run_screening(role, candidates, model_client)

    assert model_client.calls == []


def test_model_client_double_records_every_call_it_receives():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice's resume")),
        Candidate(id="bob", resume=Resume(text="Bob's resume")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified_response(), _disqualified_response(), _fit_response()]
    )

    run_screening(role, candidates, model_client)

    assert len(model_client.calls) == 3
    assert model_client.calls[0].response_model is ScreeningResponse
    assert model_client.calls[1].response_model is ScreeningResponse
    assert model_client.calls[2].response_model is RankingResponse
    assert "Alice's resume" in model_client.calls[0].prompt
    assert "Bob's resume" in model_client.calls[1].prompt


def test_a_response_with_a_duplicate_verdict_becomes_unresolved():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice's resume"))]
    response_with_duplicate = ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python", met=False, justification="No Python listed"
            ),
            RequirementVerdictResponse(
                requirement_id="req-degree", met=True, justification="B.Sc. listed"
            ),
            RequirementVerdictResponse(
                requirement_id="req-python", met=True, justification="Actually 6 years of Python"
            ),
        ]
    )
    model_client = RecordingFakeModelClient(responses=[response_with_duplicate])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Unresolved)


def test_stable_prompt_prefix_is_byte_identical_across_every_call_in_a_run():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice's resume, ten years of Python.")),
        Candidate(id="bob", resume=Resume(text="Bob's resume, five years of Rust.")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified_response(),
            _qualified_response(),
            _fit_response(),
            _fit_response(),
            _comparative_response(),
        ]
    )

    run_screening(role, candidates, model_client)

    screening_calls = [c for c in model_client.calls if c.response_model is ScreeningResponse]
    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]

    # The stable prefix is per prompt template, not shared across templates:
    # Screening and Ranking prompts differ by design, but each template's
    # own prefix must be byte-identical across every candidate in the run.
    screening_prefixes = [call.prompt.split("Resume:\n", 1)[0] for call in screening_calls]
    assert len(screening_prefixes) == 2
    assert screening_prefixes[0] == screening_prefixes[1]

    ranking_prefixes = [call.prompt.split("Redacted Resume:\n", 1)[0] for call in ranking_calls]
    assert len(ranking_prefixes) == 2
    assert ranking_prefixes[0] == ranking_prefixes[1]


def test_shortlist_preserves_submission_order_among_equally_fit_qualified_candidates():
    """A rubric-stage property, so top_band_size=0 keeps the comparative
    pass out of it entirely - which pairs within a tied band it would
    compare, and in what order, is an implementation detail of the stdlib
    sort rather than something this test should depend on.
    """
    role = _role()
    candidates = [
        Candidate(id="cara", resume=Resume(text="Cara's resume")),
        Candidate(id="alice", resume=Resume(text="Alice's resume")),
        Candidate(id="bob", resume=Resume(text="Bob's resume")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified_response(),
            _qualified_response(),
            _qualified_response(),
            _fit_response(),
            _fit_response(),
            _fit_response(),
        ]
    )

    shortlist = run_screening(role, candidates, model_client, top_band_size=0)

    assert [entry.candidate_id for entry in shortlist.entries] == ["cara", "alice", "bob"]
