"""Tests through the core's public entry point: run_screening(role,
candidates, model_client) -> Shortlist. Per the testing decisions in the
spec, these assert on the Shortlist that comes back and on the calls the
model client received - never on intermediate structure.
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
    ModelClientError,
    RequirementVerdictResponse,
    ScreeningResponse,
)
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
    model_client = RecordingFakeModelClient(responses=[_qualified_response()])

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
        responses=[_qualified_response(), _disqualified_response()]
    )

    run_screening(role, candidates, model_client)

    assert len(model_client.calls) == 2
    assert all(call.response_model is ScreeningResponse for call in model_client.calls)
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


def test_shortlist_preserves_submission_order():
    role = _role()
    candidates = [
        Candidate(id="cara", resume=Resume(text="Cara's resume")),
        Candidate(id="alice", resume=Resume(text="Alice's resume")),
        Candidate(id="bob", resume=Resume(text="Bob's resume")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified_response(), _qualified_response(), _qualified_response()]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert [entry.candidate_id for entry in shortlist.entries] == ["cara", "alice", "bob"]
