"""Tests through the extraction seam: extract_requirements(job_description,
model_client) -> tuple[Requirement, ...], and the approve_requirement_set
step a Recruiter drives afterwards. Per ADR-0004, extraction only proposes -
these tests assert that a RequirementSet only ever comes from an explicit
approval, never from extraction itself.
"""

from __future__ import annotations

from dataclasses import replace

from screening.domain import JobDescription, Requirement, approve_requirement_set
from screening.extraction import build_extraction_prompt, extract_requirements
from screening.model_client import ExtractionResponse, ProposedRequirementResponse
from tests.fakes import RecordingFakeModelClient


def _response(*texts: str) -> ExtractionResponse:
    return ExtractionResponse(
        requirements=[ProposedRequirementResponse(text=text) for text in texts]
    )


def test_extraction_returns_discrete_individually_checkable_requirements():
    job_description = JobDescription(
        text="Backend Engineer\n- 5+ years of Python\n- Bachelor's degree in CS\n"
    )
    model_client = RecordingFakeModelClient(
        responses=[_response("5+ years of Python", "Bachelor's degree in CS")]
    )

    proposed = extract_requirements(job_description, model_client)

    assert proposed == (
        Requirement(id="req-1", text="5+ years of Python"),
        Requirement(id="req-2", text="Bachelor's degree in CS"),
    )


def test_extraction_never_returns_a_requirement_set():
    job_description = JobDescription(text="- Python experience\n")
    model_client = RecordingFakeModelClient(responses=[_response("Python experience")])

    proposed = extract_requirements(job_description, model_client)

    assert isinstance(proposed, tuple)
    assert all(isinstance(item, Requirement) for item in proposed)
    assert not hasattr(proposed, "approved")


def test_extraction_calls_the_model_client_with_the_extraction_response_schema():
    job_description = JobDescription(text="- Python experience\n")
    model_client = RecordingFakeModelClient(responses=[_response("Python experience")])

    extract_requirements(job_description, model_client)

    [call] = model_client.calls
    assert call.response_model is ExtractionResponse
    assert "Python experience" in call.prompt


def test_extraction_prompt_carries_the_job_description_text():
    job_description = JobDescription(text="- Rust systems programming\n")

    prompt = build_extraction_prompt(job_description)

    assert "Rust systems programming" in prompt


def test_a_proposed_requirement_can_be_edited_deleted_or_added_before_approval():
    job_description = JobDescription(text="doesn't matter for this test")
    model_client = RecordingFakeModelClient(
        responses=[_response("5+ years of Python", "Willing to relocate")]
    )
    proposed = extract_requirements(job_description, model_client)

    edited = replace(proposed[0], text="3+ years of Python")
    added = Requirement(id="req-custom-1", text="Bachelor's degree in CS")
    reviewed = (edited, added)  # proposed[1] ("Willing to relocate") deleted

    approved = approve_requirement_set(reviewed)

    assert approved.approved is True
    assert approved.requirements == reviewed


def test_approval_is_the_only_way_a_requirement_set_becomes_approved():
    proposed = (Requirement(id="req-1", text="Python experience"),)

    approved = approve_requirement_set(proposed)

    assert approved.approved is True
    assert approved.requirements == proposed
