"""The scripted client is exercised directly on the prompt shape build_screening_prompt
produces - it is a real ModelClient implementation, not a test double."""

from __future__ import annotations

from screening.core import build_screening_prompt
from screening.domain import Candidate, JobDescription, Requirement, RequirementSet, Resume, Role
from screening.extraction import build_extraction_prompt
from screening.model_client import ExtractionResponse, RankingResponse, ScreeningResponse
from screening.ranking import DIMENSIONS, build_ranking_prompt
from screening.scripted_client import ScriptedModelClient


def _prompt(requirement_text: str, resume_text: str) -> str:
    role = Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(
            requirements=(Requirement(id="req-1", text=requirement_text),),
            approved=True,
        ),
    )
    candidate = Candidate(id="alice", resume=Resume(text=resume_text))
    return build_screening_prompt(role, candidate)


def test_scripted_client_finds_a_requirement_evidenced_in_the_resume():
    client = ScriptedModelClient()
    prompt = _prompt(
        "Python programming experience",
        "Built backend services in Python for six years.",
    )

    response = client.complete(prompt, ScreeningResponse)

    [verdict] = response.verdicts
    assert verdict.requirement_id == "req-1"
    assert verdict.met is True
    assert "python" in verdict.justification.lower()


def test_scripted_client_marks_a_requirement_missing_from_the_resume():
    client = ScriptedModelClient()
    prompt = _prompt(
        "Rust systems programming experience",
        "Built backend services in Python for six years.",
    )

    response = client.complete(prompt, ScreeningResponse)

    [verdict] = response.verdicts
    assert verdict.met is False


def test_scripted_client_does_not_match_a_keyword_inside_a_longer_word():
    client = ScriptedModelClient()
    prompt = _prompt(
        "Java experience",
        "Five years of JavaScript and Node.js development.",
    )

    response = client.complete(prompt, ScreeningResponse)

    [verdict] = response.verdicts
    assert verdict.met is False


def test_scripted_client_ignores_a_resume_line_shaped_like_a_requirement():
    client = ScriptedModelClient()
    prompt = _prompt(
        "AWS certification",
        "- (AWS) Certified Solutions Architect\nBuilt backend services in Python.",
    )

    response = client.complete(prompt, ScreeningResponse)

    [verdict] = response.verdicts
    assert verdict.requirement_id == "req-1"


def test_scripted_client_proposes_a_requirement_per_bulleted_line():
    client = ScriptedModelClient()
    job_description = JobDescription(
        text="Backend Engineer\n- 5+ years of Python\n* Bachelor's degree in CS\n"
    )
    prompt = build_extraction_prompt(job_description)

    response = client.complete(prompt, ExtractionResponse)

    assert [r.text for r in response.requirements] == [
        "5+ years of Python",
        "Bachelor's degree in CS",
    ]


def test_scripted_client_ignores_job_description_lines_without_a_bullet():
    client = ScriptedModelClient()
    job_description = JobDescription(
        text="Backend Engineer\nWe are hiring for a growing team.\n- 5+ years of Python\n"
    )
    prompt = build_extraction_prompt(job_description)

    response = client.complete(prompt, ExtractionResponse)

    assert [r.text for r in response.requirements] == ["5+ years of Python"]


def test_scripted_client_does_not_mistake_a_decimal_number_for_a_numbered_bullet():
    client = ScriptedModelClient()
    job_description = JobDescription(
        text="Backend Engineer\n3.5+ years of Python experience preferred\n3. Bachelor's degree\n"
    )
    prompt = build_extraction_prompt(job_description)

    response = client.complete(prompt, ExtractionResponse)

    assert [r.text for r in response.requirements] == ["Bachelor's degree"]


def test_scripted_client_rates_every_ranking_dimension():
    client = ScriptedModelClient()
    role = Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(requirements=(), approved=True),
    )
    prompt = build_ranking_prompt(role, "Built backend services in Python for six years.")

    response = client.complete(prompt, RankingResponse)

    assert {d.dimension for d in response.dimensions} == set(DIMENSIONS)
    assert all(d.justification for d in response.dimensions)


def test_scripted_client_ranking_is_deterministic_given_the_same_prompt():
    client = ScriptedModelClient()
    role = Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(requirements=(), approved=True),
    )
    prompt = build_ranking_prompt(role, "Built backend services in Python for six years.")

    first = client.complete(prompt, RankingResponse)
    second = client.complete(prompt, RankingResponse)

    assert first == second