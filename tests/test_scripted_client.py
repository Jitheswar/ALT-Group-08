"""The scripted client is exercised directly on the prompt shape build_screening_prompt
produces - it is a real ModelClient implementation, not a test double."""

from __future__ import annotations

from screening.core import build_screening_prompt
from screening.domain import Candidate, Requirement, RequirementSet, Resume, Role
from screening.model_client import ScreeningResponse
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
