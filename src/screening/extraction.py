"""Requirement extraction: proposes Requirements from a Job Description
through the model client seam.

Extraction only ever proposes. It returns a plain tuple of Requirement,
never a RequirementSet - there is no path from here into a Requirement Set,
so a hallucinated Requirement cannot gate Screening without passing through
an explicit Recruiter approval (screening.domain.approve_requirement_set),
per ADR-0004.
"""

from __future__ import annotations

from screening.domain import JobDescription, Requirement
from screening.model_client import ExtractionResponse, ModelClient


def build_extraction_prompt(job_description: JobDescription) -> str:
    return (
        "You are extracting hard Requirements from a Job Description.\n"
        "Return each hard Requirement separately, exactly as it would be "
        "checked against a Candidate one at a time.\n"
        "Respond with a JSON object of exactly this shape, and no other "
        "fields:\n"
        '{"requirements": [{"text": "<requirement text>"}, ...]}\n\n'
        f"Job Description:\n{job_description.text}\n"
    )


def extract_requirements(
    job_description: JobDescription, model_client: ModelClient
) -> tuple[Requirement, ...]:
    prompt = build_extraction_prompt(job_description)
    response = model_client.complete(prompt, ExtractionResponse)
    return tuple(
        Requirement(id=f"req-{index}", text=proposed.text)
        for index, proposed in enumerate(response.requirements, start=1)
    )
