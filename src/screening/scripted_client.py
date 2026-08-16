"""A deterministic, network-free ModelClient. Used by the CLI in place of the
real provider, so the full path - extraction and Screening alike - is
exercisable without network access. Sees only the rendered prompt, exactly
as a real client would - it is not given structured access to the Role, the
Job Description, or the Resume.
"""

from __future__ import annotations

import re

from screening.model_client import (
    ExtractionResponse,
    ProposedRequirementResponse,
    RequirementVerdictResponse,
    ScreeningResponse,
    T,
)

_REQUIREMENT_LINE = re.compile(r"^- \(([^)]+)\)\s*(.+)$")
# The numbered-list branch requires the marker not be followed by another
# digit, so a decimal number at the start of an ordinary sentence (e.g.
# "3.5+ years of Python experience") is not mistaken for list item "3.".
_BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)](?!\d))\s*(.+)$")
_KEYWORD = re.compile(r"[A-Za-z0-9+.#]+")
_SNIPPET_RADIUS = 40


class UnsupportedResponseModel(Exception):
    pass


class ScriptedModelClient:
    """Matches each Requirement against the Resume text by keyword search,
    and proposes Requirements from a Job Description by reading its bullet
    or numbered list lines.
    """

    def complete(self, prompt: str, response_model: type[T]) -> T:
        if response_model is ScreeningResponse:
            requirements = _parse_requirements(prompt)
            resume_text = _parse_resume(prompt)
            verdicts = [
                _verdict_for(requirement_id, requirement_text, resume_text)
                for requirement_id, requirement_text in requirements
            ]
            return response_model(verdicts=verdicts)

        if response_model is ExtractionResponse:
            job_description_text = _parse_job_description(prompt)
            return response_model(
                requirements=_propose_requirements_from(job_description_text)
            )

        raise UnsupportedResponseModel(
            f"ScriptedModelClient does not support {response_model!r}"
        )


def _parse_job_description(prompt: str) -> str:
    _, _, job_description_text = prompt.partition("Job Description:\n")
    return job_description_text.strip()


def _propose_requirements_from(job_description_text: str) -> list[ProposedRequirementResponse]:
    return [
        ProposedRequirementResponse(text=match.group(1).strip())
        for line in job_description_text.splitlines()
        if (match := _BULLET_LINE.match(line))
    ]


def _parse_requirements(prompt: str) -> list[tuple[str, str]]:
    requirement_block, _, _ = _requirement_set_section(prompt).partition("\n\n")
    return [
        (match.group(1), match.group(2))
        for line in requirement_block.splitlines()
        if (match := _REQUIREMENT_LINE.match(line))
    ]


def _requirement_set_section(prompt: str) -> str:
    _, _, after_marker = prompt.partition("Requirement Set:\n")
    return after_marker


def _parse_resume(prompt: str) -> str:
    _, _, resume_text = prompt.partition("Resume:\n")
    return resume_text.strip()


def _verdict_for(
    requirement_id: str, requirement_text: str, resume_text: str
) -> RequirementVerdictResponse:
    keywords = [w for w in _KEYWORD.findall(requirement_text) if len(w) > 2]
    matches = [m for kw in keywords if (m := _find_keyword(kw, resume_text)) is not None]
    met = bool(keywords) and len(matches) >= max(1, len(keywords) // 2)

    if met:
        justification = f'Resume cites "{_snippet(resume_text, matches[0])}"'
    else:
        justification = f"Resume does not evidence: {requirement_text}"
    return RequirementVerdictResponse(
        requirement_id=requirement_id, met=met, justification=justification
    )


def _find_keyword(keyword: str, resume_text: str) -> re.Match[str] | None:
    """Match a keyword only where it is not itself part of a longer word, so
    'Java' does not match inside 'JavaScript' and 'Rust' does not match
    inside 'Thrust'.
    """
    pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", re.IGNORECASE
    )
    return pattern.search(resume_text)


def _snippet(resume_text: str, match: re.Match[str]) -> str:
    start = max(0, match.start() - _SNIPPET_RADIUS)
    end = min(len(resume_text), match.end() + _SNIPPET_RADIUS)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(resume_text) else ""
    return f"{prefix}{resume_text[start:end].strip()}{suffix}"
