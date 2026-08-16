"""Deterministic Redaction: strips direct identity signals from a Resume
before it can reach Ranking (ADR-0005) - name, gender markers, graduation
and birth years, nationality, and photo.

Implemented as rule-based named-entity recognition rather than a model
call, so the boundary is inexpensive, offline, and testable without a
provider - a call to the model client here would put identity-bearing text
in front of the same untrusted inference path Redaction exists to keep it
away from.

Screening still sees the unredacted Resume: screening.core.build_screening_prompt
reads candidate.resume.text directly. Only screening.ranking.rank_shortlist
routes a Resume through redact_resume before it reaches the model client.
"""

from __future__ import annotations

import re

from screening.domain import Resume

_NAME_PLACEHOLDER = "[REDACTED-NAME]"
_GENDER_PLACEHOLDER = "[REDACTED-GENDER]"
_YEAR_PLACEHOLDER = "[REDACTED-YEAR]"
_NATIONALITY_PLACEHOLDER = "[REDACTED-NATIONALITY]"
_PHOTO_PLACEHOLDER = "[REDACTED-PHOTO]"

_LABELED_NAME = re.compile(r"(?im)^(?:full\s+name|name)\s*:\s*(.+)$")
_NAME_INTRO = re.compile(
    r"(?i)\b(?:my name is|i am|this is)\s+"
    r"([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){0,2})"
)
_NAME_LINE = re.compile(r"^[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3}$")
_NAME_HEADER_DENYLIST = {
    "CURRICULUM VITAE",
    "PROFESSIONAL SUMMARY",
    "CONTACT INFORMATION",
    "PERSONAL DETAILS",
    "WORK EXPERIENCE",
    "PROFESSIONAL EXPERIENCE",
    "CAREER OBJECTIVE",
    "TECHNICAL SKILLS",
    "SKILLS SUMMARY",
    "EDUCATION HISTORY",
}
_NAME_HEADER_LOOKAHEAD = 5
_NAME_LEADING_LOOKAHEAD = 2
_JOB_TITLE_WORDS = {
    "ENGINEER", "ENGINEERING", "MANAGER", "DEVELOPER", "ANALYST", "DIRECTOR",
    "SPECIALIST", "CONSULTANT", "DESIGNER", "ARCHITECT", "ADMINISTRATOR",
    "COORDINATOR", "EXECUTIVE", "OFFICER", "LEAD", "SENIOR", "JUNIOR",
    "ASSOCIATE", "INTERN", "SCIENTIST", "PRODUCT", "PROJECT", "TECHNICAL",
    "SOFTWARE", "BACKEND", "FRONTEND", "FULLSTACK", "PRINCIPAL", "STAFF",
    "HEAD", "CHIEF", "PRESIDENT", "SUPERVISOR", "REPRESENTATIVE",
}


def _looks_like_a_job_title(stripped: str) -> bool:
    return bool({w.upper() for w in stripped.split()} & _JOB_TITLE_WORDS)


def _plausible_name_line(stripped: str) -> bool:
    return (
        stripped.upper() not in _NAME_HEADER_DENYLIST
        and bool(_NAME_LINE.match(stripped))
        and not _looks_like_a_job_title(stripped)
    )


def _leading_content_lines(text: str, limit: int) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        if len(lines) >= limit:
            break
    return lines


def _candidate_name(text: str) -> str | None:
    # Tier 1: a Resume is the Candidate's own account of themselves
    # (CONTEXT.md), and convention puts the name on one of the first
    # couple of non-blank lines. This is checked before any label or
    # self-introduction search below, so a "Name:" field elsewhere in the
    # document - e.g. inside a References section - can never pre-empt
    # the Candidate's own leading name.
    for stripped in _leading_content_lines(text, _NAME_LEADING_LOOKAHEAD):
        if _plausible_name_line(stripped):
            return stripped

    # Tier 2: an explicit label or self-introduction, wherever it appears -
    # covers formats that do not lead with a bare name line.
    labeled = _LABELED_NAME.search(text)
    if labeled:
        return labeled.group(1).strip()
    intro = _NAME_INTRO.search(text)
    if intro:
        return intro.group(1).strip()

    # Tier 3: a wider but weaker scan, for formats with a section header
    # (e.g. "Curriculum Vitae") before the name.
    for stripped in _leading_content_lines(text, _NAME_HEADER_LOOKAHEAD):
        if _plausible_name_line(stripped):
            return stripped
    return None


def _redact_name(text: str) -> str:
    """Redacts only the exact name string as found - not its individual
    parts - so a first name that also happens to be an ordinary word
    elsewhere in the Resume (e.g. "Grace", "Will", "May") is not stripped
    out of unrelated content it does not identify.
    """
    name = _candidate_name(text)
    if not name:
        return text
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])")
    return pattern.sub(_NAME_PLACEHOLDER, text)


_GENDER_PRONOUNS = re.compile(r"(?i)\b(?:he|him|his|himself|she|her|hers|herself)\b")
_GENDER_HONORIFICS = re.compile(r"(?i)\b(?:mr|mrs|ms|miss|mister|sir|madam)\.?\b")
_GENDER_WORDS = re.compile(r"(?i)\b(?:male|female|man|woman|gentleman|lady)\b")


def _redact_gender_markers(text: str) -> str:
    text = _GENDER_PRONOUNS.sub(_GENDER_PLACEHOLDER, text)
    text = _GENDER_HONORIFICS.sub(_GENDER_PLACEHOLDER, text)
    text = _GENDER_WORDS.sub(_GENDER_PLACEHOLDER, text)
    return text


_BIRTH_KEYWORDS = re.compile(r"(?i)\b(?:born|date of birth|d\.?o\.?b\.?|birth year)\b")
_GRADUATION_KEYWORDS = re.compile(
    r"(?i)\b(?:graduat\w*|class of|b\.?s\.?c?\.?|b\.?a\.?|m\.?s\.?c?\.?|m\.?a\.?|ph\.?d\.?|"
    r"bachelor'?s?|master'?s?|doctorate)\b"
)
_YEAR = re.compile(r"\b(?:18|19|20)\d{2}\b")


def _redact_years(text: str) -> str:
    """Only years on a line carrying a birth or graduation/degree keyword
    are redacted - employment-history years are left alone, since the
    Requirement Set can legitimately depend on them (e.g. years of
    experience) and the ticket scopes Redaction to graduation and birth
    years specifically.
    """
    lines = text.splitlines()
    redacted = [
        _YEAR.sub(_YEAR_PLACEHOLDER, line)
        if _BIRTH_KEYWORDS.search(line) or _GRADUATION_KEYWORDS.search(line)
        else line
        for line in lines
    ]
    return "\n".join(redacted)


_NATIONALITIES = (
    "American", "British", "English", "Scottish", "Welsh", "Irish", "French",
    "German", "Italian", "Spanish", "Portuguese", "Dutch", "Belgian", "Swiss",
    "Austrian", "Polish", "Russian", "Ukrainian", "Swedish", "Norwegian",
    "Danish", "Finnish", "Greek", "Turkish", "Chinese", "Japanese", "Korean",
    "Indian", "Pakistani", "Bangladeshi", "Nepali", "Vietnamese", "Thai",
    "Filipino", "Indonesian", "Malaysian", "Singaporean", "Australian",
    "Canadian", "Mexican", "Brazilian", "Argentinian", "Chilean",
    "Colombian", "Peruvian", "Nigerian", "Kenyan", "Ghanaian",
    "South African", "Egyptian", "Moroccan", "Israeli", "Saudi", "Emirati",
    "Iranian", "Iraqi",
)
_NATIONALITY = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(n) for n in _NATIONALITIES) + r")\b"
)


def _redact_nationality(text: str) -> str:
    return _NATIONALITY.sub(_NATIONALITY_PLACEHOLDER, text)


_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_DATA_URI = re.compile(r"data:image/[^\s)]+")
_PHOTO_LABEL_LINE = re.compile(r"(?im)^(?:photo|headshot|picture|profile photo)\s*:.*$")


def _redact_photo(text: str) -> str:
    # No bare-filename heuristic (e.g. matching any "*.png" token): a
    # Resume can legitimately mention an image file in unrelated content
    # (a project screenshot, a texture asset), and that is not a photo of
    # the Candidate. Only an explicit image embed, a labeled photo line, or
    # a data URI is treated as one.
    text = _MARKDOWN_IMAGE.sub(_PHOTO_PLACEHOLDER, text)
    text = _DATA_URI.sub(_PHOTO_PLACEHOLDER, text)
    text = _PHOTO_LABEL_LINE.sub(_PHOTO_PLACEHOLDER, text)
    return text


def detect_candidate_name(text: str) -> str | None:
    """The exact name string redact_resume would strip, if any - the same
    detection redact_resume uses internally, exposed so
    screening.counterfactual can alter precisely the signal Redaction is
    responsible for removing (ADR-0005).
    """
    return _candidate_name(text)


def detect_nationality(text: str) -> str | None:
    """The first nationality token redact_resume would strip, if any."""
    match = _NATIONALITY.search(text)
    return match.group(0) if match else None


def redact_resume(resume: Resume) -> Resume:
    """Strips name, gender markers, graduation/birth years, nationality,
    and photo from a Resume, deterministically. Photo is redacted first so
    a markdown image line at the top of the Resume can never be mistaken
    for the Candidate's name.
    """
    text = resume.text
    text = _redact_photo(text)
    text = _redact_name(text)
    text = _redact_years(text)
    text = _redact_nationality(text)
    text = _redact_gender_markers(text)
    return Resume(text=text)
