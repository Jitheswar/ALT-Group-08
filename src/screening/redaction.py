"""Deterministic Redaction: strips direct identity signals from a Resume
before it can reach Ranking (ADR-0005) - name, gender markers, graduation
and birth years, nationality, and photo.

Implemented as rule-based pattern matching rather than a model call, so the
boundary is inexpensive, offline, and testable without a provider - a call
to the model client here would put identity-bearing text in front of the
same untrusted inference path Redaction exists to keep it away from.

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


_FLATTENED_LEADING_STOPWORDS = {
    "summary", "professional", "objective", "profile", "resume", "curriculum",
    "vitae", "education", "training", "activities", "honors", "skills",
    "skill", "personal", "details", "contact", "career", "experience",
    "about", "highlights", "core", "accomplishments", "executive",
}
_FLATTENED_LEADING_LOOKAHEAD = 8


def _looks_like_flattened_name_word(word: str) -> bool:
    return (
        word.isalpha() and len(word) >= 2 and word.lower() not in _FLATTENED_LEADING_STOPWORDS
    )


def _flattened_leading_name(text: str) -> str | None:
    """Tier 4: resume-atlas corpus text arrives as a single line, all
    lowercase, with no punctuation - none of the line- and capitalisation-
    based tiers above ever match it, which is why they detected a name on
    0 of the corpus's Resumes despite working on conventionally formatted
    ones. The Candidate's name is still conventionally the first thing on
    the document, sometimes after a short run of section-header words
    ("summary", "professional profile", ...), so this tier skips those,
    then takes exactly the next two tokens as the name if both look like
    name words.

    Capped at two tokens rather than following _NAME_LINE's wider span:
    without capitalisation to mark where a job title starts, a wider span
    risks swallowing "tax accountant" or "bank teller" into the name. A
    three-or-more-word name only has its first two tokens redacted as a
    result - a partial miss, not the total one this tier replaces.
    """
    words = text.split()
    index = 0
    while (
        index < min(len(words), _FLATTENED_LEADING_LOOKAHEAD)
        and words[index].lower() in _FLATTENED_LEADING_STOPWORDS
    ):
        index += 1
    candidate = words[index : index + 2]
    if len(candidate) < 2 or not all(_looks_like_flattened_name_word(w) for w in candidate):
        return None
    if _looks_like_a_job_title(" ".join(candidate)):
        return None
    return " ".join(candidate)


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

    # Tier 4: resume-atlas corpus text (evaluated at production scale by
    # every sweep) has none of the newlines or capitalisation Tiers 1-3
    # depend on, so they never match it - see _flattened_leading_name.
    # Gated on the text being entirely lowercase, the one reliable signal
    # that distinguishes flattened corpus text from ordinary capitalised
    # prose - without it, this tier would also fire on a bare sentence
    # like "She led her team...", mistaking two ordinary lowercase words
    # for a name.
    if any(c.isupper() for c in text):
        return None
    return _flattened_leading_name(text)


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


# A year is redacted only when a birth/graduation keyword falls within this
# many characters of it - roughly the span of "graduated 2014" or "born
# 1990" plus a short connector word or punctuation, not a whole section.
# A per-line scope doesn't hold here: resume-atlas corpus text (evaluated at
# production scale by every sweep) arrives as one single line, so scoping to
# "the line" the keyword is on means the whole document, and every year in
# it - employment history included - would be redacted whenever any
# graduation/birth keyword appears anywhere. A character-radius window
# around each keyword match is the unit that survives both a conventionally
# line-broken Resume and a flattened single-line one.
_YEAR_CONTEXT_RADIUS = 10


def _redact_years(text: str) -> str:
    """Redacts a year only where it sits within _YEAR_CONTEXT_RADIUS
    characters of a birth or graduation/degree keyword - employment-history
    years are left alone, since the Requirement Set can legitimately depend
    on them (e.g. years of experience) and the ticket scopes Redaction to
    graduation and birth years specifically.
    """
    keyword_spans = [
        match.span()
        for match in (*_BIRTH_KEYWORDS.finditer(text), *_GRADUATION_KEYWORDS.finditer(text))
    ]

    def repl(match: re.Match[str]) -> str:
        year_start, year_end = match.span()
        near_keyword = any(
            year_start < keyword_end + _YEAR_CONTEXT_RADIUS
            and year_end > keyword_start - _YEAR_CONTEXT_RADIUS
            for keyword_start, keyword_end in keyword_spans
        )
        return _YEAR_PLACEHOLDER if near_keyword else match.group(0)

    return _YEAR.sub(repl, text)


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
