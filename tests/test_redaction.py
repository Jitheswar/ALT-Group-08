"""Redaction is a pure function tested directly on fixed inputs, per the
spec's testing decisions - no model client involved. The held-out sample
test near the bottom is the verification pass the ticket calls for: a batch
of synthetic Resumes, each carrying a known identity signal, checked that
none of those signals survive redact_resume.
"""

from __future__ import annotations

from screening.domain import Resume
from screening.redaction import redact_resume


def test_a_name_on_its_own_first_line_is_redacted_everywhere_it_recurs():
    resume = Resume(
        text=(
            "Jordan Alvarez\n"
            "Backend Engineer\n\n"
            "Experience\n"
            "- Built payment services in Python.\n\n"
            "References available for Jordan Alvarez upon request."
        )
    )

    redacted = redact_resume(resume)

    assert "Jordan" not in redacted.text
    assert "Alvarez" not in redacted.text
    assert "[REDACTED-NAME]" in redacted.text
    assert "Built payment services in Python" in redacted.text


def test_a_labeled_name_field_is_redacted():
    resume = Resume(text="Name: Priya Natarajan\n\nSummary: Backend engineer.")

    redacted = redact_resume(resume)

    assert "Priya" not in redacted.text
    assert "Natarajan" not in redacted.text


def test_gender_pronouns_and_honorifics_are_redacted():
    resume = Resume(
        text=(
            "Summary\n"
            "Mr. Chen led his team through a major migration; she supported him "
            "throughout the project."
        )
    )

    redacted = redact_resume(resume)

    for token in ["Mr.", " his ", " she ", " him "]:
        assert token not in redacted.text


def test_a_birth_year_is_redacted_but_an_unrelated_year_is_not():
    resume = Resume(
        text=(
            "Experience\n"
            "Senior Engineer, Acme Corp, 2019 - 2023\n\n"
            "Personal\n"
            "Date of birth: 1990\n"
        )
    )

    redacted = redact_resume(resume)

    assert "1990" not in redacted.text
    assert "2019 - 2023" in redacted.text


def test_a_graduation_year_is_redacted_but_employment_years_are_not():
    resume = Resume(
        text=(
            "Education\n"
            "B.Sc. Computer Science, graduated 2014\n\n"
            "Experience\n"
            "Backend Engineer, Acme Corp, 2015 - 2020\n"
        )
    )

    redacted = redact_resume(resume)

    assert "2014" not in redacted.text
    assert "2015 - 2020" in redacted.text


def test_a_nationality_is_redacted():
    resume = Resume(text="Summary\nNigerian software engineer with 8 years of experience.")

    redacted = redact_resume(resume)

    assert "Nigerian" not in redacted.text
    assert "[REDACTED-NATIONALITY]" in redacted.text


def test_a_markdown_photo_is_redacted():
    resume = Resume(text="![headshot](https://example.com/photo.jpg)\n\nBackend Engineer")

    redacted = redact_resume(resume)

    assert "example.com" not in redacted.text
    assert "[REDACTED-PHOTO]" in redacted.text


def test_a_labeled_photo_line_is_redacted():
    resume = Resume(text="Photo: attached separately\n\nBackend Engineer")

    redacted = redact_resume(resume)

    assert "attached separately" not in redacted.text


def test_a_later_labeled_name_field_never_pre_empts_the_candidates_own_leading_name():
    resume = Resume(
        text=(
            "Jordan Alvarez\n"
            "Backend Engineer\n\n"
            "References\n"
            "Name: Morgan Lee\n"
            "Title: Engineering Manager\n"
        )
    )

    redacted = redact_resume(resume)

    assert "Jordan" not in redacted.text
    assert "Alvarez" not in redacted.text


def test_a_leading_job_title_line_is_not_mistaken_for_the_candidates_name():
    resume = Resume(text="Senior Backend Engineer\nJordan Alvarez\n\nExperience\n- 8 years of Python.")

    redacted = redact_resume(resume)

    assert "Jordan" not in redacted.text
    assert "Alvarez" not in redacted.text
    assert "Senior Backend Engineer" in redacted.text


def test_an_unrelated_image_filename_in_body_text_is_not_redacted():
    resume = Resume(
        text=(
            "Taylor Brooks\n"
            "Backend Engineer\n\n"
            "Experience\n"
            "- Built a pipeline to compress texture.png assets for the game engine.\n"
        )
    )

    redacted = redact_resume(resume)

    assert "texture.png" in redacted.text


def test_a_common_word_that_matches_the_candidates_first_name_elsewhere_is_left_alone():
    resume = Resume(
        text=(
            "Will Turner\n"
            "Backend Engineer\n\n"
            "Experience\n"
            "- Will lead the Q3 migration to a new payments platform.\n"
        )
    )

    redacted = redact_resume(resume)

    assert "Will Turner" not in redacted.text
    assert "Will lead the Q3 migration" in redacted.text


def test_redaction_leaves_unrelated_resume_content_untouched():
    resume = Resume(
        text=(
            "Taylor Brooks\n"
            "Backend Engineer\n\n"
            "Experience\n"
            "- 8 years of Python and distributed systems.\n"
            "- Led migration of a payments platform serving 2 million users.\n"
        )
    )

    redacted = redact_resume(resume)

    assert "8 years of Python and distributed systems" in redacted.text
    assert "payments platform serving 2 million users" in redacted.text


# --- Held-out sample: the verification pass ------------------------------
#
# Each fixture below carries one known instance of every signal Redaction is
# required to strip. This is the "verification pass over a held-out sample"
# the ticket calls for: not a generalisation claim about arbitrary Resumes,
# but a check that redact_resume never leaks the identity tokens it was
# actually given.

_HELD_OUT_SAMPLE = [
    {
        "name": "Alex Morgan",
        "nationality": "American",
        "birth_year": "1988",
        "graduation_year": "2010",
        "text": (
            "Alex Morgan\n"
            "American software engineer\n\n"
            "Education\n"
            "B.Sc. Computer Science, graduated 2010\n\n"
            "Personal\n"
            "Date of birth: 1988\n"
        ),
    },
    {
        "name": "Wei Zhang",
        "nationality": "Chinese",
        "birth_year": "1992",
        "graduation_year": "2015",
        "text": (
            "Wei Zhang\n"
            "Chinese national, backend engineer\n\n"
            "Education\n"
            "M.S. Computer Science, class of 2015\n\n"
            "Born 1992 in Shanghai.\n"
        ),
    },
    {
        "name": "Fatima Khan",
        "nationality": "Pakistani",
        "birth_year": "1995",
        "graduation_year": "2018",
        "text": (
            "Fatima Khan\n"
            "Pakistani data scientist\n\n"
            "Education\n"
            "Bachelor's degree, graduated 2018\n\n"
            "DOB: 1995\n"
        ),
    },
    {
        "name": "Liam O'Connor",
        "nationality": "Irish",
        "birth_year": "1985",
        "graduation_year": "2007",
        "text": (
            "Liam O'Connor\n"
            "Irish product manager\n\n"
            "Education\n"
            "MBA, graduated 2007\n\n"
            "Born 1985.\n"
        ),
    },
    {
        "name": "Amara Nwosu",
        "nationality": "Nigerian",
        "birth_year": "1991",
        "graduation_year": "2013",
        "text": (
            "Amara Nwosu\n"
            "Nigerian civil engineer\n\n"
            "Education\n"
            "Bachelor's degree, class of 2013\n\n"
            "Date of birth: 1991\n"
        ),
    },
    {
        "name": "Hiroshi Tanaka",
        "nationality": "Japanese",
        "birth_year": "1980",
        "graduation_year": "2003",
        "text": (
            "Hiroshi Tanaka\n"
            "Japanese mechanical engineer\n\n"
            "Education\n"
            "B.Eng, graduated 2003\n\n"
            "Born 1980.\n"
        ),
    },
    {
        "name": "Sofia Rossi",
        "nationality": "Italian",
        "birth_year": "1997",
        "graduation_year": "2019",
        "text": (
            "Sofia Rossi\n"
            "Italian UX designer\n\n"
            "Education\n"
            "BA Design, graduated 2019\n\n"
            "DOB: 1997\n"
        ),
    },
    {
        "name": "Carlos Mendoza",
        "nationality": "Mexican",
        "birth_year": "1989",
        "graduation_year": "2011",
        "text": (
            "Carlos Mendoza\n"
            "Mexican sales director\n\n"
            "Education\n"
            "Bachelor's degree, graduated 2011\n\n"
            "Born 1989.\n"
        ),
    },
    {
        "name": "Anya Petrova",
        "nationality": "Russian",
        "birth_year": "1993",
        "graduation_year": "2016",
        "text": (
            "Anya Petrova\n"
            "Russian financial analyst\n\n"
            "Education\n"
            "M.A. Economics, class of 2016\n\n"
            "Date of birth: 1993\n"
        ),
    },
    {
        "name": "Grace Osei",
        "nationality": "Ghanaian",
        "birth_year": "1994",
        "graduation_year": "2017",
        "text": (
            "Grace Osei\n"
            "Ghanaian marketing lead\n\n"
            "Education\n"
            "Bachelor's degree, graduated 2017\n\n"
            "DOB: 1994\n"
        ),
    },
]


def test_no_name_year_or_nationality_token_survives_across_the_held_out_sample():
    for fixture in _HELD_OUT_SAMPLE:
        redacted = redact_resume(Resume(text=fixture["text"]))

        for part in fixture["name"].split():
            assert part not in redacted.text, f"name leaked for {fixture['name']!r}"
        assert fixture["nationality"] not in redacted.text, (
            f"nationality leaked for {fixture['name']!r}"
        )
        assert fixture["birth_year"] not in redacted.text, (
            f"birth year leaked for {fixture['name']!r}"
        )
        assert fixture["graduation_year"] not in redacted.text, (
            f"graduation year leaked for {fixture['name']!r}"
        )
