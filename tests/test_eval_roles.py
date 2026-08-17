"""The category-to-posting matcher is a pure function tested directly on
fixed inputs, per the spec's testing decisions - no model client or corpus
download involved. It backs the checked-in Evaluation Roles (ticket 06) and
is the mechanism ADR-0008 calls "matched by script": a fixed keyword table
rather than a similarity score or a model call, so a bad match is legible
by reading CATEGORY_SEARCH_TERMS rather than by trusting an opaque number.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from screening.domain import Requirement, Role, approve_requirement_set
from screening.eval_roles import (
    CATEGORY_SEARCH_TERMS,
    EvaluationRole,
    PostingCandidate,
    PostingProvenance,
    match_all_categories,
    match_category_to_posting,
    read_evaluation_role,
    read_evaluation_roles,
    read_evaluation_roles_metadata,
    score_posting,
    write_evaluation_role,
)

RESUME_ATLAS_CATEGORIES = (
    "Accountant",
    "Advocate",
    "Agriculture",
    "Apparel",
    "Architecture",
    "Arts",
    "Automobile",
    "Aviation",
    "BPO",
    "Banking",
    "Blockchain",
    "Building and Construction",
    "Business Analyst",
    "Civil Engineer",
    "Consultant",
    "Data Science",
    "Database",
    "Designing",
    "DevOps",
    "Digital Media",
    "DotNet Developer",
    "ETL Developer",
    "Education",
    "Electrical Engineering",
    "Finance",
    "Food and Beverages",
    "Health and Fitness",
    "Human Resources",
    "Information Technology",
    "Java Developer",
    "Management",
    "Mechanical Engineer",
    "Network Security Engineer",
    "Operations Manager",
    "PMO",
    "Public Relations",
    "Python Developer",
    "React Developer",
    "SAP Developer",
    "SQL Developer",
    "Sales",
    "Testing",
    "Web Designing",
)


def _posting(
    posting_id: str,
    title: str,
    *,
    description: str = "Real duties and requirements.",
    minimum_qualifications: str = "",
) -> PostingCandidate:
    return PostingCandidate(
        posting_id=posting_id,
        title=title,
        normalized_title=title,
        company_name="Acme Corp",
        job_description=description,
        minimum_qualifications=minimum_qualifications,
    )


def test_every_resume_atlas_category_has_search_terms():
    assert set(CATEGORY_SEARCH_TERMS) == set(RESUME_ATLAS_CATEGORIES)
    assert len(CATEGORY_SEARCH_TERMS) == 43


def test_search_terms_are_never_empty_for_a_category():
    for category, terms in CATEGORY_SEARCH_TERMS.items():
        assert len(terms) > 0, f"{category} has no search terms"


def test_score_posting_matches_case_insensitively():
    posting = _posting("p1", "Senior Java Developer")

    scored = score_posting("Java Developer", posting)

    assert scored is not None
    score, term = scored
    assert term == "Java Developer"


def test_score_posting_prefers_the_more_specific_term():
    specific = _posting("p1", "Staff Accountant")
    generic = _posting("p2", "Accounting Associate")

    specific_scored = score_posting("Accountant", specific)
    generic_scored = score_posting("Accountant", generic)
    assert specific_scored is not None
    assert generic_scored is not None
    specific_score, _ = specific_scored
    generic_score, _ = generic_scored

    assert specific_score > generic_score


def test_score_posting_returns_none_when_no_term_matches():
    posting = _posting("p1", "Warehouse Forklift Operator")

    assert score_posting("Java Developer", posting) is None


def test_match_category_to_posting_picks_the_best_scoring_candidate():
    postings = [
        _posting("p1", "Accounting Associate"),
        _posting("p2", "Staff Accountant"),
        _posting("p3", "Delivery Driver"),
    ]

    match = match_category_to_posting("Accountant", postings)

    assert match is not None
    assert match.posting.posting_id == "p2"
    assert match.matched_term == "Staff Accountant"


def test_match_category_to_posting_skips_postings_with_no_job_description():
    postings = [_posting("p1", "Staff Accountant", description="   ")]

    match = match_category_to_posting("Accountant", postings)

    assert match is None


def test_match_category_to_posting_respects_exclusions():
    postings = [_posting("p1", "Staff Accountant"), _posting("p2", "Accounting Associate")]

    match = match_category_to_posting("Accountant", postings, excluded_posting_ids=frozenset({"p1"}))

    assert match is not None
    assert match.posting.posting_id == "p2"


def test_match_category_to_posting_returns_none_when_nothing_matches():
    postings = [_posting("p1", "Delivery Driver")]

    assert match_category_to_posting("Accountant", postings) is None


def test_match_category_to_posting_prefers_qualification_data_at_equal_specificity():
    thin = _posting("p1", "Staff Accountant")
    substantive = _posting("p2", "Staff Accountant", minimum_qualifications="- CPA required")

    match = match_category_to_posting("Accountant", [thin, substantive])

    assert match is not None
    assert match.posting.posting_id == "p2"


def test_match_category_to_posting_never_prefers_qualification_data_over_specificity():
    generic_with_quals = _posting(
        "p1", "Accounting Associate", minimum_qualifications="- 2 years experience"
    )
    specific_without_quals = _posting("p2", "Staff Accountant")

    match = match_category_to_posting("Accountant", [generic_with_quals, specific_without_quals])

    assert match is not None
    assert match.posting.posting_id == "p2"


def test_match_all_categories_never_reuses_a_posting_across_categories():
    # Only one real posting exists, and it satisfies both categories' terms.
    postings = [_posting("p1", "Senior SAP Consultant and Business Analyst")]

    matches = match_all_categories(["SAP Developer", "Business Analyst"], postings)

    claimed = [m.posting.posting_id for m in matches.values() if m is not None]
    assert len(claimed) == len(set(claimed))


def test_match_all_categories_returns_a_result_for_every_category_even_when_unmatched():
    postings = [_posting("p1", "Staff Accountant")]

    matches = match_all_categories(["Accountant", "Advocate"], postings)

    assert set(matches) == {"Accountant", "Advocate"}
    assert matches["Advocate"] is None


def _evaluation_role(evaluation_role_id: str = "eval-role-01") -> EvaluationRole:
    requirement_set = approve_requirement_set((Requirement(id="req-1", text="5+ years Python"),))
    return EvaluationRole(
        evaluation_role_id=evaluation_role_id,
        category="Python Developer",
        role=Role(id=evaluation_role_id, title="Python Developer", requirement_set=requirement_set),
        posting=PostingProvenance(
            source_dataset="NextGig-Rocks/global-job-postings-multi-ats",
            posting_id="12345",
            title="Senior Python Developer",
            company_name="Acme Corp",
            matched_term="Python Developer",
        ),
        reviewed_by_human=True,
        generated_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def test_write_then_read_evaluation_role_round_trips(tmp_path):
    original = _evaluation_role()

    path = write_evaluation_role(original, tmp_path)
    loaded = read_evaluation_role(path)

    assert loaded == original


def test_write_evaluation_role_refuses_to_overwrite(tmp_path):
    write_evaluation_role(_evaluation_role(), tmp_path)

    with pytest.raises(FileExistsError):
        write_evaluation_role(_evaluation_role(), tmp_path)


def test_read_evaluation_roles_reads_every_checked_in_file(tmp_path):
    write_evaluation_role(_evaluation_role("eval-role-01"), tmp_path)
    write_evaluation_role(_evaluation_role("eval-role-02"), tmp_path)

    roles = read_evaluation_roles(tmp_path)

    assert {r.evaluation_role_id for r in roles} == {"eval-role-01", "eval-role-02"}


def test_read_evaluation_roles_metadata_reads_the_checked_in_file(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "total_evaluation_roles": 43,
                "reviewed_sample_size": 12,
                "source_resume_corpus": "ahmedheakl/resume-atlas",
                "source_postings_corpus": "NextGig-Rocks/global-job-postings-multi-ats",
                "generated_at": "2026-08-16T12:08:50.845260+00:00",
            }
        )
    )

    metadata = read_evaluation_roles_metadata(path)

    assert metadata.total_evaluation_roles == 43
    assert metadata.reviewed_sample_size == 12
    assert metadata.source_resume_corpus == "ahmedheakl/resume-atlas"


def test_read_evaluation_roles_metadata_reads_the_actual_checked_in_metadata():
    """A read against the real, checked-in artifact (not a fixture copy) -
    a regression check that the reader's field names actually match what
    scripts/generate_evaluation_roles.py finalize writes.
    """
    path = Path(__file__).parent.parent / "data" / "evaluation_roles" / "metadata.json"

    metadata = read_evaluation_roles_metadata(path)

    assert metadata.reviewed_sample_size <= metadata.total_evaluation_roles
    assert metadata.reviewed_sample_size >= 10  # ADR-0008: at least 10 of 43
