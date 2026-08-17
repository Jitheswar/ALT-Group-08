"""The Gold Set's supporting functions (ticket 08) are pure, tested directly
on fixed inputs per the spec's testing decisions - no corpus download
involved, mirroring tests/test_proxy_relevance.py. rank_gold_set_role and
rank_gold_set run the real Screening+Ranking pipeline, so those are driven
through a fake model client instead, mirroring tests/test_sweep.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from screening.domain import Requirement, Role, approve_requirement_set
from screening.eval_roles import EvaluationRole, PostingProvenance
from screening.gold_set import (
    GoldSetLabel,
    compare_gold_set_to_proxy_relevance,
    exclude_gold_set_candidates,
    rank_gold_set,
    rank_gold_set_role,
    read_gold_set,
    resolve_gold_set_path,
    write_gold_set,
)
from screening.model_client import (
    FitDimensionResponse,
    FitRating,
    RankingResponse,
    RequirementVerdictResponse,
    ScreeningResponse,
)
from screening.proxy_relevance import CorpusResume
from screening.ranking import DIMENSIONS
from tests.fakes import StrengthBasedComparativeFakeModelClient


def _evaluation_role(evaluation_role_id: str, category: str) -> EvaluationRole:
    requirement_set = approve_requirement_set((Requirement(id="req-1", text="Some Requirement"),))
    return EvaluationRole(
        evaluation_role_id=evaluation_role_id,
        category=category,
        role=Role(id=evaluation_role_id, title=category, requirement_set=requirement_set),
        posting=PostingProvenance(
            source_dataset="NextGig-Rocks/global-job-postings-multi-ats",
            posting_id="1",
            title=category,
            company_name="Acme",
            matched_term=category,
        ),
        reviewed_by_human=True,
        generated_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )


def _label(
    evaluation_role_id: str,
    candidate_id: str,
    category: str,
    relevant: bool,
    text: str = "...",
) -> GoldSetLabel:
    return GoldSetLabel(
        evaluation_role_id=evaluation_role_id,
        resume=CorpusResume(candidate_id=candidate_id, category=category, text=text),
        relevant=relevant,
        justification="because",
    )


def _qualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[RequirementVerdictResponse(requirement_id="req-1", met=True, justification="ok")]
    )


def _fit(rating: FitRating = "strong") -> RankingResponse:
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(dimension=dimension, rating=rating, justification="ok")
            for dimension in DIMENSIONS
        ]
    )


def test_write_then_read_gold_set_round_trips(tmp_path):
    labels = [
        _label("eval-role-01", "r1", "Data Science", True),
        _label("eval-role-01", "r2", "Sales", False),
    ]

    path = write_gold_set(labels, tmp_path / "gold_set.json")
    loaded = read_gold_set(path)

    assert loaded == labels


def test_write_gold_set_refuses_to_overwrite(tmp_path):
    path = tmp_path / "gold_set.json"
    write_gold_set([_label("eval-role-01", "r1", "Data Science", True)], path)

    with pytest.raises(FileExistsError):
        write_gold_set([_label("eval-role-01", "r1", "Data Science", True)], path)


def test_exclude_gold_set_candidates_removes_only_held_out_resumes():
    corpus = [
        CorpusResume(candidate_id="r1", category="Data Science", text="..."),
        CorpusResume(candidate_id="r2", category="Sales", text="..."),
        CorpusResume(candidate_id="r3", category="Testing", text="..."),
    ]
    gold_set = [_label("eval-role-01", "r1", "Data Science", True)]

    remaining = exclude_gold_set_candidates(corpus, gold_set)

    assert [r.candidate_id for r in remaining] == ["r2", "r3"]


def test_exclude_gold_set_candidates_with_empty_gold_set_returns_full_corpus():
    corpus = [CorpusResume(candidate_id="r1", category="Data Science", text="...")]

    assert exclude_gold_set_candidates(corpus, []) == corpus


def test_resolve_gold_set_path_returns_the_explicit_path_when_it_exists(tmp_path):
    path = tmp_path / "gold_set.json"
    write_gold_set([_label("eval-role-01", "r1", "Data Science", True)], path)

    assert resolve_gold_set_path(path) == path


def test_resolve_gold_set_path_raises_when_an_explicit_path_does_not_exist(tmp_path):
    """A typo'd --gold-set must fail loudly, not be silently treated as
    "no Gold Set to hold out" - that would leak hand-labelled Resumes into
    Proxy Relevance at corpus scale with nothing in the run's output to
    reveal it.
    """
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(FileNotFoundError):
        resolve_gold_set_path(missing)


def test_resolve_gold_set_path_falls_back_to_the_default_when_unspecified(tmp_path):
    default = tmp_path / "default-gold-set.json"
    write_gold_set([_label("eval-role-01", "r1", "Data Science", True)], default)

    assert resolve_gold_set_path(None, default=default) == default


def test_resolve_gold_set_path_returns_none_when_unspecified_and_the_default_is_absent(tmp_path):
    default = tmp_path / "default-gold-set.json"

    assert resolve_gold_set_path(None, default=default) is None


def test_compare_gold_set_to_proxy_relevance_reports_agreement_and_divergence():
    evaluation_roles = [
        _evaluation_role("eval-role-01", "Data Science"),
        _evaluation_role("eval-role-02", "Sales"),
    ]
    gold_set = [
        # Proxy agrees: same category, hand also says relevant.
        _label("eval-role-01", "r1", "Data Science", True),
        # Proxy false negative: different category, hand says relevant anyway.
        _label("eval-role-01", "r2", "Sales", True),
        # Proxy false positive: same category, hand says not relevant.
        _label("eval-role-01", "r3", "Data Science", False),
        # Proxy agrees: different category, hand also says not relevant.
        _label("eval-role-02", "r4", "Testing", False),
    ]

    report = compare_gold_set_to_proxy_relevance(gold_set, evaluation_roles)

    assert report.agreement_rate == pytest.approx(0.5)
    assert report.proxy_false_positive_rate == pytest.approx(0.5)  # 1 of 2 hand-not-relevant
    assert report.proxy_false_negative_rate == pytest.approx(0.5)  # 1 of 2 hand-relevant
    assert len(report.divergences) == 4
    by_candidate = {d.candidate_id: d for d in report.divergences}
    assert by_candidate["r1"].agree is True
    assert by_candidate["r2"].agree is False
    assert by_candidate["r2"].proxy_relevant is False
    assert by_candidate["r3"].agree is False
    assert by_candidate["r3"].proxy_relevant is True


def test_compare_gold_set_to_proxy_relevance_rates_are_zero_not_undefined_with_no_opportunities():
    evaluation_roles = [_evaluation_role("eval-role-01", "Data Science")]
    # Every pair hand-labelled relevant, so there is no hand-not-relevant
    # pair to compute a false-positive rate from.
    gold_set = [
        _label("eval-role-01", "r1", "Data Science", True),
        _label("eval-role-01", "r2", "Data Science", True),
    ]

    report = compare_gold_set_to_proxy_relevance(gold_set, evaluation_roles)

    assert report.proxy_false_positive_rate == 0.0
    assert report.proxy_false_negative_rate == 0.0


def test_compare_gold_set_to_proxy_relevance_requires_at_least_one_label():
    with pytest.raises(ValueError):
        compare_gold_set_to_proxy_relevance([], [])


def test_rank_gold_set_role_scores_the_real_shortlist_order_two_ways():
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    labels = [
        # Same category as the Role: Proxy Relevance says relevant. Hand
        # label disagrees - the exact case a label-agreement rate alone
        # cannot turn into a rank-metric gap the way this can.
        _label("eval-role-01", "r1", "Data Science", False, text="Backend engineer. STRENGTH:1"),
        _label("eval-role-01", "r2", "Sales", True, text="Backend engineer. STRENGTH:4"),
        _label("eval-role-01", "r3", "Data Science", True, text="Backend engineer. STRENGTH:3"),
    ]
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[_qualified(), _qualified(), _qualified(), _fit(), _fit(), _fit()]
    )

    result = rank_gold_set_role(evaluation_role, labels, model_client)

    # Descending STRENGTH order is r2, r3, r1.
    assert result.k == 3
    assert result.hand_reciprocal_rank == pytest.approx(1.0)  # r2 (hand-relevant) ranked first
    assert result.proxy_reciprocal_rank == pytest.approx(0.5)  # r3 (proxy-relevant) ranked 2nd
    assert result.evaluation_role_id == "eval-role-01"


def test_rank_gold_set_role_requires_at_least_one_label():
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    with pytest.raises(ValueError):
        rank_gold_set_role(evaluation_role, [], StrengthBasedComparativeFakeModelClient(responses=[]))


def test_rank_gold_set_covers_every_evaluation_role_grouped_by_id():
    evaluation_roles = [
        _evaluation_role("eval-role-01", "Data Science"),
        _evaluation_role("eval-role-02", "Sales"),
    ]
    gold_set = [
        _label("eval-role-01", "r1", "Data Science", True, text="Engineer. STRENGTH:1"),
        _label("eval-role-01", "r2", "Data Science", True, text="Engineer. STRENGTH:2"),
        _label("eval-role-02", "r3", "Sales", True, text="Salesperson. STRENGTH:1"),
    ]
    # Screening then Ranking, per Role in turn: role 1 has 2 Candidates
    # (qualified, qualified, fit, fit), role 2 has 1 (qualified, fit).
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[_qualified(), _qualified(), _fit(), _fit(), _qualified(), _fit()]
    )

    results = rank_gold_set(gold_set, evaluation_roles, model_client)

    assert [r.evaluation_role_id for r in results] == ["eval-role-01", "eval-role-02"]
    assert [r.k for r in results] == [2, 1]


def test_rank_gold_set_requires_at_least_one_label():
    with pytest.raises(ValueError):
        rank_gold_set([], [], StrengthBasedComparativeFakeModelClient(responses=[]))
