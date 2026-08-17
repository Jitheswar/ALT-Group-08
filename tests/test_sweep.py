"""Tests for the full corpus sweep (ticket 07), driven through the same
seams as the rest of the suite: the core's public entry point
(run_screening, reached indirectly via sweep_role/run_sweep) and the model
client. These assert that a Shortlist's actual order gets turned into the
right rank metrics against Proxy Relevance, and that parse-failure rate is
tracked per Role and aggregated across a sweep - not that the metric
formulas themselves are correct, which test_metrics.py already covers on
fixed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from screening.domain import Requirement, RequirementSet, Role
from screening.eval_roles import EvaluationRole, PostingProvenance
from screening.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from screening.model_client import (
    FitDimensionResponse,
    FitRating,
    ModelClientError,
    RankingResponse,
    RequirementVerdictResponse,
    RunMetrics,
    ScreeningResponse,
)
from screening.proxy_relevance import CorpusResume
from screening.ranking import DIMENSIONS
from screening.sweep import run_sweep, sweep_role
from tests.fakes import RecordingFakeModelClient, StrengthBasedComparativeFakeModelClient

REQ = Requirement(id="req-1", text="Some Requirement")


def _evaluation_role(role_id: str, category: str) -> EvaluationRole:
    return EvaluationRole(
        evaluation_role_id=role_id,
        category=category,
        role=Role(
            id=role_id,
            title=category,
            requirement_set=RequirementSet(requirements=(REQ,), approved=True),
        ),
        posting=PostingProvenance(
            source_dataset="NextGig-Rocks/global-job-postings-multi-ats",
            posting_id="1",
            title=category,
            company_name="Acme",
            matched_term=category,
        ),
        reviewed_by_human=True,
        generated_at=datetime.now(timezone.utc),
    )


def _qualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[RequirementVerdictResponse(requirement_id="req-1", met=True, justification="ok")]
    )


def _disqualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[RequirementVerdictResponse(requirement_id="req-1", met=False, justification="no")]
    )


def _fit(rating: FitRating = "strong") -> RankingResponse:
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(dimension=dimension, rating=rating, justification="ok")
            for dimension in DIMENSIONS
        ]
    )


@dataclass
class MetricsFakeModelClient:
    """Like RecordingFakeModelClient, but tracks a `.metrics` RunMetrics the
    same shape a real client's would - so parse-failure aggregation can be
    tested without the live provider.
    """

    responses: list[Any]
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def complete(self, prompt: str, response_model: type) -> Any:
        self.metrics.total_attempts += 1
        if not self.responses:
            raise AssertionError("MetricsFakeModelClient has no more scripted responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            self.metrics.parse_failures += 1
            raise response
        return response


def test_sweep_role_scores_the_shortlists_actual_order_against_proxy_relevance():
    corpus = [
        CorpusResume(candidate_id="c1", category="Data Science", text="Backend engineer. STRENGTH:1"),
        CorpusResume(candidate_id="c2", category="Sales", text="Backend engineer. STRENGTH:4"),
        CorpusResume(candidate_id="c3", category="Data Science", text="Backend engineer. STRENGTH:3"),
        CorpusResume(candidate_id="c4", category="Testing", text="Backend engineer. STRENGTH:2"),
    ]
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[_qualified(), _qualified(), _qualified(), _qualified(), _fit(), _fit(), _fit(), _fit()]
    )

    result = sweep_role(evaluation_role, corpus, model_client, k=4)

    # Descending STRENGTH order is c2, c3, c4, c1; only c3 and c1 are in
    # the "Data Science" category, so the Shortlist's Proxy Relevance
    # sequence is [not, relevant, not, relevant].
    expected_relevance = [False, True, False, True]
    assert result.proxy_precision_at_k == precision_at_k(expected_relevance, 4)
    assert result.proxy_ndcg_at_k == ndcg_at_k(expected_relevance, 4)
    assert result.proxy_reciprocal_rank == reciprocal_rank(expected_relevance)
    assert result.evaluation_role_id == "eval-role-01"
    assert result.category == "Data Science"


def test_sweep_role_reports_parse_failure_rate_from_this_roles_calls_only():
    corpus = [
        CorpusResume(candidate_id="c1", category="Data Science", text="Backend engineer."),
        CorpusResume(candidate_id="c2", category="Data Science", text="Backend engineer."),
    ]
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    model_client = MetricsFakeModelClient(
        responses=[
            _qualified(),
            ModelClientError("provider returned malformed output"),
            _fit(),
        ]
    )

    result = sweep_role(evaluation_role, corpus, model_client, k=2)

    # 3 attempts total (2 Screening + 1 Ranking for the sole Qualified
    # Candidate), 1 of which failed to parse.
    assert result.parse_failure_count == 1
    assert result.parse_failure_rate == 1 / 3


def test_run_sweep_averages_rank_metrics_across_every_evaluation_role():
    corpus = [
        CorpusResume(candidate_id="c1", category="Data Science", text="Backend engineer. STRENGTH:1"),
        CorpusResume(candidate_id="c2", category="Data Science", text="Backend engineer. STRENGTH:2"),
    ]
    # Role A's category matches both Resumes, so a perfect ranking (both
    # Qualified and ranked) scores 1.0 on every metric regardless of order.
    role_a = _evaluation_role("eval-role-01", "Data Science")
    # Role B's category matches neither, so every metric is 0.0 regardless
    # of Screening's outcome - both are scripted Disqualified to avoid
    # needing any Ranking or comparative responses.
    role_b = _evaluation_role("eval-role-02", "Finance")
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[
            _qualified(), _qualified(), _fit(), _fit(),
            _disqualified(), _disqualified(),
        ]
    )

    report = run_sweep([role_a, role_b], corpus, model_client, k=2)

    assert len(report.results) == 2
    assert report.mean_proxy_precision_at_k == pytest.approx(0.5)
    assert report.mean_proxy_ndcg_at_k == pytest.approx(0.5)
    assert report.proxy_mrr == pytest.approx(0.5)


def test_run_sweep_aggregates_parse_failure_rate_across_every_evaluation_role():
    corpus = [CorpusResume(candidate_id="c1", category="Data Science", text="Backend engineer.")]
    role = _evaluation_role("eval-role-01", "Data Science")
    model_client = MetricsFakeModelClient(
        responses=[ModelClientError("provider returned malformed output")]
    )

    report = run_sweep([role], corpus, model_client, k=1)

    assert report.parse_failure_count == 1
    assert report.parse_failure_rate == 1.0


def test_run_sweep_requires_at_least_one_evaluation_role():
    with pytest.raises(ValueError):
        run_sweep([], [], RecordingFakeModelClient(responses=[]), k=1)


def test_sweep_role_reports_zeroed_metrics_for_a_client_that_never_mutates_them():
    corpus = [CorpusResume(candidate_id="c1", category="Data Science", text="Backend engineer.")]
    role = _evaluation_role("eval-role-01", "Data Science")
    model_client = RecordingFakeModelClient(responses=[_disqualified()])

    result = sweep_role(role, corpus, model_client, k=1)

    assert result.parse_failure_count == 0
    assert result.parse_failure_rate == 0.0
