"""Counterfactual Sensitivity (ticket 09, ADR-0005). Pair generation and the
leak check are pure, tested directly on fixed inputs per the spec's testing
decisions. Movement measurement goes through the same seam as the rest of
the suite - screening.ranking.judge_fit, reached via scripted/fake model
clients - mirroring tests/test_sweep.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from screening.counterfactual import (
    CounterfactualPair,
    generate_counterfactual_pairs,
    measure_counterfactual_pair,
    measure_counterfactual_sensitivity_for_role,
    redaction_leaked,
    run_counterfactual_sensitivity,
)
from screening.domain import Requirement, Resume, Role, approve_requirement_set
from screening.eval_roles import EvaluationRole, PostingProvenance
from screening.model_client import (
    FitDimensionResponse,
    FitRating,
    ModelClientError,
    RankingResponse,
)
from screening.proxy_relevance import CorpusResume
from screening.ranking import DIMENSIONS
from tests.fakes import RecordingFakeModelClient

_GOLD_SET_PATH = Path(__file__).parent.parent / "data" / "gold_set" / "gold_set.json"


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


def _fit(rating: FitRating) -> RankingResponse:
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(dimension=dimension, rating=rating, justification="ok")
            for dimension in DIMENSIONS
        ]
    )


# --- Pair generation: pure, fixed inputs ----------------------------------


def test_a_name_pair_alters_only_the_name_and_holds_the_rest_constant():
    resume = Resume(
        text=(
            "Jordan Alvarez\n"
            "Backend Engineer\n\n"
            "Experience\n"
            "- Built payment services in Python.\n\n"
            "References available for Jordan Alvarez upon request."
        )
    )

    [pair] = generate_counterfactual_pairs(resume)

    assert pair.signal == "name"
    assert pair.original_value == "Jordan Alvarez"
    assert "Jordan" not in pair.altered.text
    assert "Alvarez" not in pair.altered.text
    assert pair.altered_value in pair.altered.text
    assert "Built payment services in Python" in pair.altered.text
    assert "Backend Engineer" in pair.altered.text


def test_a_gender_pair_swaps_every_pronoun_and_holds_the_rest_constant():
    resume = Resume(
        text="Summary\nMr. Chen led his team; she supported him throughout the project."
    )

    [pair] = generate_counterfactual_pairs(resume)

    assert pair.signal == "gender"
    assert "Ms." in pair.altered.text
    assert " her " in pair.altered.text
    assert " him " not in pair.altered.text
    assert "led" in pair.altered.text and "team" in pair.altered.text


def test_a_gender_pair_keeps_possessive_pronouns_grammatical():
    # "her" as a possessive determiner ("her team") must swap to "his",
    # not to the object-pronoun form "him" - an ungrammatical swap would
    # be a confound for the Fit-movement measurement itself.
    resume = Resume(text="She led her team and grew her staff of 8.")

    [pair] = generate_counterfactual_pairs(resume)

    assert pair.altered.text == "He led his team and grew his staff of 8."


def test_a_nationality_pair_swaps_the_nationality_and_holds_the_rest_constant():
    resume = Resume(text="Summary\nNigerian software engineer with 8 years of experience.")

    [pair] = generate_counterfactual_pairs(resume)

    assert pair.signal == "nationality"
    assert "Nigerian" not in pair.altered.text
    assert pair.altered_value in pair.altered.text
    assert "8 years of experience" in pair.altered.text


def test_a_resume_with_no_detectable_identity_signal_yields_no_pairs():
    resume = Resume(text="Experience\n- Built payment services in Python.\n")

    assert generate_counterfactual_pairs(resume) == []


def test_a_resume_with_several_signals_yields_one_pair_per_signal():
    resume = Resume(
        text=(
            "Amara Nwosu\n"
            "Nigerian backend engineer\n\n"
            "Experience\n"
            "- She led a migration to a new payments platform.\n"
        )
    )

    pairs = generate_counterfactual_pairs(resume)

    assert {pair.signal for pair in pairs} == {"name", "gender", "nationality"}
    for pair in pairs:
        assert "led a migration to a new payments platform" in pair.altered.text


def test_name_pairs_are_actually_generated_on_a_held_out_sample_of_the_real_corpus():
    """Ticket 09's acceptance criterion is that the name signal - the
    primary identity signal - is measured, not just generated in principle.
    Every fixture above uses a conventionally capitalised Resume; this
    checks the real resume-atlas corpus (via the checked-in Gold Set,
    ADR-0008) actually produces name pairs at production scale, the same
    gap that left detect_candidate_name at 0% before the redaction fix
    (see tests/test_redaction.py's held-out sample).
    """
    rows = json.loads(_GOLD_SET_PATH.read_text())
    sample = rows[:100]

    name_pair_count = sum(
        1
        for row in sample
        for pair in generate_counterfactual_pairs(Resume(text=row["resume"]["text"]))
        if pair.signal == "name"
    )

    assert name_pair_count / len(sample) >= 0.85, (
        f"only {name_pair_count}/{len(sample)} sampled Resumes yielded a name pair"
    )


# --- Leak check: pure ------------------------------------------------------


def test_redaction_leaked_is_false_for_identical_redacted_text():
    assert redaction_leaked("same text", "same text") is False


def test_redaction_leaked_is_true_for_differing_redacted_text():
    assert redaction_leaked("[REDACTED-NAME] engineer", "Priya engineer") is True


# --- Movement measurement: through the real judge_fit seam ----------------


def test_measure_counterfactual_pair_reports_fit_weight_and_dimension_movement():
    role = _evaluation_role("eval-role-01", "Data Science").role
    pair = CounterfactualPair(
        signal="name",
        original=Resume(text="Jordan Alvarez\nBackend engineer."),
        altered=Resume(text="Priya Natarajan\nBackend engineer."),
        original_value="Jordan Alvarez",
        altered_value="Priya Natarajan",
    )
    model_client = RecordingFakeModelClient(responses=[_fit("minimal"), _fit("exceptional")])

    measurement = measure_counterfactual_pair(
        "eval-role-01", "c1", role, pair, model_client
    )

    assert measurement.resolved is True
    assert measurement.redaction_leaked is False
    assert measurement.fit_weight_movement == 3 * 3  # every dimension moved 3 rating-levels
    assert measurement.dimensions_moved == len(DIMENSIONS)


def test_measure_counterfactual_pair_is_unresolved_when_either_side_fails():
    role = _evaluation_role("eval-role-01", "Data Science").role
    pair = CounterfactualPair(
        signal="name",
        original=Resume(text="Jordan Alvarez\nBackend engineer."),
        altered=Resume(text="Priya Natarajan\nBackend engineer."),
        original_value="Jordan Alvarez",
        altered_value="Priya Natarajan",
    )
    model_client = RecordingFakeModelClient(
        responses=[_fit("strong"), ModelClientError("provider returned malformed output")]
    )

    measurement = measure_counterfactual_pair(
        "eval-role-01", "c1", role, pair, model_client
    )

    assert measurement.resolved is False
    assert measurement.fit_weight_movement is None
    assert measurement.dimensions_moved is None


def test_measure_counterfactual_pair_detects_no_leak_when_redaction_strips_both_names():
    role = _evaluation_role("eval-role-01", "Data Science").role
    [pair] = generate_counterfactual_pairs(
        Resume(text="Jordan Alvarez\nBackend engineer with 8 years of Python.")
    )
    model_client = RecordingFakeModelClient(responses=[_fit("strong"), _fit("strong")])

    measurement = measure_counterfactual_pair("eval-role-01", "c1", role, pair, model_client)

    # Real redact_resume strips both names to the same placeholder, so a
    # working Redaction reports no leak even though the names differed.
    assert measurement.redaction_leaked is False
    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]
    assert len(ranking_calls) == 2
    assert "Jordan" not in ranking_calls[0].prompt
    assert pair.altered_value not in ranking_calls[1].prompt
    assert all("[REDACTED-NAME]" in c.prompt for c in ranking_calls)


# --- Aggregation across Evaluation Roles -----------------------------------


def test_measure_counterfactual_sensitivity_for_role_aggregates_across_sampled_candidates():
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    corpus = [
        CorpusResume(candidate_id="c1", category="Data Science", text="Jordan Alvarez\nEngineer."),
        CorpusResume(candidate_id="c2", category="Data Science", text="Priya Natarajan\nEngineer."),
        CorpusResume(candidate_id="c3", category="Sales", text="Alex Kim\nSales lead."),
    ]
    # Two sampled Candidates (only the "Data Science" ones are Proxy
    # Relevant), one name pair each: minimal->exceptional then
    # strong->strong.
    model_client = RecordingFakeModelClient(
        responses=[_fit("minimal"), _fit("exceptional"), _fit("strong"), _fit("strong")]
    )

    result = measure_counterfactual_sensitivity_for_role(
        evaluation_role, corpus, model_client, sample_size=5
    )

    assert result.evaluation_role_id == "eval-role-01"
    assert len(result.measurements) == 2
    assert result.mean_fit_weight_movement == pytest.approx((9 + 0) / 2)
    assert result.max_fit_weight_movement == 9
    assert result.redaction_leak_count == 0
    assert result.redaction_leak_rate == 0.0


def test_measure_counterfactual_sensitivity_for_role_respects_sample_size():
    evaluation_role = _evaluation_role("eval-role-01", "Data Science")
    names = ["Ada Lovelace", "Grace Hopper", "Rosalind Franklin", "Katherine Johnson", "Marie Curie"]
    corpus = [
        CorpusResume(candidate_id=f"c{i}", category="Data Science", text=f"{name}\nEngineer.")
        for i, name in enumerate(names)
    ]
    model_client = RecordingFakeModelClient(responses=[_fit("strong")] * 4)

    result = measure_counterfactual_sensitivity_for_role(
        evaluation_role, corpus, model_client, sample_size=2
    )

    assert len(result.measurements) == 2


def test_run_counterfactual_sensitivity_covers_every_evaluation_role_not_one_ad_hoc_example():
    role_a = _evaluation_role("eval-role-01", "Data Science")
    role_b = _evaluation_role("eval-role-02", "Sales")
    corpus = [
        CorpusResume(candidate_id="c1", category="Data Science", text="Jordan Alvarez\nEngineer."),
        CorpusResume(candidate_id="c2", category="Sales", text="Priya Natarajan\nSales lead."),
    ]
    model_client = RecordingFakeModelClient(responses=[_fit("strong")] * 4)

    report = run_counterfactual_sensitivity(
        [role_a, role_b], corpus, model_client, sample_size=5
    )

    assert {result.evaluation_role_id for result in report.results} == {
        "eval-role-01",
        "eval-role-02",
    }
    assert report.mean_fit_weight_movement == 0.0
    assert report.redaction_leak_rate == 0.0


def test_run_counterfactual_sensitivity_requires_at_least_one_evaluation_role():
    with pytest.raises(ValueError):
        run_counterfactual_sensitivity([], [], RecordingFakeModelClient(responses=[]))
