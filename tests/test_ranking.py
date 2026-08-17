"""Tests through the core's public entry point for rubric-based Ranking and
the comparative pass layered over it: run_screening(role, candidates,
model_client) -> Shortlist, with Ranking wired in after Screening. Per the
testing decisions in the spec, these assert on the Shortlist that comes
back and on the calls the model client received - the seam that makes
ADR-0002 and ADR-0005 testable rather than merely intended.

Whenever two or more Qualified Candidates land in the top band, the
comparative pass makes at least one ComparativeResponse call, so tests
that produce two or more Qualified outcomes need to account for it. A
band of exactly two always makes exactly one comparison, so
RecordingFakeModelClient can script that call with `_comparison()` - but
which of the two entries the sort passes as "a" versus "b" is an
implementation detail, so `_comparison()` is only safe for a test that
doesn't care which candidate wins (e.g. a redaction check, or a failure
path). Any test asserting on the resulting *order* - for a band of two or
more - uses StrengthBasedComparativeFakeModelClient instead: a comparator
that judges by a "STRENGTH:<n>" marker embedded in each Candidate's resume
text, correct regardless of call order or argument order.
"""

from __future__ import annotations

from screening.core import run_screening
from screening.domain import (
    Candidate,
    Disqualified,
    Fit,
    Qualified,
    Requirement,
    RequirementSet,
    Resume,
    Role,
    Unresolved,
)
from screening.model_client import (
    ComparativeResponse,
    ComparativeWinner,
    FitDimensionResponse,
    FitRating,
    ModelClientError,
    RankingResponse,
    RequirementVerdictResponse,
    ScreeningResponse,
)
from screening.ranking import build_comparative_prompt, build_ranking_prompt
from screening.ranking import DIMENSIONS
from tests.fakes import RecordingFakeModelClient, StrengthBasedComparativeFakeModelClient

REQ_PYTHON = Requirement(id="req-python", text="5+ years of Python")


def _role() -> Role:
    return Role(
        id="role-1",
        title="Backend Engineer",
        requirement_set=RequirementSet(requirements=(REQ_PYTHON,), approved=True),
    )


def _qualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python", met=True, justification="Cites 8 years of Python"
            )
        ]
    )


def _disqualified() -> ScreeningResponse:
    return ScreeningResponse(
        verdicts=[
            RequirementVerdictResponse(
                requirement_id="req-python", met=False, justification="No Python listed"
            )
        ]
    )


def _fit(rating: FitRating) -> RankingResponse:
    return RankingResponse(
        dimensions=[
            FitDimensionResponse(
                dimension=dimension, rating=rating, justification=f"Evidences {dimension}"
            )
            for dimension in DIMENSIONS
        ]
    )


def _comparison(winner: ComparativeWinner = "a") -> ComparativeResponse:
    return ComparativeResponse(winner=winner, justification="Stronger overall Fit")


def test_qualified_candidates_are_ordered_by_fit_best_first():
    role = _role()
    # Which of the two candidates a comparator call labels "a" vs "b" is an
    # implementation detail of sorted()'s argument order, not something
    # this test should assume - so it uses the content-based comparative
    # fake and gives bob the higher comparative strength too, agreeing
    # with (rather than fighting) his higher rubric fit.
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nSenior backend engineer. STRENGTH:1")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nJunior backend engineer. STRENGTH:9")),
    ]
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[_qualified(), _qualified(), _fit("minimal"), _fit("exceptional")]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert [entry.candidate_id for entry in shortlist.entries] == ["bob", "alice"]


def test_the_comparative_pass_records_a_citable_justification_for_the_bands_order():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nSenior backend engineer. STRENGTH:1")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nJunior backend engineer. STRENGTH:9")),
    ]
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[_qualified(), _qualified(), _fit("minimal"), _fit("exceptional")]
    )

    shortlist = run_screening(role, candidates, model_client)

    [comparison] = shortlist.comparisons
    assert comparison.winner_id == "bob"
    assert comparison.loser_id == "alice"
    assert comparison.justification


def test_a_failed_comparative_call_records_no_comparison():
    role = _role()
    candidates = [_candidate("a", strength=9), _candidate("b", strength=1)]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified(), _qualified(),
            _fit("exceptional"), _fit("minimal"),
            ModelClientError("provider returned malformed output"),
        ]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert shortlist.comparisons == ()


def test_qualified_candidates_carry_a_structured_per_dimension_fit_not_a_bare_scalar():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    fit = entry.outcome.fit
    assert isinstance(fit, Fit)
    assert len(fit.dimensions) >= 2
    for dimension in fit.dimensions:
        assert dimension.justification


def test_disqualified_and_unresolved_candidates_still_appear_alongside_ranked_qualified():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer.")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nNo Python here.")),
        Candidate(id="cara", resume=Resume(text="Cara Lee\nBackend engineer.")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified(),
            _disqualified(),
            ModelClientError("provider returned malformed output"),
            _fit("strong"),
        ]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert {entry.candidate_id for entry in shortlist.entries} == {"alice", "bob", "cara"}
    outcomes = {entry.candidate_id: entry.outcome for entry in shortlist.entries}
    assert isinstance(outcomes["alice"], Qualified)
    assert isinstance(outcomes["bob"], Disqualified)
    assert isinstance(outcomes["cara"], Unresolved)


def test_no_ranking_call_ever_receives_a_candidate_who_failed_screening():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer.")),
        Candidate(
            id="bob",
            resume=Resume(text="Bob Roe\nNo Python here, distinctively unique bob-only text."),
        ),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), _disqualified(), _fit("strong")]
    )

    run_screening(role, candidates, model_client)

    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]
    assert len(ranking_calls) == 1
    assert "distinctively unique bob-only text" not in ranking_calls[0].prompt


def test_no_ranking_call_ever_receives_an_unredacted_resume():
    role = _role()
    candidates = [
        Candidate(
            id="alice",
            resume=Resume(text="Alice Doe\nBackend engineer with 8 years of Python."),
        )
    ]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    run_screening(role, candidates, model_client)

    ranking_calls = [c for c in model_client.calls if c.response_model is RankingResponse]
    [call] = ranking_calls
    assert "Alice" not in call.prompt
    assert "Doe" not in call.prompt
    assert "[REDACTED-NAME]" in call.prompt


def test_screening_still_receives_the_unredacted_resume():
    role = _role()
    candidates = [
        Candidate(
            id="alice",
            resume=Resume(text="Alice Doe\nBackend engineer with 8 years of Python."),
        )
    ]
    model_client = RecordingFakeModelClient(responses=[_qualified(), _fit("strong")])

    run_screening(role, candidates, model_client)

    screening_calls = [c for c in model_client.calls if c.response_model is ScreeningResponse]
    [call] = screening_calls
    assert "Alice Doe" in call.prompt


def test_a_qualified_candidate_whose_ranking_call_fails_is_not_dropped():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), ModelClientError("provider returned malformed output")]
    )

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    assert entry.outcome.fit is None


def test_a_ranking_response_missing_a_dimension_leaves_the_candidate_unranked_not_dropped():
    role = _role()
    candidates = [Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer."))]
    incomplete = RankingResponse(
        dimensions=[
            FitDimensionResponse(
                dimension=DIMENSIONS[0], rating="strong", justification="Evidences it"
            )
        ]
    )
    model_client = RecordingFakeModelClient(responses=[_qualified(), incomplete])

    shortlist = run_screening(role, candidates, model_client)

    [entry] = shortlist.entries
    assert isinstance(entry.outcome, Qualified)
    assert entry.outcome.fit is None


def _candidate(id_: str, *, strength: int) -> Candidate:
    return Candidate(id=id_, resume=Resume(text=f"Backend engineer. STRENGTH:{strength}"))


def test_comparative_pass_reorders_the_top_band_and_leaves_the_tail_rubric_ordered():
    role = _role()
    # Rubric order is a, b, c, d (by descending fit rating). Comparative
    # strength disagrees with rubric order for the top two only, so a
    # top_band_size of 2 must flip a and b while leaving c, d untouched.
    candidates = [
        _candidate("a", strength=1),
        _candidate("b", strength=9),
        _candidate("c", strength=5),
        _candidate("d", strength=5),
    ]
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=[
            _qualified(), _qualified(), _qualified(), _qualified(),
            _fit("exceptional"), _fit("strong"), _fit("moderate"), _fit("minimal"),
        ]
    )

    shortlist = run_screening(role, candidates, model_client, top_band_size=2)

    assert [entry.candidate_id for entry in shortlist.entries] == ["b", "a", "c", "d"]


def test_comparative_pass_makes_on_the_order_of_n_log_n_calls_over_the_band_not_the_full_batch():
    role = _role()
    candidates = [_candidate(f"c{i}", strength=i) for i in range(8)] + [
        _candidate("d1", strength=0)
    ]
    model_client = StrengthBasedComparativeFakeModelClient(
        responses=(
            [_qualified() for _ in range(8)]
            + [_disqualified()]
            + [_fit("strong") for _ in range(8)]
        )
    )

    run_screening(role, candidates, model_client, top_band_size=6)

    comparative_calls = [c for c in model_client.calls if c.response_model is ComparativeResponse]
    # A comparison sort over a band of 6 makes far fewer calls than
    # pairwise-comparing all 8 Qualified Candidates (28) or the full batch
    # of 9 (36), and is independent of either.
    assert 5 <= len(comparative_calls) <= 15
    for call in comparative_calls:
        assert "STRENGTH:6" not in call.prompt  # c6, outside the band
        assert "STRENGTH:7" not in call.prompt  # c7, outside the band
        assert "d1" not in call.prompt


def test_comparative_pass_ordering_is_stable_given_identical_inputs_and_a_deterministic_double():
    role = _role()
    candidates = [
        _candidate("a", strength=3),
        _candidate("b", strength=9),
        _candidate("c", strength=1),
    ]

    def _run() -> list[str]:
        model_client = StrengthBasedComparativeFakeModelClient(
            responses=[
                _qualified(), _qualified(), _qualified(),
                _fit("strong"), _fit("strong"), _fit("strong"),
            ]
        )
        shortlist = run_screening(role, candidates, model_client)
        return [entry.candidate_id for entry in shortlist.entries]

    assert _run() == _run() == ["b", "a", "c"]


def test_a_failed_comparative_call_leaves_the_pair_in_rubric_order():
    role = _role()
    candidates = [_candidate("a", strength=9), _candidate("b", strength=1)]
    model_client = RecordingFakeModelClient(
        responses=[
            _qualified(), _qualified(),
            _fit("exceptional"), _fit("minimal"),
            ModelClientError("provider returned malformed output"),
        ]
    )

    shortlist = run_screening(role, candidates, model_client)

    assert [entry.candidate_id for entry in shortlist.entries] == ["a", "b"]


def test_comparative_pass_never_receives_an_unredacted_resume():
    role = _role()
    candidates = [
        Candidate(id="alice", resume=Resume(text="Alice Doe\nBackend engineer.")),
        Candidate(id="bob", resume=Resume(text="Bob Roe\nBackend engineer.")),
    ]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), _qualified(), _fit("strong"), _fit("strong"), _comparison()]
    )

    run_screening(role, candidates, model_client)

    [comparative_call] = [c for c in model_client.calls if c.response_model is ComparativeResponse]
    assert "Alice" not in comparative_call.prompt
    assert "Doe" not in comparative_call.prompt
    assert "Bob" not in comparative_call.prompt
    assert "Roe" not in comparative_call.prompt
    assert comparative_call.prompt.count("[REDACTED-NAME]") == 2


def test_band_size_is_configurable_via_a_parameter_not_by_touching_ranking_logic():
    role = _role()
    candidates = [_candidate("a", strength=9), _candidate("b", strength=1)]
    model_client = RecordingFakeModelClient(
        responses=[_qualified(), _qualified(), _fit("exceptional"), _fit("minimal")]
    )

    shortlist = run_screening(role, candidates, model_client, top_band_size=1)

    assert [entry.candidate_id for entry in shortlist.entries] == ["a", "b"]
    assert not any(c.response_model is ComparativeResponse for c in model_client.calls)


def test_ranking_prompt_mentions_json():
    # The live provider rejects response_format={"type": "json_object"}
    # with a 400 unless the prompt itself contains the word "json" - this
    # is a provider-side requirement, not a style preference, so it is
    # pinned here rather than left to be rediscovered against the network.
    assert "json" in build_ranking_prompt(_role(), "anything").lower()


def test_comparative_prompt_mentions_json():
    assert "json" in build_comparative_prompt(_role(), "resume a", "resume b").lower()
