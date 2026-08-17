"""Counterfactual Sensitivity (ticket 09, ADR-0005): alters exactly one
identity signal in a Resume, holds everything else constant, and measures
how far the resulting Fit judgement moves when both variants are judged
through screening.ranking.judge_fit - the exact function real Ranking
calls go through, not a re-implementation of it.

Because both variants are redacted by the same redact_resume call
judge_fit always makes, this doubles as the check ADR-0005 asks for on
whether Redaction is leaking: if Redaction is doing its job, altering an
identity signal upstream of it should leave the pair's Redacted Resume
identical, and any Fit movement is then evidence of the model inferring
identity from context that survives redaction rather than a literal
token leak. `redaction_leaked` below tells the two apart directly, from
the Redacted Resume text itself, independent of whatever the model went
on to judge.

A high Counterfactual Sensitivity number is the finding this module
exists to surface, not a defect to average away or gate behind a
threshold - there is no pass/fail anywhere here, only a measurement.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from screening.domain import Resume, Role
from screening.eval_roles import EvaluationRole
from screening.model_client import ModelClient
from screening.proxy_relevance import CorpusResume, construct_proxy_relevance
from screening.ranking import fit_weight, judge_fit
from screening.redaction import detect_candidate_name, detect_nationality

# A deliberately diverse pool, not tied to any one axis of identity - a
# generator picks the first entry that differs from the Resume's own
# detected value, so which alternate lands is a function of the input
# rather than a hand-picked pairing.
_ALTERNATE_NAMES: tuple[str, ...] = (
    "Emily Carter", "Lakisha Washington", "Wei Chen", "Fatima Ahmed",
    "Liam O'Brien", "Amara Nwosu", "Hiroshi Tanaka", "Sofia Rossi",
    "Carlos Mendoza", "Anya Petrova", "Grace Osei", "Mohammed Al-Farsi",
)

_ALTERNATE_NATIONALITIES: tuple[str, ...] = (
    "American", "Nigerian", "Indian", "Chinese", "Brazilian", "French",
    "Japanese", "Egyptian", "Mexican", "Russian",
)

# "her" is grammatically ambiguous in English (object pronoun - "helped
# her" - and possessive determiner - "her team") and a flat word swap
# cannot tell those apart. Resume prose overwhelmingly uses the
# possessive-determiner form ("led his/her team", "grew his/her
# portfolio"), so "her" is mapped as that case: paired with "his" rather
# than with "him". Getting this wrong would corrupt the resulting Resume's
# grammar, which is a confound for the very Fit-movement measurement this
# module produces - a rating could shift because the text now reads as
# broken English, not because of the gender signal alone.
_GENDER_SWAP: dict[str, str] = {
    "he": "she", "she": "he",
    "his": "her", "her": "his",
    "him": "her", "hers": "his",
    "himself": "herself", "herself": "himself",
    "mr": "ms", "ms": "mr",
    "man": "woman", "woman": "man",
    "gentleman": "lady", "lady": "gentleman",
}
_GENDER_TOKEN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in _GENDER_SWAP) + r")\.?\b", re.IGNORECASE
)


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def _pick_alternate(current: str, pool: Sequence[str]) -> str:
    for candidate in pool:
        if candidate.lower() != current.lower():
            return candidate
    raise ValueError(f"no alternate in {pool!r} distinct from {current!r}")


def _match_source_case(source: str, target: str) -> str:
    """Matches `target`'s case to `source`'s. resume-atlas corpus text is
    entirely lowercase, with no capitalisation to mark a name - exactly the
    signal screening.redaction._candidate_name's Tier 4 keys off to detect
    it there. Substituting a Title Case alternate (e.g. "Emily Carter" from
    _ALTERNATE_NAMES) into that lowercase text would flip the altered
    Resume out of Tier 4's lowercase gate, so Redaction would strip the
    original's name but not the altered one's - a case difference alone
    manufacturing a Redaction "leak" that was never a token leak at all.
    """
    if source.islower():
        return target.lower()
    if source.isupper():
        return target.upper()
    return target


def _exact_token_sub(text: str, value: str, replacement: str) -> str:
    """Substitutes every occurrence of the exact string `value`, not
    occurrences of its individual words - mirrors
    screening.redaction._redact_name, so a swapped name does not also
    clobber an unrelated word elsewhere in the Resume that happens to
    share part of it.
    """
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(value)}(?![A-Za-z])")
    return pattern.sub(replacement, text)


@dataclass(frozen=True)
class CounterfactualPair:
    """One counterfactual: `altered` differs from `original` in exactly
    the one identity signal named by `signal`, and nothing else.
    """

    signal: str
    original: Resume
    altered: Resume
    original_value: str
    altered_value: str


def _detected_value_pair(
    resume: Resume,
    *,
    signal: str,
    detect: Callable[[str], str | None],
    pool: Sequence[str],
) -> CounterfactualPair | None:
    """Shared by `_name_pair` and `_nationality_pair`, which differ only in
    which signal they detect and which alternate pool they swap in from.
    """
    value = detect(resume.text)
    if not value:
        return None
    alternate = _match_source_case(value, _pick_alternate(value, pool))
    altered_text = _exact_token_sub(resume.text, value, alternate)
    return CounterfactualPair(
        signal=signal,
        original=resume,
        altered=Resume(text=altered_text),
        original_value=value,
        altered_value=alternate,
    )


def _name_pair(resume: Resume) -> CounterfactualPair | None:
    return _detected_value_pair(
        resume, signal="name", detect=detect_candidate_name, pool=_ALTERNATE_NAMES
    )


def _nationality_pair(resume: Resume) -> CounterfactualPair | None:
    return _detected_value_pair(
        resume, signal="nationality", detect=detect_nationality, pool=_ALTERNATE_NATIONALITIES
    )


def _gender_pair(resume: Resume) -> CounterfactualPair | None:
    if not _GENDER_TOKEN.search(resume.text):
        return None

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        bare = token.rstrip(".")
        mapped = _GENDER_SWAP[bare.lower()]
        return _match_case(bare, mapped) + ("." if token.endswith(".") else "")

    altered_text = _GENDER_TOKEN.sub(repl, resume.text)
    return CounterfactualPair(
        signal="gender",
        original=resume,
        altered=Resume(text=altered_text),
        original_value="detected gender marker(s)",
        altered_value="swapped gender marker(s)",
    )


def generate_counterfactual_pairs(resume: Resume) -> list[CounterfactualPair]:
    """Every counterfactual pair generate-able from `resume`: one per
    identity signal actually present in it. A Resume carrying no
    detectable identity signal yields an empty list, rather than a pair
    that would alter nothing.
    """
    pairs = (_name_pair(resume), _gender_pair(resume), _nationality_pair(resume))
    return [pair for pair in pairs if pair is not None]


def redaction_leaked(original_redacted_text: str, altered_redacted_text: str) -> bool:
    """Whether Redaction actually removed the altered identity signal: if
    it did, the pair's two Redacted Resumes are identical regardless of
    which value the signal held beforehand, so any difference here is a
    literal token surviving Redaction (ADR-0005) - the "real check on
    whether redaction is leaking" the ticket calls for, independent of
    whatever the model went on to judge.
    """
    return original_redacted_text != altered_redacted_text


@dataclass(frozen=True)
class CounterfactualMeasurement:
    """One counterfactual pair's outcome, for one Evaluation Role and one
    sampled Candidate: whether Redaction leaked the altered signal, and
    how far the resulting Fit judgement moved. `resolved` is False when
    either side's Ranking call failed to produce a usable Fit, in which
    case `fit_weight_movement` and `dimensions_moved` are None rather
    than a misleading zero.
    """

    evaluation_role_id: str
    candidate_id: str
    signal: str
    redaction_leaked: bool
    resolved: bool
    fit_weight_movement: int | None
    dimensions_moved: int | None


def measure_counterfactual_pair(
    evaluation_role_id: str,
    candidate_id: str,
    role: Role,
    pair: CounterfactualPair,
    model_client: ModelClient,
) -> CounterfactualMeasurement:
    """Judges both sides of `pair` through screening.ranking.judge_fit -
    the same redact-then-rank path every real Ranking call goes through -
    and measures how far the resulting Fit judgement moved.
    """
    original_judgement = judge_fit(role, pair.original, model_client)
    altered_judgement = judge_fit(role, pair.altered, model_client)
    leaked = redaction_leaked(original_judgement.redacted_text, altered_judgement.redacted_text)

    if original_judgement.fit is None or altered_judgement.fit is None:
        return CounterfactualMeasurement(
            evaluation_role_id=evaluation_role_id,
            candidate_id=candidate_id,
            signal=pair.signal,
            redaction_leaked=leaked,
            resolved=False,
            fit_weight_movement=None,
            dimensions_moved=None,
        )

    movement = abs(fit_weight(altered_judgement.fit) - fit_weight(original_judgement.fit))
    dimensions_moved = sum(
        1
        for original_dimension, altered_dimension in zip(
            original_judgement.fit.dimensions, altered_judgement.fit.dimensions
        )
        if original_dimension.rating != altered_dimension.rating
    )
    return CounterfactualMeasurement(
        evaluation_role_id=evaluation_role_id,
        candidate_id=candidate_id,
        signal=pair.signal,
        redaction_leaked=leaked,
        resolved=True,
        fit_weight_movement=movement,
        dimensions_moved=dimensions_moved,
    )


@dataclass(frozen=True)
class RoleCounterfactualResult:
    """One Evaluation Role's Counterfactual Sensitivity: every measurement
    taken for it, plus the movement and leak rate aggregated across them.
    """

    evaluation_role_id: str
    category: str
    measurements: tuple[CounterfactualMeasurement, ...]
    mean_fit_weight_movement: float
    max_fit_weight_movement: int
    redaction_leak_count: int
    redaction_leak_rate: float


@dataclass(frozen=True)
class CounterfactualSensitivityReport:
    """Counterfactual Sensitivity's headline numbers (ticket 09, ADR-0005):
    reported as a first-class result rather than an appendix, and reported
    as measured - a high number here is a finding to publish, not a
    defect to suppress.
    """

    results: tuple[RoleCounterfactualResult, ...]
    mean_fit_weight_movement: float
    max_fit_weight_movement: int
    redaction_leak_count: int
    redaction_leak_rate: float


DEFAULT_SAMPLE_SIZE = 5


def _sample_resumes(
    evaluation_role: EvaluationRole, corpus: Sequence[CorpusResume], sample_size: int
) -> list[CorpusResume]:
    """The first `sample_size` Proxy-Relevant corpus Resumes for this
    Evaluation Role's category, in corpus order - the Resumes plausibly
    Qualified for this Role in production, and the ones whose Fit
    movement is worth measuring. Deterministic, so a re-run samples the
    same Candidates.
    """
    relevance_by_id = construct_proxy_relevance(evaluation_role.category, corpus)
    return [resume for resume in corpus if relevance_by_id[resume.candidate_id]][:sample_size]


@dataclass(frozen=True)
class _MovementAggregate:
    mean_fit_weight_movement: float
    max_fit_weight_movement: int
    redaction_leak_count: int
    redaction_leak_rate: float


def _aggregate_measurements(
    measurements: Sequence[CounterfactualMeasurement],
) -> _MovementAggregate:
    """The mean/max Fit-weight movement and Redaction-leak rate shared by
    both RoleCounterfactualResult (one Evaluation Role) and
    CounterfactualSensitivityReport (every Role) - the same formula
    computed at two different groupings of the same measurements.
    """
    movements = [m.fit_weight_movement for m in measurements if m.fit_weight_movement is not None]
    leaks = sum(1 for m in measurements if m.redaction_leaked)
    return _MovementAggregate(
        mean_fit_weight_movement=sum(movements) / len(movements) if movements else 0.0,
        max_fit_weight_movement=max(movements, default=0),
        redaction_leak_count=leaks,
        redaction_leak_rate=leaks / len(measurements) if measurements else 0.0,
    )


def _aggregate(
    evaluation_role_id: str, category: str, measurements: Sequence[CounterfactualMeasurement]
) -> RoleCounterfactualResult:
    agg = _aggregate_measurements(measurements)
    return RoleCounterfactualResult(
        evaluation_role_id=evaluation_role_id,
        category=category,
        measurements=tuple(measurements),
        mean_fit_weight_movement=agg.mean_fit_weight_movement,
        max_fit_weight_movement=agg.max_fit_weight_movement,
        redaction_leak_count=agg.redaction_leak_count,
        redaction_leak_rate=agg.redaction_leak_rate,
    )


def measure_counterfactual_sensitivity_for_role(
    evaluation_role: EvaluationRole,
    corpus: Sequence[CorpusResume],
    model_client: ModelClient,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> RoleCounterfactualResult:
    """Every counterfactual pair generate-able from up to `sample_size`
    Proxy-Relevant corpus Resumes for `evaluation_role`, measured through
    the real redact-then-rank path and aggregated into this Role's
    Counterfactual Sensitivity.
    """
    measurements = [
        measure_counterfactual_pair(
            evaluation_role.evaluation_role_id,
            corpus_resume.candidate_id,
            evaluation_role.role,
            pair,
            model_client,
        )
        for corpus_resume in _sample_resumes(evaluation_role, corpus, sample_size)
        for pair in generate_counterfactual_pairs(Resume(text=corpus_resume.text))
    ]
    return _aggregate(evaluation_role.evaluation_role_id, evaluation_role.category, measurements)


def run_counterfactual_sensitivity(
    evaluation_roles: Sequence[EvaluationRole],
    corpus: Sequence[CorpusResume],
    model_client: ModelClient,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> CounterfactualSensitivityReport:
    """Counterfactual Sensitivity (ticket 09, ADR-0005) across every
    Evaluation Role in `evaluation_roles`, rather than a single ad hoc
    example - the acceptance criterion this function exists to satisfy.
    """
    if not evaluation_roles:
        raise ValueError("run_counterfactual_sensitivity requires at least one Evaluation Role")

    results = tuple(
        measure_counterfactual_sensitivity_for_role(
            evaluation_role, corpus, model_client, sample_size=sample_size
        )
        for evaluation_role in evaluation_roles
    )
    all_measurements = [m for result in results for m in result.measurements]
    agg = _aggregate_measurements(all_measurements)
    return CounterfactualSensitivityReport(
        results=results,
        mean_fit_weight_movement=agg.mean_fit_weight_movement,
        max_fit_weight_movement=agg.max_fit_weight_movement,
        redaction_leak_count=agg.redaction_leak_count,
        redaction_leak_rate=agg.redaction_leak_rate,
    )
