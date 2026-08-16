"""The Gold Set (ticket 08, ADR-0003, CONTEXT.md): roughly 150 Role-and-
Resume pairs judged by hand against the written rubric at
data/gold_set/rubric.md, held out of the corpus a full sweep runs against,
and compared here against Proxy Relevance for the same pairs - so a
divergence between the two tiers is a finding worth reporting, not a bug to
smooth over.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from screening.eval_roles import EvaluationRole
from screening.proxy_relevance import CorpusResume, construct_proxy_relevance


@dataclass(frozen=True)
class GoldSetLabel:
    """One hand-labelled Role-and-Resume pair: the Evaluation Role it was
    judged against, the Resume judged (kept as a CorpusResume so its own
    corpus category travels with it, even though the rubric's label ignores
    that category - see data/gold_set/rubric.md), the human relevance
    judgement, and the Justification the rubric requires.
    """

    evaluation_role_id: str
    resume: CorpusResume
    relevant: bool
    justification: str


def write_gold_set(labels: Sequence[GoldSetLabel], path: Path) -> Path:
    """Writes the whole Gold Set as one checked-in file, created exclusively
    and made read-only, mirroring screening.eval_roles.write_evaluation_role:
    regenerating the set is a deliberate versioned event, never a silent
    overwrite of hand-produced labels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "evaluation_role_id": label.evaluation_role_id,
            "resume": asdict(label.resume),
            "relevant": label.relevant,
            "justification": label.justification,
        }
        for label in labels
    ]

    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o444)
    return path


def read_gold_set(path: Path) -> list[GoldSetLabel]:
    """Every hand-labelled pair in the checked-in Gold Set file at `path`."""
    data = json.loads(path.read_text())
    return [
        GoldSetLabel(
            evaluation_role_id=row["evaluation_role_id"],
            resume=CorpusResume(**row["resume"]),
            relevant=row["relevant"],
            justification=row["justification"],
        )
        for row in data
    ]


def exclude_gold_set_candidates(
    corpus: Sequence[CorpusResume], gold_set: Sequence[GoldSetLabel]
) -> list[CorpusResume]:
    """Every corpus Resume except the ones hand-labelled into the Gold Set -
    what a full sweep should actually run against, so a Resume already
    hand-labelled by a human is never also scored through Proxy Relevance at
    corpus scale in the same run. This is the "held out of the sweep" half
    of ticket 08's acceptance criteria; scripts/run_sweep.py applies it to
    the live corpus before sweeping.
    """
    held_out = {label.resume.candidate_id for label in gold_set}
    return [resume for resume in corpus if resume.candidate_id not in held_out]


@dataclass(frozen=True)
class GoldSetDivergence:
    """One Gold Set pair's hand label set against Proxy Relevance for the
    same pair. `agree` is False exactly when the two tiers disagree.
    """

    evaluation_role_id: str
    candidate_id: str
    hand_relevant: bool
    proxy_relevant: bool
    agree: bool


@dataclass(frozen=True)
class GoldSetReport:
    """The Gold Set's headline comparison: how often Proxy Relevance agrees
    with the hand-labelled Gold Set over the same pairs, broken out into the
    two ways it can disagree - the hand label treated as the more trustworthy
    signal for this comparison only, per ADR-0003 - plus every pair's own
    comparison for a reader who wants to see what the agreement rate hides.
    """

    divergences: tuple[GoldSetDivergence, ...]
    agreement_rate: float
    proxy_false_positive_rate: float
    proxy_false_negative_rate: float


def compare_gold_set_to_proxy_relevance(
    gold_set: Sequence[GoldSetLabel], evaluation_roles: Sequence[EvaluationRole]
) -> GoldSetReport:
    """Computes Proxy Relevance for each Gold Set pair - via the same
    construct_proxy_relevance the sweep uses, so both tiers are measured by
    the identical construction - and reports where it agrees and disagrees
    with the hand label.

    `proxy_false_positive_rate` is the fraction of hand-labelled-not-relevant
    pairs Proxy Relevance called relevant anyway; `proxy_false_negative_rate`
    is the fraction of hand-labelled-relevant pairs Proxy Relevance missed.
    Each is 0.0, not undefined, when its denominator (pairs of that hand
    label) is empty.
    """
    if not gold_set:
        raise ValueError("compare_gold_set_to_proxy_relevance requires at least one label")

    category_by_role_id = {role.evaluation_role_id: role.category for role in evaluation_roles}

    divergences = []
    for label in gold_set:
        role_category = category_by_role_id[label.evaluation_role_id]
        proxy_relevant = construct_proxy_relevance(role_category, [label.resume])[
            label.resume.candidate_id
        ]
        divergences.append(
            GoldSetDivergence(
                evaluation_role_id=label.evaluation_role_id,
                candidate_id=label.resume.candidate_id,
                hand_relevant=label.relevant,
                proxy_relevant=proxy_relevant,
                agree=label.relevant == proxy_relevant,
            )
        )

    count = len(divergences)
    hand_negative = [d for d in divergences if not d.hand_relevant]
    hand_positive = [d for d in divergences if d.hand_relevant]
    false_positives = sum(1 for d in hand_negative if d.proxy_relevant)
    false_negatives = sum(1 for d in hand_positive if not d.proxy_relevant)

    return GoldSetReport(
        divergences=tuple(divergences),
        agreement_rate=sum(d.agree for d in divergences) / count,
        proxy_false_positive_rate=(false_positives / len(hand_negative)) if hand_negative else 0.0,
        proxy_false_negative_rate=(false_negatives / len(hand_positive)) if hand_positive else 0.0,
    )
