"""Shared input-loading for the evaluation scripts (scripts/run_sweep.py,
scripts/run_counterfactual_sensitivity.py): read the checked-in Evaluation
Roles, load the resume-atlas corpus, and hold the checked-in Gold Set out of
it - the same three steps and the same failure modes both scripts need
before they can do their own measurement.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from screening.eval_roles import EvaluationRole, read_evaluation_roles
from screening.gold_set import exclude_gold_set_candidates, read_gold_set, resolve_gold_set_path
from screening.proxy_relevance import CorpusResume
from screening.resume_atlas import load_resume_atlas_corpus

GOLD_SET_HELP = (
    "Path to the checked-in Gold Set, held out of the {noun}; defaults to "
    "data/gold_set/gold_set.json if present, or none if the Gold Set has "
    "not been built yet. A path given explicitly must exist - it is an "
    "error, not a silent no-op, to point this at a file that isn't there"
)


@dataclass(frozen=True)
class EvaluationInputs:
    evaluation_roles: list[EvaluationRole]
    corpus: list[CorpusResume]


def load_evaluation_inputs(
    *,
    evaluation_roles_dir: Path,
    resume_atlas_parquet: Path,
    gold_set: Path | None,
    held_out_noun: str,
) -> EvaluationInputs | None:
    """Every input a sweep or measurement script needs: the checked-in
    Evaluation Roles, and the resume-atlas corpus with the checked-in Gold
    Set (if any) held out of it. Prints its own error and returns None on
    any failure, so a caller can just check for that and exit - `held_out_noun`
    names what the Gold Set is being held out of (e.g. "sweep corpus") in
    that message, since the two scripts describe it slightly differently.
    """
    evaluation_roles = read_evaluation_roles(evaluation_roles_dir)
    if not evaluation_roles:
        print(f"error: no Evaluation Roles found under {evaluation_roles_dir}", file=sys.stderr)
        return None

    corpus = load_resume_atlas_corpus(resume_atlas_parquet)
    if not corpus:
        print(f"error: no Resumes loaded from {resume_atlas_parquet}", file=sys.stderr)
        return None

    try:
        gold_set_path = resolve_gold_set_path(gold_set)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None

    if gold_set_path is not None:
        labels = read_gold_set(gold_set_path)
        before = len(corpus)
        corpus = exclude_gold_set_candidates(corpus, labels)
        print(f"Held {before - len(corpus)} Gold Set Resume(s) out of the {held_out_noun}")

    return EvaluationInputs(evaluation_roles=evaluation_roles, corpus=corpus)
