"""The full corpus sweep (ticket 07): runs the core's public entry point
across the checked-in Evaluation Roles, scores each resulting Shortlist
against Proxy Relevance constructed from corpus category labels, and
reports rank metrics alongside parse-failure rate - the headline numbers
for the whole project.

Every Evaluation Role in a call to run_sweep is swept against the same
`model_client` instance, so the reported numbers describe one system
rather than an average over whatever configuration happened to run each
Role. That is a caller obligation this module cannot enforce by itself,
but it is why `model_client` is taken once, up front, rather than being
built per Role.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from screening.core import run_screening
from screening.domain import Candidate, Resume
from screening.eval_roles import EvaluationRole
from screening.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from screening.model_client import ModelClient, RunMetrics
from screening.proxy_relevance import CorpusResume, construct_proxy_relevance
from screening.ranking import DEFAULT_TOP_BAND_SIZE


@dataclass(frozen=True)
class RoleSweepResult:
    """One Evaluation Role's sweep outcome: rank metrics computed over the
    full Shortlist order it produced, against Proxy Relevance constructed
    from the corpus's own category labels, plus the parse-failure rate
    accrued while producing it.
    """

    evaluation_role_id: str
    category: str
    precision_at_k: float
    ndcg_at_k: float
    reciprocal_rank: float
    parse_failure_count: int
    parse_failure_rate: float


@dataclass(frozen=True)
class SweepReport:
    """The sweep's headline numbers: rank metrics averaged across every
    Evaluation Role swept, reported alongside the parse-failure rate
    accrued over the whole sweep, plus every Role's own result for a
    reader who wants to see what the average is hiding.
    """

    k: int
    results: tuple[RoleSweepResult, ...]
    mean_precision_at_k: float
    mean_ndcg_at_k: float
    mrr: float
    parse_failure_count: int
    parse_failure_rate: float


def _metrics_snapshot(model_client: ModelClient) -> RunMetrics:
    """A point-in-time copy of the model client's cumulative RunMetrics -
    ScriptedModelClient and other doubles with nothing to report simply
    have no `metrics` attribute (model_client.RunMetrics's own docstring),
    so a missing attribute is read as zeroed rather than an error.
    """
    metrics = getattr(model_client, "metrics", None)
    return replace(metrics) if metrics is not None else RunMetrics()


def _parse_failure_stats(before: RunMetrics, after: RunMetrics) -> tuple[int, float]:
    """Parse-failure count and rate accrued between two RunMetrics
    snapshots of the same client - shared by sweep_role (one Role's
    calls) and run_sweep (the whole sweep's).
    """
    count = after.parse_failures - before.parse_failures
    total_attempts = after.total_attempts - before.total_attempts
    rate = count / total_attempts if total_attempts else 0.0
    return count, rate


def sweep_role(
    evaluation_role: EvaluationRole,
    corpus: Sequence[CorpusResume],
    model_client: ModelClient,
    *,
    k: int = DEFAULT_TOP_BAND_SIZE,
) -> RoleSweepResult:
    """Runs the full pipeline once for `evaluation_role` over every Resume
    in `corpus` - the "full corpus sweep" ticket 07 asks for - and scores
    the resulting Shortlist order against Proxy Relevance constructed from
    the corpus's own category labels.

    `k` is passed through as Ranking's `top_band_size` as well as the
    metrics cutoff: a k the comparative pass never actually band-ordered
    to would score a wider band than the one the system ran.
    """
    candidates = [Candidate(id=r.candidate_id, resume=Resume(text=r.text)) for r in corpus]
    relevance_by_id = construct_proxy_relevance(evaluation_role.category, corpus)

    before = _metrics_snapshot(model_client)
    shortlist = run_screening(evaluation_role.role, candidates, model_client, top_band_size=k)
    after = _metrics_snapshot(model_client)

    ordered_relevance = [relevance_by_id[entry.candidate_id] for entry in shortlist.entries]
    parse_failure_count, parse_failure_rate = _parse_failure_stats(before, after)

    return RoleSweepResult(
        evaluation_role_id=evaluation_role.evaluation_role_id,
        category=evaluation_role.category,
        precision_at_k=precision_at_k(ordered_relevance, k),
        ndcg_at_k=ndcg_at_k(ordered_relevance, k),
        reciprocal_rank=reciprocal_rank(ordered_relevance),
        parse_failure_count=parse_failure_count,
        parse_failure_rate=parse_failure_rate,
    )


def run_sweep(
    evaluation_roles: Sequence[EvaluationRole],
    corpus: Sequence[CorpusResume],
    model_client: ModelClient,
    *,
    k: int = DEFAULT_TOP_BAND_SIZE,
) -> SweepReport:
    """Sweeps every Evaluation Role in `evaluation_roles` over `corpus`
    with the single `model_client` passed in, and averages the per-Role
    rank metrics into the sweep's headline numbers. `evaluation_roles` is
    normally the full checked-in set
    (screening.eval_roles.read_evaluation_roles), so the sweep is
    reproducible from what is under version control.
    """
    if not evaluation_roles:
        raise ValueError("run_sweep requires at least one Evaluation Role")

    overall_before = _metrics_snapshot(model_client)
    results = tuple(
        sweep_role(evaluation_role, corpus, model_client, k=k)
        for evaluation_role in evaluation_roles
    )
    overall_after = _metrics_snapshot(model_client)

    count = len(results)
    parse_failure_count, parse_failure_rate = _parse_failure_stats(overall_before, overall_after)

    return SweepReport(
        k=k,
        results=results,
        mean_precision_at_k=sum(r.precision_at_k for r in results) / count,
        mean_ndcg_at_k=sum(r.ndcg_at_k for r in results) / count,
        mrr=sum(r.reciprocal_rank for r in results) / count,
        parse_failure_count=parse_failure_count,
        parse_failure_rate=parse_failure_rate,
    )
