"""Rank-quality metrics over a binary relevance sequence: Precision@k,
NDCG@k, and Reciprocal Rank (averaged by a caller into MRR across
Evaluation Roles). Pure functions, tested directly on fixed inputs per the
spec's testing decisions.

Callers supply relevance already resolved to booleans, in the order a
Shortlist actually returned them - nothing here knows anything about
corpus categories, the model client, or what "relevant" was constructed
from. See screening.proxy_relevance for that construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    """The fraction of the first k entries that are relevant. 0.0 if there
    is nothing to look at - k <= 0, or the ranking itself is empty - since
    a Shortlist can legitimately be shorter than k rather than that being
    an error.
    """
    top = relevance[:k] if k > 0 else []
    return sum(top) / len(top) if top else 0.0


def _dcg_at_k(relevance: Sequence[bool], k: int) -> float:
    top = relevance[:k] if k > 0 else []
    return sum(1.0 / math.log2(position + 2) for position, relevant in enumerate(top) if relevant)


def ndcg_at_k(relevance: Sequence[bool], k: int) -> float:
    """Normalized against the best possible ordering of this same
    relevance set (every relevant entry first), so 1.0 means Ranking put
    every relevant Candidate as early as the Proxy Relevance judgement
    allows - not that every Candidate was relevant. 0.0, not undefined,
    when there is no relevant entry to rank at all.
    """
    ideal_dcg = _dcg_at_k(sorted(relevance, reverse=True), k)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg_at_k(relevance, k) / ideal_dcg


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1 / (1-indexed rank of the first relevant entry), 0.0 if none of
    the ranking is relevant. This is one query's contribution to Mean
    Reciprocal Rank - the mean across Evaluation Roles is the caller's to
    take, since a single ranking has no "mean" of its own.
    """
    for position, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / position
    return 0.0
