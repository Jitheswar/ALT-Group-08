"""Rank metrics are pure functions tested directly on fixed inputs, per
the spec's testing decisions - no model client or corpus involved.
"""

from __future__ import annotations

import math

from screening.metrics import ndcg_at_k, precision_at_k, reciprocal_rank


def test_precision_at_k_counts_relevant_entries_in_the_first_k():
    relevance = [True, False, True, True, False]
    assert precision_at_k(relevance, 3) == 2 / 3


def test_precision_at_k_over_the_whole_ranking_when_k_exceeds_its_length():
    relevance = [True, False, True]
    assert precision_at_k(relevance, 10) == 2 / 3


def test_precision_at_k_is_zero_for_an_empty_ranking():
    assert precision_at_k([], 5) == 0.0


def test_precision_at_k_is_zero_when_k_is_not_positive():
    assert precision_at_k([True, True], 0) == 0.0
    assert precision_at_k([True, True], -1) == 0.0


def test_ndcg_at_k_is_one_when_relevant_entries_are_already_ranked_first():
    relevance = [True, True, False, False]
    assert ndcg_at_k(relevance, 4) == 1.0


def test_ndcg_at_k_is_zero_when_there_is_no_relevant_entry_at_all():
    assert ndcg_at_k([False, False, False], 3) == 0.0


def test_ndcg_at_k_penalizes_a_relevant_entry_ranked_lower_than_ideal():
    # One relevant entry, ranked second instead of first: DCG is
    # 1/log2(3), the ideal is 1/log2(2) - the ratio is the expected NDCG,
    # strictly less than 1.0.
    relevance = [False, True, False]
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert math.isclose(ndcg_at_k(relevance, 3), expected)


def test_ndcg_at_k_is_zero_when_k_is_not_positive():
    assert ndcg_at_k([True, True], 0) == 0.0
    assert ndcg_at_k([True, True], -1) == 0.0


def test_ndcg_at_k_only_considers_the_first_k_entries():
    # The one relevant entry sits outside the top 2, so within k=2 there is
    # nothing to reward, and the ideal-within-k is also all-irrelevant -
    # 0.0, not an error, per ndcg_at_k's own contract.
    relevance = [False, False, True]
    assert ndcg_at_k(relevance, 2) == 0.0


def test_reciprocal_rank_of_a_relevant_entry_ranked_first():
    assert reciprocal_rank([True, False, False]) == 1.0


def test_reciprocal_rank_of_a_relevant_entry_ranked_third():
    assert reciprocal_rank([False, False, True]) == 1 / 3


def test_reciprocal_rank_is_zero_when_nothing_is_relevant():
    assert reciprocal_rank([False, False, False]) == 0.0


def test_reciprocal_rank_of_an_empty_ranking_is_zero():
    assert reciprocal_rank([]) == 0.0
