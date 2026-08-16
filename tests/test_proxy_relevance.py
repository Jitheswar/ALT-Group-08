"""Proxy Relevance construction is a pure function tested directly on
fixed inputs, per the spec's testing decisions - no model client or
corpus download involved.
"""

from __future__ import annotations

from screening.proxy_relevance import CorpusResume, construct_proxy_relevance


def test_a_resume_sharing_the_roles_category_is_relevant():
    resumes = [
        CorpusResume(candidate_id="r1", category="Data Science", text="..."),
        CorpusResume(candidate_id="r2", category="Sales", text="..."),
    ]
    relevance = construct_proxy_relevance("Data Science", resumes)
    assert relevance == {"r1": True, "r2": False}


def test_every_other_category_is_not_relevant_regardless_of_how_many_there_are():
    resumes = [
        CorpusResume(candidate_id="r1", category="Sales", text="..."),
        CorpusResume(candidate_id="r2", category="Testing", text="..."),
        CorpusResume(candidate_id="r3", category="Finance", text="..."),
    ]
    relevance = construct_proxy_relevance("Data Science", resumes)
    assert relevance == {"r1": False, "r2": False, "r3": False}


def test_an_empty_corpus_produces_an_empty_relevance_mapping():
    assert construct_proxy_relevance("Data Science", []) == {}


def test_category_matching_is_exact_not_a_substring_match():
    resumes = [
        CorpusResume(candidate_id="r1", category="Data Science", text="..."),
        CorpusResume(candidate_id="r2", category="Data", text="..."),
    ]
    relevance = construct_proxy_relevance("Data", resumes)
    assert relevance == {"r1": False, "r2": True}
