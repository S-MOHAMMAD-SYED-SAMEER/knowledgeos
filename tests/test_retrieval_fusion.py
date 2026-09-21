"""Reciprocal Rank Fusion. Pure function, exact formula — no database, no
provider, and every test here is over plain `chunk_uid` strings a fixture
made up.
"""

import inspect

import pytest

from app.retrieval import fusion
from app.retrieval.fusion import DEFAULT_RRF_K, fuse


def test_default_k_is_sixty() -> None:
    assert DEFAULT_RRF_K == 60


def test_a_candidate_in_both_lists_sums_both_terms() -> None:
    result = fuse(vector_ranking=["a"], lexical_ranking=["a"], k=60)

    assert len(result) == 1
    entry = result[0]
    assert entry.chunk_uid == "a"
    assert entry.vector_rank == 1
    assert entry.lexical_rank == 1
    assert entry.rrf_score == pytest.approx(1 / 61 + 1 / 61)


def test_a_candidate_in_only_one_channel_gets_one_term_and_a_null_rank() -> None:
    result = fuse(vector_ranking=["a", "b"], lexical_ranking=[], k=60)
    by_uid = {entry.chunk_uid: entry for entry in result}

    assert by_uid["a"].rrf_score == pytest.approx(1 / 61)
    assert by_uid["a"].lexical_rank is None
    assert by_uid["a"].vector_rank == 1
    assert by_uid["b"].rrf_score == pytest.approx(1 / 62)
    assert by_uid["b"].lexical_rank is None


def test_ranks_are_one_based() -> None:
    result = fuse(vector_ranking=["a", "b", "c"], lexical_ranking=[], k=60)
    ranks = {entry.chunk_uid: entry.vector_rank for entry in result}
    assert ranks == {"a": 1, "b": 2, "c": 3}


def test_k_is_configurable() -> None:
    default = fuse(vector_ranking=["a"], lexical_ranking=[], k=60)[0]
    custom = fuse(vector_ranking=["a"], lexical_ranking=[], k=10)[0]

    assert default.rrf_score == pytest.approx(1 / 61)
    assert custom.rrf_score == pytest.approx(1 / 11)


def test_no_score_normalization_the_function_never_sees_a_channel_score() -> None:
    """Fusion takes rank-ordered `chunk_uid` lists only — there is no score
    parameter for anything to normalize, even by accident."""
    signature = inspect.signature(fusion.fuse)
    assert set(signature.parameters) == {"vector_ranking", "lexical_ranking", "k"}


def test_empty_inputs_produce_an_empty_result() -> None:
    assert fuse(vector_ranking=[], lexical_ranking=[]) == []


def test_one_empty_channel_still_ranks_the_other() -> None:
    result = fuse(vector_ranking=["a", "b"], lexical_ranking=[])
    assert [entry.chunk_uid for entry in result] == ["a", "b"]


def test_fusion_produces_no_duplicate_chunk_uids() -> None:
    """A chunk_uid appearing in both inputs still yields exactly one entry —
    fusion cannot itself create a duplicate."""
    result = fuse(vector_ranking=["a", "b"], lexical_ranking=["b", "a", "c"])
    uids = [entry.chunk_uid for entry in result]
    assert len(uids) == len(set(uids)) == 3


def test_tied_scores_break_deterministically_on_chunk_uid() -> None:
    """Two chunks each seen by exactly one distinct channel, at the same
    rank, score identically — and the ordering falls back to `chunk_uid`."""
    result = fuse(vector_ranking=["z"], lexical_ranking=["a"])

    assert result[0].rrf_score == pytest.approx(result[1].rrf_score)
    assert [entry.chunk_uid for entry in result] == ["a", "z"]


def test_repeated_fusion_of_the_same_inputs_is_identical() -> None:
    first = fuse(vector_ranking=["c", "a", "b"], lexical_ranking=["b", "c"])
    second = fuse(vector_ranking=["c", "a", "b"], lexical_ranking=["b", "c"])
    assert first == second


def test_ordering_is_by_score_descending() -> None:
    result = fuse(vector_ranking=["first", "second", "third"], lexical_ranking=[])
    scores = [entry.rrf_score for entry in result]
    assert scores == sorted(scores, reverse=True)
