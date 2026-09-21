"""Retrieval evaluation metrics: hand-computed values, no database, no
provider, no fixture corpus — `evals/retrieval/metrics.py` is pure and every
test here proves it by never touching anything but plain Python values.
"""

import math

import pytest

from app.retrieval.filters import RetrievalFilters
from app.retrieval.vector import ChunkEvidence
from evals.retrieval.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    metadata_filter_correctness,
    metadata_filter_violated,
    ndcg_at_10,
    precision_at_5,
    recall_at_k,
)


def _evidence(chunk_uid: str, **overrides) -> ChunkEvidence:
    import uuid

    fields = dict(
        chunk_uid=chunk_uid,
        text="text",
        sequence=0,
        page=None,
        section=None,
        char_start=0,
        char_end=4,
        token_count=1,
        document_id=uuid.uuid4(),
        document_title="doc",
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )
    fields.update(overrides)
    return ChunkEvidence(**fields)


# --- 1. perfect ranking ------------------------------------------------


def test_perfect_ranking() -> None:
    expected = {"a"}
    retrieved = ["a", "b", "c"]

    assert recall_at_k(expected, retrieved, 5) == 1.0
    assert precision_at_5(expected, retrieved) == pytest.approx(0.2)
    assert mean_reciprocal_rank(expected, retrieved) == 1.0
    assert ndcg_at_10(expected, retrieved) == pytest.approx(1.0)


# --- 2. no relevant results ----------------------------------------------


def test_no_relevant_results() -> None:
    expected = {"z"}
    retrieved = ["a", "b", "c"]

    assert recall_at_k(expected, retrieved, 5) == 0.0
    assert precision_at_5(expected, retrieved) == 0.0
    assert mean_reciprocal_rank(expected, retrieved) == 0.0
    assert ndcg_at_10(expected, retrieved) == 0.0


# --- 3. relevant result at rank 2 -----------------------------------------


def test_relevant_result_at_rank_two() -> None:
    expected = {"b"}
    retrieved = ["a", "b", "c", "d", "e"]

    assert recall_at_k(expected, retrieved, 5) == 1.0
    assert precision_at_5(expected, retrieved) == pytest.approx(0.2)
    assert mean_reciprocal_rank(expected, retrieved) == pytest.approx(0.5)
    assert ndcg_at_10(expected, retrieved) == pytest.approx(0.6309297535714575)


# --- 4. relevant result at rank 20 (MRR's own depth) ----------------------


def test_relevant_result_at_rank_twenty() -> None:
    expected = {"x"}
    retrieved = [f"n{i}" for i in range(19)] + ["x"]
    assert len(retrieved) == 20

    assert recall_at_k(expected, retrieved, 5) == 0.0
    assert recall_at_k(expected, retrieved, 10) == 0.0
    assert precision_at_5(expected, retrieved) == 0.0
    assert mean_reciprocal_rank(expected, retrieved) == pytest.approx(0.05)
    assert ndcg_at_10(expected, retrieved) == 0.0  # outside the top-10 window


def test_mrr_never_looks_past_its_depth() -> None:
    """A relevant item at rank 21 is invisible to MRR at depth 20 -- the
    locked "whole returned top-20 list" rule, not "search forever"."""
    expected = {"x"}
    retrieved = [f"n{i}" for i in range(20)] + ["x"]

    assert mean_reciprocal_rank(expected, retrieved, depth=20) == 0.0


# --- 5. multiple relevant chunks ------------------------------------------


def test_multiple_relevant_chunks() -> None:
    expected = {"a", "c"}
    retrieved = ["a", "b", "c", "d", "e"]

    assert recall_at_k(expected, retrieved, 5) == 1.0
    assert precision_at_5(expected, retrieved) == pytest.approx(0.4)
    assert mean_reciprocal_rank(expected, retrieved) == 1.0
    assert ndcg_at_10(expected, retrieved) == pytest.approx(0.9197207891481876)


# --- 6. expected set larger than the cutoff -------------------------------


def test_expected_set_larger_than_recall_at_5_cutoff() -> None:
    """|E| = 6, all in the top 5 -- Recall@5 has a structural ceiling below
    1.0 no matter how good the retriever is."""
    expected = set("ABCDEF")
    retrieved = list("ABCDEFGHIJ")

    assert recall_at_k(expected, retrieved, 5) == pytest.approx(5 / 6)
    assert precision_at_5(expected, retrieved) == 1.0
    # The 6 relevant items already occupy positions 1-6, which is also the
    # ideal ordering for a 10-item window -- nDCG is exactly 1.0 despite
    # Recall@5 being capped below 1.0.
    assert ndcg_at_10(expected, retrieved) == pytest.approx(1.0)


# --- 7. empty expected set: undefined, not zero ---------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda: recall_at_k(set(), ["a"], 5),
        lambda: precision_at_5(set(), ["a"]),
        lambda: mean_reciprocal_rank(set(), ["a"]),
        lambda: ndcg_at_10(set(), ["a"]),
    ],
)
def test_empty_expected_set_is_refused_not_scored_as_zero(fn) -> None:
    with pytest.raises(ValueError, match="undefined"):
        fn()


# --- 8. empty result list --------------------------------------------------


def test_empty_result_list() -> None:
    expected = {"a"}
    retrieved: list[str] = []

    assert recall_at_k(expected, retrieved, 5) == 0.0
    assert precision_at_5(expected, retrieved) == 0.0
    assert mean_reciprocal_rank(expected, retrieved) == 0.0
    assert ndcg_at_10(expected, retrieved) == 0.0


# --- 9. duplicate expected ids ---------------------------------------------


def test_duplicate_expected_ids_collapse_in_a_set() -> None:
    """`expected_chunk_uids` arrives from YAML as a list and may contain a
    duplicate; converting to a `set` (the question loader's job, exercised
    in tests/test_evals_questions.py) is what makes recall count each
    distinct chunk once, not once per repetition in the list."""
    expected = set(["a", "a", "b"])
    assert expected == {"a", "b"}
    assert len(expected) == 2

    retrieved = ["a", "b", "c"]
    assert recall_at_k(expected, retrieved, 5) == 1.0


# --- 10. nDCG's IDCG=0 guard -----------------------------------------------


def test_ndcg_never_divides_by_a_zero_idcg() -> None:
    """IDCG is 0 only when `expected` is empty (an all-zero ideal ranking),
    which `ndcg_at_10` refuses outright before it would ever compute one --
    proving the function cannot reach a ZeroDivisionError through its public
    signature."""
    with pytest.raises(ValueError):
        ndcg_at_10(set(), ["a", "b", "c"])


def test_dcg_at_k_of_an_all_zero_relevance_list_is_zero() -> None:
    """The numerator side of the same guard: an all-irrelevant ranking's
    DCG is exactly 0, not merely close to it."""
    assert dcg_at_k([0, 0, 0], 3) == 0.0


def test_dcg_at_k_matches_the_formula_by_hand() -> None:
    # rel = [1, 0, 1] -> 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5
    assert dcg_at_k([1, 0, 1], 3) == pytest.approx(1.5)
    assert dcg_at_k([1, 0, 1], 3) == pytest.approx(
        1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    )


# --- metadata-filter correctness -------------------------------------------


def test_metadata_filter_violated_checks_department() -> None:
    filters = RetrievalFilters(department="Engineering")
    matching = _evidence("a", department="Engineering")
    violating = _evidence("b", department="Sales")

    assert metadata_filter_violated(matching, filters) is False
    assert metadata_filter_violated(violating, filters) is True


def test_metadata_filter_violated_checks_category() -> None:
    filters = RetrievalFilters(category="Policy")
    assert metadata_filter_violated(_evidence("a", category="Policy"), filters) is False
    assert metadata_filter_violated(_evidence("a", category="SOP"), filters) is True


def test_metadata_filter_violated_checks_document_id() -> None:
    import uuid

    doc_id = uuid.uuid4()
    filters = RetrievalFilters(document_id=doc_id)
    assert metadata_filter_violated(_evidence("a", document_id=doc_id), filters) is False
    assert metadata_filter_violated(_evidence("a", document_id=uuid.uuid4()), filters) is True


def test_metadata_filter_violated_checks_tags_containment() -> None:
    filters = RetrievalFilters(tags=("access", "urgent"))
    assert (
        metadata_filter_violated(_evidence("a", tags=("access", "urgent", "extra")), filters)
        is False
    )
    assert metadata_filter_violated(_evidence("a", tags=("access",)), filters) is True


def test_metadata_filter_violated_excludes_superseded_by_default() -> None:
    filters = RetrievalFilters()
    assert (
        metadata_filter_violated(_evidence("a", version_status="superseded"), filters) is True
    )
    assert (
        metadata_filter_violated(_evidence("a", version_status="active"), filters) is False
    )


def test_metadata_filter_violated_allows_superseded_when_requested() -> None:
    filters = RetrievalFilters(include_superseded=True)
    assert (
        metadata_filter_violated(_evidence("a", version_status="superseded"), filters) is False
    )


def test_metadata_filter_correctness_all_correct() -> None:
    filters = RetrievalFilters(department="Engineering")
    questions = [
        (filters, [_evidence("a", department="Engineering")]),
        (filters, [_evidence("b", department="Engineering")]),
    ]
    assert metadata_filter_correctness(questions) == 1.0


def test_metadata_filter_correctness_one_violation() -> None:
    filters = RetrievalFilters(department="Engineering")
    questions = [
        (filters, [_evidence("a", department="Engineering")]),
        (filters, [_evidence("b", department="Sales")]),  # violates
    ]
    assert metadata_filter_correctness(questions) == pytest.approx(0.5)


def test_metadata_filter_correctness_one_violating_candidate_fails_the_question() -> None:
    """A single violating candidate among several correct ones still fails
    that question -- correctness is "zero violations", not "mostly right"."""
    filters = RetrievalFilters(department="Engineering")
    questions = [
        (
            filters,
            [
                _evidence("a", department="Engineering"),
                _evidence("b", department="Engineering"),
                _evidence("c", department="Sales"),
            ],
        )
    ]
    assert metadata_filter_correctness(questions) == 0.0


def test_metadata_filter_correctness_refuses_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        metadata_filter_correctness([])


def test_metadata_filter_correctness_with_no_candidates_is_trivially_correct() -> None:
    """Nothing was retrieved, so nothing could have violated the filter."""
    filters = RetrievalFilters(department="Engineering")
    assert metadata_filter_correctness([(filters, [])]) == 1.0
