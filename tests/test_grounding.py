"""Deterministic grounding, and the honest "unavailable" report for the
semantic layer."""

from app.generation.grounding import (
    SEMANTIC_UNAVAILABLE_REASON,
    deterministic_grounding,
)

UID_A = "a" * 32
UID_B = "b" * 32


def test_a_fully_valid_non_abstained_answer_is_grounded() -> None:
    result = deterministic_grounding(
        answer_text=f"Fact one [{UID_A}]. Fact two [{UID_B}].",
        declared_citations=[UID_A, UID_B],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A, UID_B}),
        abstained=False,
    )
    assert result.grounded
    assert result.citation_valid
    assert result.detail["deterministic"]["citation_valid"] is True
    assert result.detail["deterministic"]["citation_coverage"] == 1.0
    assert result.detail["deterministic"]["abstention_flag_consistent"] is True


def test_a_valid_abstention_is_grounded() -> None:
    result = deterministic_grounding(
        answer_text="There is insufficient evidence to answer this question.",
        declared_citations=[],
        retrieved_chunk_uids=frozenset(),
        selected_chunk_uids=frozenset(),
        abstained=True,
    )
    assert result.grounded
    assert result.detail["deterministic"]["citation_coverage"] == 1.0


def test_partial_coverage_is_not_grounded() -> None:
    result = deterministic_grounding(
        answer_text=f"Cited fact [{UID_A}]. Uncited fact.",
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=False,
    )
    assert not result.grounded
    assert result.detail["deterministic"]["citation_coverage"] == 0.5


def test_a_citation_to_a_non_selected_chunk_is_not_grounded() -> None:
    result = deterministic_grounding(
        answer_text=f"Fact [{UID_B}].",
        declared_citations=[UID_B],
        retrieved_chunk_uids=frozenset({UID_A, UID_B}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=False,
    )
    assert not result.grounded
    assert result.citation_valid is False
    assert result.detail["deterministic"]["citation_valid"] is False


def test_an_abstention_carrying_citations_is_not_consistent() -> None:
    result = deterministic_grounding(
        answer_text=f"Fact [{UID_A}].",
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=True,
    )
    assert not result.grounded
    assert result.detail["deterministic"]["abstention_flag_consistent"] is False


# --- the semantic layer is honestly reported as unavailable -----------


def test_semantic_layer_is_always_reported_as_unavailable() -> None:
    result = deterministic_grounding(
        answer_text=f"Fact [{UID_A}].",
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=False,
    )
    assert result.detail["semantic"]["status"] == "unavailable"
    assert result.detail["semantic"]["reason"] == SEMANTIC_UNAVAILABLE_REASON


def test_semantic_layer_is_never_a_number() -> None:
    """The specification permits an approximate NLI score; this milestone
    must never fabricate one from the reranker's own relevance score."""
    result = deterministic_grounding(
        answer_text=f"Fact [{UID_A}].",
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=False,
    )
    semantic = result.detail["semantic"]
    assert "score" not in semantic
    assert not isinstance(semantic.get("status"), (int, float))


def test_grounding_detail_is_json_serializable() -> None:
    import json

    result = deterministic_grounding(
        answer_text=f"Fact [{UID_A}].",
        declared_citations=[UID_A],
        retrieved_chunk_uids=frozenset({UID_A}),
        selected_chunk_uids=frozenset({UID_A}),
        abstained=False,
    )
    json.dumps(result.detail)  # must not raise
