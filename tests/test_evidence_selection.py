"""Selecting which reranked candidates reach the model — deterministic,
top 8 by `final_rank`, no threshold (locked project decision D17)."""

import uuid

from app.generation.evidence import SELECTION_LIMIT, select_evidence
from app.reranking.pipeline import RerankedChunk
from app.retrieval.vector import ChunkEvidence


def _reranked(final_rank: int) -> RerankedChunk:
    uid = f"{final_rank:032x}"
    evidence = ChunkEvidence(
        chunk_uid=uid,
        text=f"chunk {final_rank}",
        sequence=final_rank,
        page=None,
        section=None,
        char_start=0,
        char_end=10,
        token_count=2,
        document_id=uuid.uuid4(),
        document_title="Doc",
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )
    return RerankedChunk(
        evidence=evidence,
        lexical_rank=1,
        vector_rank=1,
        rrf_score=1.0,
        fusion_rank=final_rank,
        rerank_score=1.0,
        final_rank=final_rank,
    )


def test_the_selection_limit_is_eight() -> None:
    """The locked project decision, D17."""
    assert SELECTION_LIMIT == 8


def test_exactly_the_top_eight_are_selected_from_twenty() -> None:
    candidates = [_reranked(i) for i in range(1, 21)]

    result = select_evidence(candidates)

    selected_ranks = {item.candidate.final_rank for item in result if item.selected}
    assert selected_ranks == set(range(1, 9))


def test_one_entry_per_input_candidate_selected_or_not() -> None:
    candidates = [_reranked(i) for i in range(1, 21)]

    result = select_evidence(candidates)

    assert len(result) == 20
    assert sum(1 for item in result if item.selected) == 8
    assert sum(1 for item in result if not item.selected) == 12


def test_fewer_candidates_than_the_limit_selects_all_of_them() -> None:
    candidates = [_reranked(i) for i in range(1, 4)]

    result = select_evidence(candidates)

    assert all(item.selected for item in result)


def test_zero_candidates_selects_nothing() -> None:
    assert select_evidence([]) == []


def test_selection_never_reorders_the_input() -> None:
    """Deterministic and order-preserving: the caller's own iteration order
    (already `final_rank`-contiguous from milestone 6) is untouched."""
    candidates = [_reranked(i) for i in range(1, 11)]

    result = select_evidence(candidates)

    assert [item.candidate.final_rank for item in result] == list(range(1, 11))


def test_the_limit_is_overridable_for_testing_but_defaults_to_eight() -> None:
    candidates = [_reranked(i) for i in range(1, 6)]

    result = select_evidence(candidates, limit=2)

    assert {item.candidate.final_rank for item in result if item.selected} == {1, 2}
