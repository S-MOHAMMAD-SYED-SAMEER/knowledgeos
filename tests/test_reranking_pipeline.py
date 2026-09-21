"""The real integration boundary: milestone 5's `retrieve()` against a real
database, handed to milestone 6's `rerank()`.

Everything here runs the actual retrieval SQL (through `FakeEmbeddingProvider`,
per the specification's unit-test allowance) and the actual reranking
orchestration (through the passthrough or the deterministic fake, per the
same allowance for reranking). `app/retrieval/` and `app/reranking/` are
exercised together exactly as `POST /query` wires them, without going
through HTTP.
"""

import pytest

from app.providers import FakeEmbeddingProvider, FakeRerankProvider, PassthroughRerankProvider
from app.reranking.pipeline import rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import FINAL_CANDIDATE_LIMIT, retrieve
from app.storage import LocalStorage

from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def test_m5_top_twenty_reaches_the_reranker_unchanged_in_count(
    session, storage, embeddings
) -> None:
    for i in range(30):
        seed_active_version(
            session, storage, text=f"widget candidate {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )
    assert len(result.candidates) == FINAL_CANDIDATE_LIMIT == 20

    reranked = rerank("widget", result.candidates, PassthroughRerankProvider())

    assert len(reranked) == 20
    assert {c.evidence.chunk_uid for c in reranked} == {
        c.evidence.chunk_uid for c in result.candidates
    }


def test_reranked_order_can_differ_from_fusion_order(session, storage, embeddings) -> None:
    for i in range(8):
        seed_active_version(
            session, storage, text=f"widget entry {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )
    fusion_order = [c.evidence.chunk_uid for c in result.candidates]

    reranked = rerank("widget", result.candidates, FakeRerankProvider())
    reranked_order = [c.evidence.chunk_uid for c in reranked]

    # The fake's scores are content-derived, not position-derived, so with
    # 8 genuinely different chunks the two orders are not required to match
    # -- and the fusion order is still recoverable from `fusion_rank`.
    assert {c.evidence.chunk_uid: c.fusion_rank for c in reranked} == {
        c.evidence.chunk_uid: c.final_rank for c in result.candidates
    }
    assert set(reranked_order) == set(fusion_order)


def test_final_rank_is_contiguous_after_reranking(session, storage, embeddings) -> None:
    for i in range(6):
        seed_active_version(
            session, storage, text=f"widget item {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )
    reranked = rerank("widget", result.candidates, FakeRerankProvider())

    assert [c.final_rank for c in reranked] == list(range(1, len(reranked) + 1))


def test_zero_retrieval_candidates_means_reranking_is_skipped(
    session, storage, embeddings
) -> None:
    result = retrieve(
        session,
        normalized_text="nothing has been indexed",
        query_vector=embeddings.embed(["nothing has been indexed"])[0],
        filters=RetrievalFilters(),
    )
    assert result.candidates == []

    calls = []

    class Spy:
        model_name = "spy"

        def rerank(self, query, candidates):
            calls.append(candidates)
            return []

    reranked = rerank("nothing has been indexed", result.candidates, Spy())

    assert reranked == []
    assert calls == []


def test_evidence_survives_reranking_unchanged(session, storage, embeddings) -> None:
    """Text, page, section, document/version metadata are untouched by
    reranking -- only the ranking changes."""
    seed_active_version(
        session,
        storage,
        text="the emergency access procedure",
        department="Engineering",
        embeddings=embeddings,
    )

    result = retrieve(
        session,
        normalized_text="emergency access",
        query_vector=embeddings.embed(["emergency access"])[0],
        filters=RetrievalFilters(),
    )
    reranked = rerank("emergency access", result.candidates, PassthroughRerankProvider())

    original = result.candidates[0].evidence
    survived = reranked[0].evidence
    assert survived.text == original.text
    assert survived.department == original.department
    assert survived.chunk_uid == original.chunk_uid
