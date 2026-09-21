"""`POST /query`'s reranking stage, end to end against a real database.

Extends `tests/test_query_api.py`'s coverage with the milestone 6-specific
behavior: `rerank_score`/`fusion_rank`/`final_rank` semantics, reranker
failure, and the honest proof that the unmodified endpoint reaches for the
real cross-encoder rather than silently falling back to the passthrough.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, rerank_provider
from app.providers import FakeEmbeddingProvider, FakeRerankProvider
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.providers.reranker import Candidate, RerankError
from app.storage import LocalStorage

from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _seed(session, storage, embeddings, **overrides):
    fields = {"text": "production database access requires manager approval"}
    fields.update(overrides)
    return seed_active_version(session, storage, embeddings=embeddings, **fields)


@pytest.fixture
def passthrough_client(client: TestClient, embeddings: FakeEmbeddingProvider):
    """`/query` with the fake embedding provider and the real,
    order-preserving passthrough reranker — so `final_rank` is provably
    still 1..N over the fusion order, and `fusion_rank == final_rank`."""
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    try:
        yield client
    finally:
        client.app.dependency_overrides.clear()


@pytest.fixture
def reordering_client(client: TestClient, embeddings: FakeEmbeddingProvider):
    """`/query` with the fake embedding provider and the deterministic fake
    reranker — genuinely capable of reordering, so `final_rank` and
    `fusion_rank` can provably differ."""
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: FakeRerankProvider()
    try:
        yield client
    finally:
        client.app.dependency_overrides.clear()


# --- final_rank / fusion_rank semantics --------------------------------


def test_final_rank_matches_fusion_rank_under_passthrough(
    passthrough_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = passthrough_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    for candidate in body["candidates"]:
        assert candidate["final_rank"] == candidate["fusion_rank"]


def test_final_rank_is_contiguous_from_one(
    passthrough_client, migrated_engine: Engine, storage, embeddings
) -> None:
    for i in range(5):
        with Session(migrated_engine) as session:
            _seed(session, storage, embeddings, text=f"widget candidate {i}", title=f"d{i}.txt")

    body = passthrough_client.post("/query", json={"query": "widget"}).json()

    ranks = [c["final_rank"] for c in body["candidates"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_rerank_score_is_present_and_numeric(
    passthrough_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = passthrough_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    for candidate in body["candidates"]:
        assert isinstance(candidate["rerank_score"], (int, float))


def test_final_rank_can_differ_from_fusion_rank_when_reranking_reorders(
    reordering_client, migrated_engine: Engine, storage, embeddings
) -> None:
    for i in range(6):
        with Session(migrated_engine) as session:
            _seed(session, storage, embeddings, text=f"widget entry {i}", title=f"d{i}.txt")

    body = reordering_client.post("/query", json={"query": "widget"}).json()

    final_ranks = [c["final_rank"] for c in body["candidates"]]
    fusion_ranks = [c["fusion_rank"] for c in body["candidates"]]

    assert final_ranks == list(range(1, len(final_ranks) + 1))
    # Both are permutations of 1..N, but reranking by content -- rather
    # than preserving position -- genuinely reorders this six-chunk corpus:
    # `fusion_rank`, read off in `final_rank` order, is not 1, 2, 3, ...
    assert sorted(fusion_ranks) == list(range(1, len(fusion_ranks) + 1))
    assert fusion_ranks != final_ranks


# --- reranker failure -----------------------------------------------------


def test_reranker_failure_is_503_with_a_safe_body(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, text="SUPERSECRET production access policy")

    class BrokenReranker:
        model_name = "broken"

        def rerank(self, query, candidates):
            raise RerankError("the model could not be loaded")

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: BrokenReranker()
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert "/home/" not in response.text
    assert 'File "' not in response.text
    assert "SUPERSECRET" not in response.text


def test_reranker_failure_does_not_return_m5_order_as_if_reranked(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    """A 503 body carries no candidates at all -- never a silent
    "unreranked but labelled as reranked" response."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    class BrokenReranker:
        model_name = "broken"

        def rerank(self, query, candidates):
            raise RerankError("inference failed")

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: BrokenReranker()
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    body = response.json()
    assert "candidates" not in body


def test_zero_candidates_never_reaches_a_failing_reranker(
    client: TestClient, migrated_engine: Engine, embeddings
) -> None:
    """Nothing retrieved means the reranker is never called -- so even a
    provider that always raises must not turn an empty result into a 503."""

    class AlwaysBroken:
        model_name = "always-broken"

        def rerank(self, query, candidates):
            raise RerankError("must never be called")

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: AlwaysBroken()
    try:
        response = client.post(
            "/query", json={"query": "nothing has been indexed at all"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["candidates"] == []


# --- honest, unmodified defaults ------------------------------------------


def test_the_unmodified_endpoint_reaches_for_the_real_reranker(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    """With only the embedding provider overridden (so retrieval succeeds),
    the reranking stage still uses the real, cached `CrossEncoderRerankProvider`
    by default -- which fails in this environment because the model cannot
    be loaded from the local cache, and that is the honest `503`, not a
    silent fallback to the passthrough."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "reranking model is unavailable" in response.json()["detail"]


# --- identity mapping through the API --------------------------------------


def test_every_candidate_score_traces_back_to_its_own_chunk(
    reordering_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """Cross-checks the API's scores against calling the fake reranker
    directly on the same texts -- proving the mapping from provider output
    back to chunk_uid is exact, not merely "some permutation"."""
    with Session(migrated_engine) as session:
        seeded = _seed(session, storage, embeddings, text="alpha widget", title="a.txt")
        _seed(session, storage, embeddings, text="beta widget", title="b.txt")

    body = reordering_client.post("/query", json={"query": "widget"}).json()

    fake = FakeRerankProvider()
    for candidate in body["candidates"]:
        expected = fake.rerank(
            "widget", [Candidate(id=candidate["chunk_uid"], text=candidate["text"])]
        )[0].score
        assert candidate["rerank_score"] == pytest.approx(expected)
