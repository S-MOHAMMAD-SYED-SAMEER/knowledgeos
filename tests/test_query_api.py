"""`POST /query`, end to end against a real database.

The application's real dependency is the cached `BgeEmbeddingProvider` (see
`app/api/query.py::embedding_provider`), and every test here overrides that
one FastAPI dependency to inject `FakeEmbeddingProvider` — the same pattern
`tests/test_upload_api.py` uses for settings. The application itself never
makes that substitution; `tests/test_embeddings.py` already proves nothing
in `app/` names `FakeEmbeddingProvider`, and one test below proves the
unmodified endpoint really does reach for the real, currently-unavailable
model instead.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider
from app.providers import EmbeddingError, FakeEmbeddingProvider
from app.storage import LocalStorage

from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_client(client: TestClient, embeddings: FakeEmbeddingProvider):
    """`/query` through the fake provider — injected by dependency override,
    never chosen by the application."""
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    try:
        yield client
    finally:
        client.app.dependency_overrides.clear()


def _seed(session, storage, embeddings, **overrides):
    fields = {"text": "production database access requires manager approval"}
    fields.update(overrides)
    return seed_active_version(session, storage, embeddings=embeddings, **fields)


# --- the successful case -----------------------------------------------


def test_a_successful_query_returns_200(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post(
        "/query", json={"query": "production database access"}
    )

    assert response.status_code == 200


def test_the_response_shape(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    assert set(body) == {"query", "normalized_query", "filters", "candidates", "counts"}
    assert body["candidates"]
    candidate = body["candidates"][0]
    assert set(candidate) == {
        "chunk_uid",
        "text",
        "sequence",
        "page",
        "section",
        "char_start",
        "char_end",
        "document",
        "version",
        "lexical_rank",
        "vector_rank",
        "rrf_score",
        "final_rank",
    }
    assert set(candidate["document"]) == {"id", "title", "department", "category", "tags"}
    assert set(candidate["version"]) == {"id", "version_number", "status"}
    assert set(body["counts"]) == {"vector", "lexical", "fused", "returned"}


def test_evidence_only_no_answer_no_citations_no_rerank_score(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    assert "answer" not in body
    assert "citations" not in body
    assert "rerank_score" not in str(body)


def test_storage_path_never_appears(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post(
        "/query", json={"query": "production database access"}
    )

    assert "storage_path" not in response.text
    assert "documents/" not in response.text


def test_zero_candidates_is_still_200(
    fake_client, migrated_engine: Engine
) -> None:
    response = fake_client.post(
        "/query", json={"query": "nothing has been indexed yet"}
    )

    assert response.status_code == 200
    assert response.json()["candidates"] == []


# --- validation ----------------------------------------------------------


def test_empty_query_is_422(fake_client, migrated_engine: Engine) -> None:
    assert fake_client.post("/query", json={"query": ""}).status_code == 422


def test_whitespace_only_query_is_422(fake_client, migrated_engine: Engine) -> None:
    assert (
        fake_client.post("/query", json={"query": "   \n\t  "}).status_code == 422
    )


def test_missing_query_field_is_422(fake_client, migrated_engine: Engine) -> None:
    assert fake_client.post("/query", json={}).status_code == 422


def test_a_query_over_the_limit_is_422(
    fake_client, migrated_engine: Engine
) -> None:
    response = fake_client.post("/query", json={"query": "a" * 1001})
    assert response.status_code == 422


def test_a_query_at_the_limit_is_accepted(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post("/query", json={"query": "a" * 1000})

    assert response.status_code != 422


def test_the_query_is_not_silently_truncated(
    fake_client, migrated_engine: Engine
) -> None:
    """The normalized query returned is the whole query, not a cut-down
    version of it."""
    long_query = ("widget " * 100).strip()

    response = fake_client.post("/query", json={"query": long_query})

    assert response.status_code == 200
    assert response.json()["normalized_query"] == long_query


# --- provider failure ------------------------------------------------------


def test_provider_failure_is_503_with_a_safe_body(
    client: TestClient, migrated_engine: Engine
) -> None:
    class BrokenProvider:
        dimensions = 384
        model_name = "broken"

        def embed(self, texts):
            raise EmbeddingError("the model could not be loaded")

    client.app.dependency_overrides[embedding_provider] = lambda: BrokenProvider()
    try:
        response = client.post(
            "/query", json={"query": "a secret query nobody should log"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert "/home/" not in response.text
    assert "File \"" not in response.text
    assert "a secret query nobody should log" not in response.text


def test_the_unmodified_endpoint_reaches_for_the_real_provider(
    client: TestClient, migrated_engine: Engine
) -> None:
    """Without an override, `/query` uses the real, cached
    `BgeEmbeddingProvider` — which fails in this environment because BGE
    cannot be loaded from the local cache, and that is the honest `503`,
    not a silent fallback to the fake. (`tests/test_embeddings.py` proves
    statically that no code in `app/` ever names `FakeEmbeddingProvider`;
    this proves the wiring behaviorally.)"""
    response = client.post("/query", json={"query": "anything"})

    assert response.status_code == 503
    assert "embedding model is unavailable" in response.json()["detail"]


# --- filters ---------------------------------------------------------------


def test_an_unmatched_filter_is_200_with_empty_candidates(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, department="Engineering")

    response = fake_client.post(
        "/query",
        json={
            "query": "production database access",
            "filters": {"department": "Nonexistent"},
        },
    )

    assert response.status_code == 200
    assert response.json()["candidates"] == []


def test_department_filter_actually_restricts(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(
            session,
            storage,
            embeddings,
            text="engineering policy widget",
            department="Engineering",
            title="a.txt",
        )
        _seed(
            session,
            storage,
            embeddings,
            text="sales policy widget",
            department="Sales",
            title="b.txt",
        )

    response = fake_client.post(
        "/query",
        json={"query": "policy widget", "filters": {"department": "Engineering"}},
    )

    body = response.json()
    assert body["candidates"]
    assert all(c["document"]["department"] == "Engineering" for c in body["candidates"])


def test_an_unknown_document_id_filter_matches_nothing(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post(
        "/query",
        json={"query": "production database access", "filters": {"document_id": str(uuid.uuid4())}},
    )

    assert response.status_code == 200
    assert response.json()["candidates"] == []
