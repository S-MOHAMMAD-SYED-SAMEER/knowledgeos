"""`POST /query`, end to end against a real database.

The application's real dependencies are the cached `BgeEmbeddingProvider`
(see `app/api/query.py::embedding_provider`), the cached
`CrossEncoderRerankProvider` (`app/api/query.py::rerank_provider`), and, as
of milestone 8, the cached `GeminiLLMProvider`
(`app/api/query.py::llm_provider`). Every test here overrides all three
FastAPI dependencies to inject `FakeEmbeddingProvider`, a passthrough or
deterministic reranker, and a generation test double — the same pattern
`tests/test_upload_api.py` uses for settings. The application itself never
makes any of these substitutions; `tests/test_embeddings.py`,
`tests/test_reranking_scope.py` and `tests/test_llm_provider.py` prove
nothing in `app/` names a fake or passthrough provider, and one test below
proves the unmodified endpoint really does reach for the real,
currently-unavailable models instead.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.providers import EmbeddingError, FakeEmbeddingProvider
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.providers.reranker import RerankError
from app.storage import LocalStorage

from .generation_fixtures import AutoCitingLLMProvider
from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def reranker() -> PassthroughRerankProvider:
    """The passthrough, not a test-only fake: milestone 6's own SPEC-named
    "reranking disabled" configuration, used here so these tests exercise
    the wiring without depending on the real cross-encoder's ordering."""
    return PassthroughRerankProvider()


@pytest.fixture
def fake_client(
    client: TestClient,
    embeddings: FakeEmbeddingProvider,
    reranker: PassthroughRerankProvider,
):
    """`/query` through the fake embedding provider, the passthrough
    reranker, and a generation test double that cites whatever evidence it
    was actually given — all three injected by dependency override, none
    chosen by the application."""
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: reranker
    client.app.dependency_overrides[llm_provider] = lambda: AutoCitingLLMProvider()
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

    assert set(body) == {
        "query",
        "normalized_query",
        "filters",
        "candidates",
        "counts",
        "query_id",
        "answer_id",
        "answer",
        "abstained",
        "citations",
        "citation_valid",
        "grounded",
        "grounding_detail",
        "model",
        "prompt_version",
    }
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
        "fusion_rank",
        "rerank_score",
        "final_rank",
        "selected",
    }
    assert set(candidate["document"]) == {"id", "title", "department", "category", "tags"}
    assert set(candidate["version"]) == {"id", "version_number", "status"}
    assert set(body["counts"]) == {"vector", "lexical", "fused", "returned"}


def test_the_response_now_carries_the_generated_answer_and_its_validation(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """Milestone 6 added `rerank_score` and `final_rank` (post-rerank rank)
    and deliberately added nothing belonging to generation. Milestone 8 is
    what finishes `/query` — this test used to assert the opposite
    (`answer`/`citations`/`selected` absent); it now asserts milestone 8's
    own fields are genuinely present and well-formed, the same inversion
    `tests/test_retrieval_scope.py` already applies to its own guards when
    a later milestone legitimately arrives."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    assert "answer" in body and isinstance(body["answer"], str)
    assert "citations" in body and isinstance(body["citations"], list)
    assert "abstained" in body and isinstance(body["abstained"], bool)
    assert "citation_valid" in body and body["citation_valid"] is True
    assert "grounded" in body
    assert all("selected" in c for c in body["candidates"])
    assert body["abstained"] is False  # real evidence was seeded and cited


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


# --- observability (milestone 9, D5) ----------------------------------------


def test_a_successful_query_records_measured_stage_timing(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """The response shape itself is unchanged (`test_the_response_shape`
    above) -- these fields are written straight to the `queries` row, not
    the API response."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    body = fake_client.post(
        "/query", json={"query": "production database access"}
    ).json()

    from app.models import Query

    with Session(migrated_engine) as session:
        row = session.get(Query, uuid.UUID(body["query_id"]))
        assert row.retrieval_ms is not None and row.retrieval_ms >= 0
        assert row.rerank_ms is not None and row.rerank_ms >= 0
        assert row.llm_ms is not None and row.llm_ms >= 0
        assert row.total_ms is not None
        assert row.total_ms >= row.retrieval_ms + row.rerank_ms


def test_a_pre_llm_abstention_records_no_generation_latency(
    fake_client, migrated_engine: Engine
) -> None:
    """Zero candidates -- `generate_answer` never calls the provider, so
    there is no real generation latency to report."""
    body = fake_client.post(
        "/query", json={"query": "nothing has been indexed yet"}
    ).json()

    from app.models import Query

    with Session(migrated_engine) as session:
        row = session.get(Query, uuid.UUID(body["query_id"]))
        assert row.llm_ms is None
        assert row.retrieval_ms is not None
        assert row.rerank_ms is not None
        assert row.total_ms == row.retrieval_ms + row.rerank_ms


def test_unconfigured_pricing_leaves_cost_null_without_failing_the_query(
    fake_client, migrated_engine: Engine, storage, embeddings
) -> None:
    """D7: unconfigured pricing must never fail a live query -- only leave
    its cost unmeasured. This build's default pricing table is empty."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    response = fake_client.post(
        "/query", json={"query": "production database access"}
    )
    assert response.status_code == 200

    from app.models import Query

    with Session(migrated_engine) as session:
        row = session.get(Query, uuid.UUID(response.json()["query_id"]))
        assert row.cost_usd is None


def test_configured_pricing_computes_a_real_cost(
    fake_client, migrated_engine: Engine, storage, embeddings, monkeypatch
) -> None:
    from app.config import get_settings

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    llm = AutoCitingLLMProvider()
    monkeypatch.setenv(
        "KNOWLEDGEOS_LLM_PRICING_USD_PER_MILLION_TOKENS",
        f'{{"{llm.model_name}": {{"input": 1.0, "output": 2.0}}}}',
    )
    get_settings.cache_clear()
    try:
        response = fake_client.post(
            "/query", json={"query": "production database access"}
        )
        assert response.status_code == 200

        from app.models import Query

        with Session(migrated_engine) as session:
            row = session.get(Query, uuid.UUID(response.json()["query_id"]))
            assert row.cost_usd is not None
            assert row.cost_usd >= 0
    finally:
        get_settings.cache_clear()
