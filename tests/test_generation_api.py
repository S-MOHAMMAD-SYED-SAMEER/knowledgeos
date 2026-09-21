"""`POST /query`'s generation stage, and `GET /queries/{id}`, end to end
against a real database. Extends `tests/test_query_api.py` and
`tests/test_reranking_api.py` with milestone 8's own behavior.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.providers import FakeEmbeddingProvider, PassthroughRerankProvider
from app.providers.fake_llm import FakeLLMProvider
from app.providers.llm import LLMError
from app.storage import LocalStorage

from .generation_fixtures import valid_json
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


def _override_all(client: TestClient, embeddings, llm):
    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    client.app.dependency_overrides[llm_provider] = lambda: llm


# --- the successful generation path -----------------------------------


def test_a_successful_query_returns_a_grounded_cited_answer(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        seeded = _seed(session, storage, embeddings)
        session.refresh(seeded.version)
        from sqlalchemy import select

        from app.models import Chunk

        uids = [
            row
            for row in session.execute(
                select(Chunk.chunk_uid).where(
                    Chunk.document_version_id == seeded.version.id
                )
            ).scalars().all()
        ]

    fake_llm = FakeLLMProvider(
        [valid_json(" ".join(f"Fact [{u}]." for u in uids), uids)]
    )
    _override_all(client, embeddings, fake_llm)
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert body["citation_valid"] is True
    assert body["grounded"] is True
    assert set(body["citations"]) == set(uids)
    assert body["model"] == fake_llm.model_name
    assert body["prompt_version"] == "knowledge_answer_v1"
    assert body["query_id"]
    assert body["answer_id"]


def test_the_response_shape_includes_generation_fields(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    """A `document_id` filter matching nothing indexed guarantees zero
    candidates deterministically -- unlike an "unrelated" query text,
    which the fake embedding provider's deterministic-but-meaningless
    vectors can still return a nearest neighbour for."""
    import uuid

    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    _override_all(client, embeddings, FakeLLMProvider([]))  # must never be called
    try:
        body = client.post(
            "/query",
            json={
                "query": "production database access",
                "filters": {"document_id": str(uuid.uuid4())},
            },
        ).json()
    finally:
        client.app.dependency_overrides.clear()

    assert body["candidates"] == []

    for field in (
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
    ):
        assert field in body, field


def test_candidates_carry_the_selected_flag(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    fake_llm = FakeLLMProvider([])
    _override_all(client, embeddings, fake_llm)
    try:
        # Force a real retrieval hit with a real citation-carrying answer.
        with Session(migrated_engine) as session:
            from sqlalchemy import select

            from app.models import Chunk

            uids = session.execute(select(Chunk.chunk_uid)).scalars().all()
        fake_llm2 = FakeLLMProvider(
            [valid_json(" ".join(f"Fact [{u}]." for u in uids), uids)]
        )
        client.app.dependency_overrides[llm_provider] = lambda: fake_llm2

        body = client.post(
            "/query", json={"query": "production database access"}
        ).json()
    finally:
        client.app.dependency_overrides.clear()

    assert body["candidates"]
    for candidate in body["candidates"]:
        assert "selected" in candidate
        assert isinstance(candidate["selected"], bool)


# --- abstention -----------------------------------------------------------


def test_zero_candidates_abstains_with_200_and_never_calls_the_model(
    client: TestClient, migrated_engine: Engine
) -> None:
    fake_llm = FakeLLMProvider([])  # would raise LLMError if ever called
    _override_all(client, FakeEmbeddingProvider(), fake_llm)
    try:
        response = client.post(
            "/query", json={"query": "nothing has been indexed at all"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["citations"] == []
    assert fake_llm.call_count == 0


def test_an_abstained_answer_has_no_unsupported_claims_in_its_text(
    client: TestClient, migrated_engine: Engine
) -> None:
    _override_all(client, FakeEmbeddingProvider(), FakeLLMProvider([]))
    try:
        response = client.post(
            "/query", json={"query": "nothing has been indexed at all"}
        )
    finally:
        client.app.dependency_overrides.clear()

    body = response.json()
    assert "[" not in body["answer"]  # no citation markers in an abstention


# --- provider failure -----------------------------------------------------


def test_llm_provider_failure_is_503_with_a_safe_body(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings, text="SUPERSECRET production access policy")

    class BrokenLLM:
        model_name = "broken"

        def complete(self, *, system, user, max_tokens):
            raise LLMError("the model could not be reached")

    _override_all(client, embeddings, BrokenLLM())
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


def test_the_unmodified_endpoint_reaches_for_the_real_gemini_provider(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    """With embedding and reranking overridden (so retrieval succeeds) but
    generation left unmodified, `/query` reaches for the real, cached
    `GeminiLLMProvider` -- which fails in this environment because no
    Gemini credential is configured, and that is the honest `503`, not a
    silent fallback to the fake."""
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "language model is unavailable" in response.json()["detail"]


def test_invalid_citation_from_the_model_is_502(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    fake_llm = FakeLLMProvider([valid_json("Fact [" + "f" * 32 + "].", ["f" * 32])])
    _override_all(client, embeddings, fake_llm)
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "could not be validated" in response.json()["detail"]
    assert "Traceback" not in response.text


def test_malformed_json_from_the_model_is_502(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        _seed(session, storage, embeddings)

    fake_llm = FakeLLMProvider(["not valid json at all {"])
    _override_all(client, embeddings, fake_llm)
    try:
        response = client.post(
            "/query", json={"query": "production database access"}
        )
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 502


# --- GET /queries/{id} ------------------------------------------------


def test_get_queries_returns_the_persisted_query_and_answer(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        from sqlalchemy import select

        from app.models import Chunk

        _seed(session, storage, embeddings)
        uids = session.execute(select(Chunk.chunk_uid)).scalars().all()

    fake_llm = FakeLLMProvider(
        [valid_json(" ".join(f"Fact [{u}]." for u in uids), uids)]
    )
    _override_all(client, embeddings, fake_llm)
    try:
        post_body = client.post(
            "/query", json={"query": "production database access"}
        ).json()
        query_id = post_body["query_id"]

        get_response = client.get(f"/queries/{query_id}")
    finally:
        client.app.dependency_overrides.clear()

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["id"] == query_id
    assert body["answer"] is not None
    assert body["answer"]["abstained"] is False
    assert body["retrieved_chunks"]
    assert all("selected" in c for c in body["retrieved_chunks"])


def test_get_queries_404_for_an_unknown_id(client: TestClient, migrated_engine: Engine) -> None:
    import uuid

    response = client.get(f"/queries/{uuid.uuid4()}")
    assert response.status_code == 404


def test_get_queries_404_for_a_malformed_id(client: TestClient, migrated_engine: Engine) -> None:
    response = client.get("/queries/not-a-uuid")
    assert response.status_code == 404
