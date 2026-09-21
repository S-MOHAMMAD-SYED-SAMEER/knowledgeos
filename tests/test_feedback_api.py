"""`POST /answers/{answer_id}/feedback` (milestone 10, §11) — the
specification's own feedback endpoint, writing to its own `feedback`
table (§5)."""

import uuid

import pytest
import sqlalchemy.exc
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.generation.generator import generate_answer
from app.generation.persistence import persist_query
from app.generation.prompt import load_prompt
from app.models import Feedback
from app.parsing import normalize
from app.providers import FakeEmbeddingProvider, PassthroughRerankProvider
from app.providers.fake_llm import FakeLLMProvider
from app.reranking.pipeline import rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from app.storage import LocalStorage

from .generation_fixtures import cited_answer_text, valid_json
from .retrieval_fixtures import seed_active_version

PROMPT = load_prompt()


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _persisted_answer_id(session, storage, embeddings) -> uuid.UUID:
    """A real, persisted answer to attach feedback to — through the real
    retrieve/rerank/generate/persist pipeline, the same way
    `tests/test_query_persistence.py` builds one."""
    seed_active_version(
        session, storage, embeddings=embeddings,
        text="production database access requires manager approval and expires after seven days",
    )
    normalized = normalize("production database access")
    vector = embeddings.embed([normalized])[0]
    result = retrieve(
        session, normalized_text=normalized, query_vector=vector, filters=RetrievalFilters(),
    )
    candidates = rerank(normalized, result.candidates, PassthroughRerankProvider())
    uids = [c.evidence.chunk_uid for c in candidates]
    fake_llm = FakeLLMProvider([valid_json(cited_answer_text(*uids), uids)])
    answer = generate_answer(
        query_text=normalized, candidates=candidates, llm=fake_llm,
        abstention_threshold=None, max_tokens=200, prompt=PROMPT,
    )
    _, answer_row = persist_query(
        session, query_text="production database access", normalized_text=normalized,
        filters={}, candidates=candidates, answer=answer,
    )
    return answer_row.id


def test_helpful_feedback_is_accepted_and_persisted(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    response = client.post(
        f"/answers/{answer_id}/feedback", json={"rating": "helpful"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["rating"] == "helpful"
    assert body["reason"] is None
    assert body["answer_id"] == str(answer_id)

    with Session(migrated_engine) as session:
        rows = session.execute(select(Feedback).where(Feedback.answer_id == answer_id)).scalars().all()
        assert len(rows) == 1
        assert rows[0].rating == "helpful"


def test_not_helpful_feedback_is_accepted(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    response = client.post(
        f"/answers/{answer_id}/feedback",
        json={"rating": "not_helpful", "reason": "missed the second approver requirement"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["rating"] == "not_helpful"
    assert body["reason"] == "missed the second approver requirement"


def test_an_invalid_rating_is_422(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    response = client.post(
        f"/answers/{answer_id}/feedback", json={"rating": "sort of helpful"}
    )
    assert response.status_code == 422


def test_feedback_for_an_unknown_answer_is_404(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.post(
        f"/answers/{uuid.uuid4()}/feedback", json={"rating": "helpful"}
    )
    assert response.status_code == 404


def test_reason_is_optional(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    response = client.post(
        f"/answers/{answer_id}/feedback", json={"rating": "helpful"}
    )
    assert response.status_code == 201
    assert response.json()["reason"] is None


def test_a_reason_over_the_limit_is_422(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    from app.models import MAX_REASON_LENGTH

    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    response = client.post(
        f"/answers/{answer_id}/feedback",
        json={"rating": "helpful", "reason": "a" * (MAX_REASON_LENGTH + 1)},
    )
    assert response.status_code == 422


def test_multiple_feedback_rows_may_exist_for_one_answer(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    """Append-only: the specification states no uniqueness rule, and a
    second opinion on the same answer is not an error."""
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    r1 = client.post(f"/answers/{answer_id}/feedback", json={"rating": "helpful"})
    r2 = client.post(f"/answers/{answer_id}/feedback", json={"rating": "not_helpful"})
    assert r1.status_code == 201
    assert r2.status_code == 201

    with Session(migrated_engine) as session:
        rows = session.execute(select(Feedback).where(Feedback.answer_id == answer_id)).scalars().all()
        assert len(rows) == 2


def test_the_database_rejects_an_invalid_rating_bypassing_the_api(
    migrated_engine: Engine, storage, embeddings
) -> None:
    """The CHECK constraint (migration 0007), not just the pydantic layer
    -- a write that bypassed the API still cannot store a bad value."""
    from app.models import Answer, Feedback as FeedbackModel

    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)
        assert session.get(Answer, answer_id) is not None

        session.add(FeedbackModel(answer_id=answer_id, rating="excellent"))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()


def test_feedback_response_shape(
    client: TestClient, migrated_engine: Engine, storage, embeddings
) -> None:
    with Session(migrated_engine) as session:
        answer_id = _persisted_answer_id(session, storage, embeddings)

    body = client.post(f"/answers/{answer_id}/feedback", json={"rating": "helpful"}).json()
    assert set(body) == {"id", "answer_id", "rating", "reason", "created_at"}


def test_feedback_endpoint_carries_no_traceback_on_error(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.post(
        f"/answers/{uuid.uuid4()}/feedback", json={"rating": "helpful"}
    )
    assert "Traceback" not in response.text
    assert "File \"" not in response.text
