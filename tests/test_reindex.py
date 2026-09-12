"""`POST /documents/{id}/reindex`, end to end.

The endpoint exists to re-run a document that is already answering questions —
after a parser fix, or a model change. So the assertions are about what it must
*not* do: it creates no new version, it produces no duplicate chunks, and it
refuses rather than guessing when there is nothing active to re-run or when a
job is already in flight on that version.

The document reaches `active` the only way it can: by being ingested for real,
by the runner, against a real database and a real file on disk.
"""

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.ingestion.runner import run_pending
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus
from app.providers import FakeEmbeddingProvider
from app.storage import get_storage

from .fixtures import words

TEXT = words(120).encode()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5, max_attempts=3
    )


def _upload(client: TestClient, content: bytes = TEXT, name: str = "access-sop.txt"):
    response = client.post(
        "/documents",
        files={"file": (name, io.BytesIO(content), "text/plain")},
        data={"title": "Engineering Access SOP"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _drain(engine: Engine, settings: Settings, limit: int = 10) -> list:
    """Run the runner until there is nothing outstanding."""
    return run_pending(
        sessions=sessionmaker(engine),
        storage=get_storage(),
        settings=settings,
        embeddings=FakeEmbeddingProvider(),
        limit=limit,
    )


@pytest.fixture
def indexed(client: TestClient, migrated_engine: Engine, settings: Settings):
    """An uploaded document, ingested all the way to `active`."""
    uploaded = _upload(client)
    _drain(migrated_engine, settings)

    with Session(migrated_engine) as session:
        version = session.execute(select(DocumentVersion)).scalar_one()
        assert version.status == "active", "the fixture did not reach active"
    return uploaded


def _chunks(engine: Engine, version_id) -> list[Chunk]:
    with Session(engine) as session:
        return list(
            session.execute(
                select(Chunk)
                .where(Chunk.document_version_id == version_id)
                .order_by(Chunk.sequence)
            ).scalars()
        )


# --- the accepted case ------------------------------------------------------


def test_reindexing_an_active_document_is_accepted(
    client: TestClient, indexed
) -> None:
    response = client.post(f"/documents/{indexed['document']['id']}/reindex")

    assert response.status_code == 202, response.text


def test_the_response_carries_the_document_version_and_job(
    client: TestClient, indexed
) -> None:
    body = client.post(f"/documents/{indexed['document']['id']}/reindex").json()

    assert set(body) == {"document", "version", "job"}
    assert body["job"]["status"] == "queued"
    assert body["job"]["attempts"] == 0


def test_reindexing_targets_the_version_that_is_active(
    client: TestClient, indexed
) -> None:
    body = client.post(f"/documents/{indexed['document']['id']}/reindex").json()

    assert body["version"]["id"] == indexed["version"]["id"]
    assert body["version"]["status"] == "active"


def test_reindexing_creates_no_new_version(
    client: TestClient, indexed, migrated_engine: Engine
) -> None:
    client.post(f"/documents/{indexed['document']['id']}/reindex")

    with Session(migrated_engine) as session:
        versions = session.execute(select(DocumentVersion)).scalars().all()
    assert len(versions) == 1


def test_reindexing_queues_a_second_job_for_the_same_version(
    client: TestClient, indexed, migrated_engine: Engine
) -> None:
    client.post(f"/documents/{indexed['document']['id']}/reindex")

    with Session(migrated_engine) as session:
        jobs = session.execute(select(IngestionJob)).scalars().all()
    assert len(jobs) == 2
    assert {job.document_version_id for job in jobs} == {
        uuid.UUID(indexed["version"]["id"])
    }


def test_the_response_never_carries_a_storage_path(
    client: TestClient, indexed
) -> None:
    response = client.post(f"/documents/{indexed['document']['id']}/reindex")

    assert "storage_path" not in response.text
    assert "documents/" not in response.text


# --- what a re-index actually does ------------------------------------------


def test_a_reindexed_document_returns_to_ready(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    body = client.post(f"/documents/{indexed['document']['id']}/reindex").json()
    _drain(migrated_engine, settings)

    with Session(migrated_engine) as session:
        job = session.get(IngestionJob, uuid.UUID(body["job"]["id"]))
        assert job.status == JobStatus.READY


def test_reindexing_writes_the_same_chunk_identifiers(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    version_id = uuid.UUID(indexed["version"]["id"])
    before = [chunk.chunk_uid for chunk in _chunks(migrated_engine, version_id)]
    assert before

    client.post(f"/documents/{indexed['document']['id']}/reindex")
    _drain(migrated_engine, settings)

    after = [chunk.chunk_uid for chunk in _chunks(migrated_engine, version_id)]
    assert after == before


def test_reindexing_creates_no_duplicate_chunks(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    version_id = uuid.UUID(indexed["version"]["id"])
    before = len(_chunks(migrated_engine, version_id))

    client.post(f"/documents/{indexed['document']['id']}/reindex")
    _drain(migrated_engine, settings)

    assert len(_chunks(migrated_engine, version_id)) == before


def test_reindexing_leaves_every_chunk_embedded(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    version_id = uuid.UUID(indexed["version"]["id"])

    client.post(f"/documents/{indexed['document']['id']}/reindex")
    _drain(migrated_engine, settings)

    chunks = _chunks(migrated_engine, version_id)
    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)


def test_the_version_is_still_the_active_one_afterwards(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    client.post(f"/documents/{indexed['document']['id']}/reindex")
    _drain(migrated_engine, settings)

    with Session(migrated_engine) as session:
        version = session.get(
            DocumentVersion, uuid.UUID(indexed["version"]["id"])
        )
    assert version.status == "active"


# --- the refusals -----------------------------------------------------------


def test_an_unknown_document_is_a_404(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.post(f"/documents/{uuid.uuid4()}/reindex")

    assert response.status_code == 404


def test_a_document_with_no_active_version_is_a_409(
    client: TestClient, migrated_engine: Engine
) -> None:
    """Uploaded but never indexed: the only version is still a draft."""
    uploaded = _upload(client)

    response = client.post(f"/documents/{uploaded['document']['id']}/reindex")

    assert response.status_code == 409


def test_the_conflict_says_there_is_nothing_active(
    client: TestClient, migrated_engine: Engine
) -> None:
    uploaded = _upload(client)

    response = client.post(f"/documents/{uploaded['document']['id']}/reindex")

    assert "active" in response.json()["detail"]


def test_a_second_reindex_while_one_is_outstanding_is_a_409(
    client: TestClient, indexed
) -> None:
    first = client.post(f"/documents/{indexed['document']['id']}/reindex")
    assert first.status_code == 202

    second = client.post(f"/documents/{indexed['document']['id']}/reindex")

    assert second.status_code == 409


def test_a_refused_reindex_queues_nothing(
    client: TestClient, indexed, migrated_engine: Engine
) -> None:
    client.post(f"/documents/{indexed['document']['id']}/reindex")
    client.post(f"/documents/{indexed['document']['id']}/reindex")

    with Session(migrated_engine) as session:
        jobs = session.execute(select(IngestionJob)).scalars().all()
    assert len(jobs) == 2


def test_reindexing_is_accepted_again_once_the_job_has_finished(
    client: TestClient, indexed, migrated_engine: Engine, settings: Settings
) -> None:
    client.post(f"/documents/{indexed['document']['id']}/reindex")
    _drain(migrated_engine, settings)

    again = client.post(f"/documents/{indexed['document']['id']}/reindex")

    assert again.status_code == 202


def test_a_failed_job_does_not_block_a_reindex(
    client: TestClient, indexed, migrated_engine: Engine
) -> None:
    """`failed` is terminal: it must not wedge the document for ever."""
    version_id = uuid.UUID(indexed["version"]["id"])
    with Session(migrated_engine) as session:
        session.add(
            IngestionJob(document_version_id=version_id, status=JobStatus.FAILED)
        )
        session.commit()

    response = client.post(f"/documents/{indexed['document']['id']}/reindex")

    assert response.status_code == 202


def test_no_error_body_carries_a_traceback_or_a_path(
    client: TestClient, migrated_engine: Engine
) -> None:
    uploaded = _upload(client)

    conflict = client.post(f"/documents/{uploaded['document']['id']}/reindex")
    missing = client.post(f"/documents/{uuid.uuid4()}/reindex")

    for response in (conflict, missing):
        assert "Traceback" not in response.text
        assert "/home/" not in response.text
        assert "documents/" not in response.text
