"""The indexing stage, against a real PostgreSQL.

What this milestone adds is the last stage of ingestion, and the assertions
here are about the things that stage promises:

* a job goes `indexing` → `ready`, and only then;
* its version goes `draft` → `active`, and the previous active one becomes
  `superseded`, in the same transaction;
* every chunk gets a vector of the declared width;
* `tsv` is maintained by PostgreSQL, not by this code;
* a failure leaves **none** of it — no vectors, no promotion — and costs an
  attempt that survives the rollback.

The embeddings come from `FakeEmbeddingProvider` throughout. The specification
permits the fake for unit tests, and nothing here measures retrieval quality:
these tests are about transactions, state transitions and shapes, all of which
a deterministic vector exercises exactly as well as a real one.
"""

import io
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.indexing import IndexingFailed, chunks_to_embed, generate_embeddings, texts_of
from app.ingestion import service
from app.ingestion.pipeline import StageFailed, run_indexing, run_job
from app.ingestion.runner import process_one
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus
from app.providers import DIMENSIONS, EmbeddingError, FakeEmbeddingProvider
from app.storage import LocalStorage

from .fixtures import words


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5, max_attempts=3
    )


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _uploaded(
    session: Session,
    storage: LocalStorage,
    content: bytes | None = None,
    filename: str = "notes.txt",
    document_id: uuid.UUID | None = None,
):
    """A version with its file really in storage, exactly as upload leaves it."""
    document_id = document_id or uuid.uuid4()
    version_id = uuid.uuid4()
    extension = "." + filename.rsplit(".", 1)[-1]
    key = service.storage_key(document_id, version_id, extension)
    storage.write(key, io.BytesIO(content if content is not None else words(120).encode()))

    result = service.create_document_with_version(
        session,
        document_id=document_id,
        title=filename,
        department=None,
        category=None,
        tags=[],
        original_filename=filename,
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    return result


def _second_version(session: Session, storage: LocalStorage, document_id, content: bytes):
    """Another version of an existing document, as a second upload leaves it."""
    version_id = uuid.uuid4()
    key = service.storage_key(document_id, version_id, ".txt")
    storage.write(key, io.BytesIO(content))

    result = service.add_version(
        session,
        document_id=document_id,
        original_filename="notes.txt",
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    return result


def _parsed(session, storage, settings, **kwargs):
    """A version taken through milestone 3, so it is waiting at `indexing`."""
    result = _uploaded(session, storage, **kwargs)
    run_job(session, result.job, storage, settings)
    session.commit()
    assert result.job.status == JobStatus.INDEXING
    return result


# --- the successful path ----------------------------------------------------


def test_indexing_takes_a_job_from_indexing_to_ready(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)

    run_indexing(session, result.job, embeddings)
    session.commit()

    assert result.job.status == JobStatus.READY


def test_indexing_takes_the_version_from_draft_to_active(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)
    assert result.version.status == "draft"

    run_indexing(session, result.job, embeddings)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, result.version.id).status == "active"


def test_the_previous_active_version_becomes_superseded(
    session, storage, settings, embeddings
) -> None:
    """One document, two versions, indexed in order."""
    first = _parsed(session, storage, settings)
    run_indexing(session, first.job, embeddings)
    session.commit()

    second = _second_version(
        session, storage, first.document.id, words(120, prefix="b").encode()
    )
    run_job(session, second.job, storage, settings)
    session.commit()
    run_indexing(session, second.job, embeddings)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, first.version.id).status == "superseded"
    assert session.get(DocumentVersion, second.version.id).status == "active"


def test_exactly_one_version_is_active_at_a_time(
    session, storage, settings, embeddings
) -> None:
    first = _parsed(session, storage, settings)
    run_indexing(session, first.job, embeddings)
    session.commit()

    second = _second_version(
        session, storage, first.document.id, words(120, prefix="b").encode()
    )
    run_job(session, second.job, storage, settings)
    session.commit()
    run_indexing(session, second.job, embeddings)
    session.commit()

    active = session.execute(
        select(DocumentVersion).where(
            DocumentVersion.document_id == first.document.id,
            DocumentVersion.status == "active",
        )
    ).scalars().all()
    assert len(active) == 1


# --- what lands in the columns ----------------------------------------------


def test_every_chunk_gets_a_vector(session, storage, settings, embeddings) -> None:
    result = _parsed(session, storage, settings)

    run_indexing(session, result.job, embeddings)
    session.commit()
    session.expire_all()

    chunks = chunks_to_embed(session, result.version.id)
    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)


def test_the_vectors_are_the_declared_width(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)

    run_indexing(session, result.job, embeddings)
    session.commit()
    session.expire_all()

    for chunk in chunks_to_embed(session, result.version.id):
        assert len(chunk.embedding) == DIMENSIONS


def test_a_vector_survives_the_round_trip_through_postgresql(
    session, storage, settings, embeddings
) -> None:
    """What comes back is what was sent, not a re-quantized approximation."""
    result = _parsed(session, storage, settings)
    chunks = chunks_to_embed(session, result.version.id)
    expected = embeddings.embed([chunks[0].text])[0]

    run_indexing(session, result.job, embeddings)
    session.commit()
    session.expire_all()

    stored = chunks_to_embed(session, result.version.id)[0].embedding
    assert [float(value) for value in stored] == pytest.approx(expected)


def test_the_full_text_vector_is_populated_by_postgresql(
    session, storage, settings
) -> None:
    """`tsv` is filled in without indexing, because the column is generated."""
    result = _parsed(session, storage, settings)
    session.expire_all()

    rows = session.execute(
        text("SELECT tsv FROM chunks WHERE document_version_id = :version"),
        {"version": result.version.id},
    ).scalars().all()

    assert rows
    assert all(row is not None and row != "" for row in rows)


def test_the_full_text_vector_tracks_the_text(session, storage, settings) -> None:
    """Change the text, and PostgreSQL recomputes. Nothing here writes `tsv`."""
    _parsed(session, storage, settings, content=b"alpha beta gamma")

    before = session.execute(text("SELECT tsv FROM chunks LIMIT 1")).scalar_one()
    session.execute(text("UPDATE chunks SET text = 'entirely different wording'"))
    session.flush()
    after = session.execute(text("SELECT tsv FROM chunks LIMIT 1")).scalar_one()
    session.rollback()

    assert before != after
    assert "differ" in after


def test_the_full_text_vector_cannot_be_written_to(
    session, storage, settings
) -> None:
    """A generated column refuses a write, which is the guarantee wanted."""
    import sqlalchemy

    _parsed(session, storage, settings)

    with pytest.raises(sqlalchemy.exc.DatabaseError):
        session.execute(text("UPDATE chunks SET tsv = to_tsvector('english', 'x')"))
    session.rollback()


def test_the_full_text_column_has_a_gin_index(session) -> None:
    definition = session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_tsv'")
    ).scalar_one()

    assert "USING gin" in definition
    assert "tsv" in definition


def test_there_is_no_index_on_the_embedding_column(session) -> None:
    """The specification forbids one until a measured latency number exists."""
    definitions = session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks'")
    ).scalars().all()

    assert not any("embedding" in definition for definition in definitions)
    assert not any(
        kind in definition.lower()
        for definition in definitions
        for kind in ("hnsw", "ivfflat")
    )


# --- failure ----------------------------------------------------------------


class _BrokenProvider:
    """A provider that raises where the real one would return vectors."""

    dimensions = DIMENSIONS
    model_name = "broken"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("the model could not be loaded")


class _WrongWidthProvider:
    """A provider whose vectors are the wrong shape."""

    dimensions = DIMENSIONS
    model_name = "wrong-width"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * (DIMENSIONS - 1) for _ in texts]


def test_a_failing_provider_fails_the_stage(session, storage, settings) -> None:
    result = _parsed(session, storage, settings)

    with pytest.raises(StageFailed) as raised:
        run_indexing(session, result.job, _BrokenProvider())

    assert raised.value.stage == JobStatus.INDEXING


def test_a_failure_leaves_the_version_a_draft(session, storage, settings) -> None:
    result = _parsed(session, storage, settings)
    version_id = result.version.id

    with pytest.raises(StageFailed):
        run_indexing(session, result.job, _BrokenProvider())
    session.rollback()
    session.expire_all()

    assert session.get(DocumentVersion, version_id).status == "draft"


def test_a_failure_leaves_no_embeddings(session, storage, settings) -> None:
    """A provider whose vectors PostgreSQL itself rejects.

    The rows are assigned in memory before the flush refuses them, so this is
    the case where a half-written index would show up if the writes were not
    one transaction.
    """
    result = _parsed(session, storage, settings)
    version_id = result.version.id

    outcome = process_one(session, storage, settings, _WrongWidthProvider())
    session.expire_all()

    assert outcome is not None
    assert outcome.status == JobStatus.QUEUED
    assert all(
        chunk.embedding is None for chunk in chunks_to_embed(session, version_id)
    )
    assert session.get(DocumentVersion, version_id).status == "draft"


def test_a_failure_does_not_supersede_the_incumbent(
    session, storage, settings, embeddings
) -> None:
    """The version that works keeps working."""
    first = _parsed(session, storage, settings)
    run_indexing(session, first.job, embeddings)
    session.commit()

    second = _second_version(
        session, storage, first.document.id, words(120, prefix="b").encode()
    )
    run_job(session, second.job, storage, settings)
    session.commit()

    with pytest.raises(StageFailed):
        run_indexing(session, second.job, _BrokenProvider())
    session.rollback()
    session.expire_all()

    assert session.get(DocumentVersion, first.version.id).status == "active"
    assert session.get(DocumentVersion, second.version.id).status == "draft"


def test_a_chunkless_version_cannot_be_indexed(session, storage, settings) -> None:
    """Nothing to embed is a failure, not a silently empty success."""
    result = _parsed(session, storage, settings)
    session.execute(
        text("DELETE FROM chunks WHERE document_version_id = :version"),
        {"version": result.version.id},
    )
    session.commit()

    with pytest.raises(StageFailed):
        run_indexing(session, result.job, FakeEmbeddingProvider())


def test_the_stage_error_carries_no_document_content(
    session, storage, settings
) -> None:
    body = "SUPERSECRETPHRASE alpha beta " + words(120)
    result = _parsed(session, storage, settings, content=body.encode())

    with pytest.raises(StageFailed) as raised:
        run_indexing(session, result.job, _BrokenProvider())

    detail = raised.value.detail
    assert "SUPERSECRETPHRASE" not in detail
    assert result.version.storage_path not in detail
    assert "Traceback" not in detail
    assert "File \"" not in detail


def test_the_stage_error_reaches_the_job_row_through_the_runner(
    session, storage, settings
) -> None:
    """The runner's failure bookkeeping survives the rollback of the work."""
    body = "SUPERSECRETPHRASE alpha beta " + words(120)
    result = _parsed(session, storage, settings, content=body.encode())
    job_id = result.job.id

    outcome = process_one(session, storage, settings, _BrokenProvider())

    assert outcome is not None
    assert outcome.job_id == job_id
    assert outcome.status == JobStatus.QUEUED  # below max_attempts
    assert "SUPERSECRETPHRASE" not in outcome.detail
    assert "indexing" in outcome.detail


def test_a_failed_indexing_attempt_is_counted(session, storage, settings) -> None:
    """`attempts` is committed at the claim, so the rollback does not undo it."""
    result = _parsed(session, storage, settings)
    job_id = result.job.id
    before = session.get(IngestionJob, job_id).attempts

    process_one(session, storage, settings, _BrokenProvider())
    session.expire_all()

    assert session.get(IngestionJob, job_id).attempts == before + 1


def test_indexing_stops_at_failed_once_attempts_are_exhausted(
    session, storage, settings
) -> None:
    result = _parsed(session, storage, settings)
    job_id = result.job.id

    outcomes = []
    while True:
        outcome = process_one(session, storage, settings, _BrokenProvider())
        if outcome is None:
            break
        outcomes.append(outcome)

    assert outcomes[-1].status == JobStatus.FAILED
    job = session.get(IngestionJob, job_id)
    assert job.attempts == settings.max_attempts
    assert job.status == JobStatus.FAILED


# --- the transaction boundary -----------------------------------------------


def test_embeddings_are_generated_outside_a_transaction(
    session, storage, settings
) -> None:
    """Inference must not be holding a transaction open.

    The provider records whether the session had a transaction open when it
    was asked for vectors. Reading the chunks needs one; `run_indexing` ends
    it before the model runs and opens a fresh one for the writes, and this
    is the assertion that keeps that ordering honest.
    """
    result = _parsed(session, storage, settings)
    session.commit()

    observed: list[bool] = []

    class _Observing(FakeEmbeddingProvider):
        def embed(self, texts):
            observed.append(session.in_transaction())
            return super().embed(texts)

    run_indexing(session, result.job, _Observing())
    session.commit()

    assert observed == [False]


def test_indexing_refuses_a_session_with_unwritten_changes(
    session, storage, settings, embeddings
) -> None:
    """Because ending the read transaction would discard them."""
    result = _parsed(session, storage, settings)
    session.add(IngestionJob(document_version_id=result.version.id))

    with pytest.raises(StageFailed) as raised:
        run_indexing(session, result.job, embeddings)

    assert raised.value.stage == JobStatus.INDEXING
    session.rollback()


def test_generating_embeddings_writes_nothing(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)
    texts = texts_of(chunks_to_embed(session, result.version.id))

    generate_embeddings(texts, embeddings)
    session.expire_all()

    assert all(
        chunk.embedding is None for chunk in chunks_to_embed(session, result.version.id)
    )
    assert session.get(DocumentVersion, result.version.id).status == "draft"


def test_generating_embeddings_for_nothing_is_refused(embeddings) -> None:
    with pytest.raises(IndexingFailed):
        generate_embeddings([], embeddings)


# --- idempotency ------------------------------------------------------------


def test_re_indexing_writes_the_same_chunk_identifiers(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)
    run_indexing(session, result.job, embeddings)
    session.commit()
    before = [chunk.chunk_uid for chunk in chunks_to_embed(session, result.version.id)]

    job = IngestionJob(
        document_version_id=result.version.id, status=JobStatus.QUEUED
    )
    session.add(job)
    session.commit()
    assert process_one(session, storage, settings, embeddings) is not None  # parse
    assert process_one(session, storage, settings, embeddings) is not None  # index
    session.expire_all()

    after = [chunk.chunk_uid for chunk in chunks_to_embed(session, result.version.id)]
    assert after == before


def test_re_indexing_creates_no_duplicate_chunks(
    session, storage, settings, embeddings
) -> None:
    result = _parsed(session, storage, settings)
    run_indexing(session, result.job, embeddings)
    session.commit()
    before = len(chunks_to_embed(session, result.version.id))

    job = IngestionJob(
        document_version_id=result.version.id, status=JobStatus.QUEUED
    )
    session.add(job)
    session.commit()
    process_one(session, storage, settings, embeddings)
    process_one(session, storage, settings, embeddings)
    session.expire_all()

    assert len(chunks_to_embed(session, result.version.id)) == before
    assert session.execute(
        select(Chunk).where(Chunk.document_version_id == result.version.id)
    ).scalars().all()


def test_re_indexing_an_active_version_leaves_it_active(
    session, storage, settings, embeddings
) -> None:
    """Promotion is idempotent: the version that was active still is."""
    result = _parsed(session, storage, settings)
    run_indexing(session, result.job, embeddings)
    session.commit()

    job = IngestionJob(
        document_version_id=result.version.id, status=JobStatus.QUEUED
    )
    session.add(job)
    session.commit()
    process_one(session, storage, settings, embeddings)
    process_one(session, storage, settings, embeddings)
    session.expire_all()

    assert session.get(DocumentVersion, result.version.id).status == "active"


def test_the_same_text_always_gets_the_same_vector(embeddings) -> None:
    """What makes the idempotency assertions above mean anything."""
    first = FakeEmbeddingProvider().embed(["the access policy"])
    second = FakeEmbeddingProvider().embed(["the access policy"])

    assert first == second
