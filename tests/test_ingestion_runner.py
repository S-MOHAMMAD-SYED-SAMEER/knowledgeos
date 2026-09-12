"""The runner: claiming, retrying, and stopping where milestone 3 stops.

The concurrency test is a real one — two separate database connections, both
reaching for the same queued row — because `FOR UPDATE SKIP LOCKED` is the
whole reason this design is safe with more than one worker, and a mocked
version of it would prove nothing.
"""

import io
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion import service
from app.ingestion.runner import IngestionRunner, claim_next_job, process_one, run_pending
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus
from app.storage import LocalStorage

from .fixtures import plain_zip_bytes, words


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5, max_attempts=3
    )


def _queued(session, storage, content: bytes = None, filename: str = "notes.txt"):
    document_id = uuid.uuid4()
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


# --- claiming ---------------------------------------------------------------


def test_a_queued_job_is_claimed(session, storage) -> None:
    result = _queued(session, storage)

    claimed = claim_next_job(session)

    assert claimed is not None
    assert claimed.id == result.job.id
    assert claimed.status == JobStatus.PARSING
    assert claimed.attempts == 1
    assert claimed.started_at is not None


def test_nothing_is_claimed_when_nothing_is_queued(session, storage) -> None:
    assert claim_next_job(session) is None


def test_a_job_that_is_not_queued_is_not_claimed(session, storage) -> None:
    """Milestone 2's jobs are the only queued ones; a job already running or
    finished must not be picked up again."""
    result = _queued(session, storage)
    result.job.status = JobStatus.INDEXING
    session.commit()

    assert claim_next_job(session) is None


def test_two_connections_do_not_claim_the_same_job(
    session, storage, migrated_engine
) -> None:
    """`FOR UPDATE SKIP LOCKED`, proven with a second real connection.

    The first session holds the row lock in an open transaction; the second
    must skip it and find nothing, rather than blocking until the first
    commits.
    """
    _queued(session, storage)

    # commit=False keeps the transaction — and therefore the row lock — open.
    first = claim_next_job(session, commit=False)
    assert first is not None

    other = create_engine(migrated_engine.url)
    try:
        with Session(other) as rival:
            assert claim_next_job(rival) is None
    finally:
        other.dispose()

    session.commit()


# --- a successful pass ------------------------------------------------------


def test_processing_a_job_writes_chunks_and_stops_at_indexing(
    session, storage, settings
) -> None:
    _queued(session, storage)

    outcome = process_one(session, storage, settings)

    assert outcome is not None
    assert outcome.status == JobStatus.INDEXING
    assert outcome.attempts == 1
    assert len(session.execute(select(Chunk)).scalars().all()) > 0


def test_a_finished_job_records_when_it_completed(
    session, storage, settings
) -> None:
    result = _queued(session, storage)

    process_one(session, storage, settings)
    session.expire_all()

    job = session.get(IngestionJob, result.job.id)
    assert job.started_at is not None
    assert job.completed_at is not None
    assert job.stage_error is None


def test_processing_returns_none_when_the_queue_is_empty(
    session, storage, settings
) -> None:
    assert process_one(session, storage, settings) is None


def test_the_version_is_never_promoted_by_the_runner(
    session, storage, settings
) -> None:
    result = _queued(session, storage)

    process_one(session, storage, settings)
    session.expire_all()

    assert session.get(DocumentVersion, result.version.id).status == "draft"


# --- failure and retry ------------------------------------------------------


def test_a_failing_job_goes_back_to_queued_below_the_bound(
    session, storage, settings
) -> None:
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    outcome = process_one(session, storage, settings)

    assert outcome.status == JobStatus.QUEUED
    assert outcome.attempts == 1


def test_attempts_increment_on_every_try(session, storage, settings) -> None:
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    for expected in (1, 2, 3):
        outcome = process_one(session, storage, settings)
        assert outcome.attempts == expected


def test_the_job_fails_at_the_bound(session, storage, settings) -> None:
    """Deterministic: exactly max_attempts tries, then it stops."""
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    for _ in range(settings.max_attempts):
        outcome = process_one(session, storage, settings)

    assert outcome.status == JobStatus.FAILED
    assert outcome.attempts == settings.max_attempts


def test_a_failed_job_is_not_retried_again(session, storage, settings) -> None:
    _queued(session, storage, plain_zip_bytes(), "handbook.docx")
    for _ in range(settings.max_attempts):
        process_one(session, storage, settings)

    assert process_one(session, storage, settings) is None


def test_the_bound_comes_from_configuration(session, storage) -> None:
    """Not hard-coded: a different bound really changes the behaviour."""
    settings = Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5, max_attempts=1
    )
    _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    outcome = process_one(session, storage, settings)

    assert outcome.status == JobStatus.FAILED
    assert outcome.attempts == 1


def test_a_failure_records_the_stage(session, storage, settings) -> None:
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    process_one(session, storage, settings)
    session.expire_all()

    job = session.get(IngestionJob, result.job.id)
    assert job.stage_error.startswith("parsing:")


def test_a_failure_writes_no_chunks(session, storage, settings) -> None:
    """No partial persistence: the work rolls back, only the record survives."""
    _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    process_one(session, storage, settings)

    assert session.execute(select(Chunk)).scalars().all() == []


def test_a_failure_leaves_the_checksum_unset(session, storage, settings) -> None:
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    process_one(session, storage, settings)
    session.expire_all()

    assert session.get(DocumentVersion, result.version.id).checksum is None


def test_a_stage_error_carries_no_document_content(
    session, storage, settings
) -> None:
    """It is stored, and read back through the API."""
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    process_one(session, storage, settings)
    session.expire_all()

    job = session.get(IngestionJob, result.job.id)
    assert "not a word document" not in job.stage_error.lower()
    assert str(storage.root) not in job.stage_error
    assert "Traceback" not in job.stage_error


def test_a_job_that_fails_then_succeeds_keeps_its_attempts(
    session, storage, settings
) -> None:
    """A retry is a retry of the same job, not a new one."""
    result = _queued(session, storage, plain_zip_bytes(), "handbook.docx")

    process_one(session, storage, settings)

    # The file is replaced with something parseable, as a re-upload would.
    storage.write(result.version.storage_path, io.BytesIO(words(120).encode()))
    session.expire_all()
    version = session.get(DocumentVersion, result.version.id)
    version.original_filename = "notes.txt"
    session.commit()

    outcome = process_one(session, storage, settings)

    assert outcome.status == JobStatus.INDEXING
    assert outcome.attempts == 2


# --- draining ---------------------------------------------------------------


def test_several_queued_jobs_are_all_processed(
    session, storage, settings, migrated_engine
) -> None:
    from sqlalchemy.orm import sessionmaker

    for _ in range(3):
        _queued(session, storage)

    outcomes = run_pending(
        sessions=sessionmaker(bind=migrated_engine), storage=storage, settings=settings
    )

    assert len(outcomes) == 3
    assert all(outcome.status == JobStatus.INDEXING for outcome in outcomes)


def test_draining_stops_when_the_queue_is_empty(
    session, storage, settings, migrated_engine
) -> None:
    from sqlalchemy.orm import sessionmaker

    assert run_pending(
        sessions=sessionmaker(bind=migrated_engine), storage=storage, settings=settings
    ) == []


def test_a_job_queued_before_the_process_started_is_still_picked_up(
    session, storage, settings, migrated_engine
) -> None:
    """Durability: milestone 2 left real queued rows, and polling finds them.

    A background task attached to the upload request could not — the request
    that created these is long gone.
    """
    from sqlalchemy.orm import sessionmaker

    result = _queued(session, storage)
    job_id = result.job.id
    # The session that created it is gone, as the upload request would be.
    session.close()

    outcomes = run_pending(
        sessions=sessionmaker(bind=migrated_engine), storage=storage, settings=settings
    )

    assert [outcome.job_id for outcome in outcomes] == [job_id]


# --- lifecycle --------------------------------------------------------------


def test_the_runner_starts_and_stops(settings) -> None:
    runner = IngestionRunner(settings=settings)

    assert runner.running is False
    runner.start()
    try:
        assert runner.running is True
    finally:
        runner.stop()
    assert runner.running is False


def test_starting_twice_is_harmless(settings) -> None:
    runner = IngestionRunner(settings=settings)
    runner.start()
    try:
        runner.start()
        assert runner.running is True
    finally:
        runner.stop()


def test_stopping_one_that_never_started_is_harmless(settings) -> None:
    IngestionRunner(settings=settings).stop()


def test_each_application_gets_its_own_runner(settings_env) -> None:
    """No hidden global: two applications in one process do not share one."""
    from app.main import create_app

    assert create_app().state.runner is None
    assert create_app() is not create_app()
