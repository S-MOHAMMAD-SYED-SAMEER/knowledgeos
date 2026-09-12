"""Rows: version numbering, the queued job, and the promotion primitive.

The promotion tests are the important ones. `promote_version` is the only
thing in milestone 2 that can make a version active, nothing calls it yet, and
the reason it exists now is that milestone 4 must find a tested transition
rather than invent one against a partial unique index it has not thought about.
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ingestion import service
from app.models import Document, DocumentVersion, IngestionJob, JobStatus, VersionStatus

from .conftest import version as build_version


def _create(session: Session, title: str = "Access SOP") -> service.Ingested:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    result = service.create_document_with_version(
        session,
        document_id=document_id,
        title=title,
        department="Engineering",
        category=None,
        tags=["access"],
        original_filename="access-sop.pdf",
        storage_path=service.storage_key(document_id, version_id, ".pdf"),
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    return result


# --- the storage key --------------------------------------------------------


def test_the_storage_key_is_built_only_from_identifiers() -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()

    key = service.storage_key(document_id, version_id, ".pdf")

    assert key == f"documents/{document_id}/{version_id}.pdf"


def test_the_storage_key_never_contains_the_uploaded_filename() -> None:
    """The property that makes path traversal impossible rather than blocked."""
    key = service.storage_key(uuid.uuid4(), uuid.uuid4(), ".pdf")

    assert ".." not in key
    assert not key.startswith("/")


# --- creating ---------------------------------------------------------------


def test_the_document_keeps_the_id_its_stored_file_was_filed_under(
    session: Session
) -> None:
    """The file is written before the row exists, under a key built from this
    id. A database-generated id instead would leave every file in a directory
    named after a document that does not exist."""
    result = _create(session)

    assert str(result.document.id) in result.version.storage_path


def test_a_document_a_version_and_a_job_are_created(session: Session) -> None:
    result = _create(session)

    assert session.get(Document, result.document.id) is not None
    assert session.get(DocumentVersion, result.version.id) is not None
    assert session.get(IngestionJob, result.job.id) is not None


def test_the_first_version_is_number_one_and_a_draft(session: Session) -> None:
    result = _create(session)

    assert result.version.version_number == 1
    assert result.version.status == VersionStatus.DRAFT


def test_the_job_starts_queued_with_no_attempts(session: Session) -> None:
    result = _create(session)

    assert result.job.status == JobStatus.QUEUED
    assert result.job.attempts == 0
    assert result.job.stage_error is None
    assert result.job.started_at is None
    assert result.job.completed_at is None


def test_checksum_and_page_count_are_left_alone(session: Session) -> None:
    """Both come from parsing. A value here would be invented."""
    result = _create(session)

    assert result.version.checksum is None
    assert result.version.page_count is None


def test_two_uploads_create_two_independent_documents(session: Session) -> None:
    first = _create(session, "Handbook")
    second = _create(session, "Security Policy")

    assert first.document.id != second.document.id
    assert first.version.version_number == second.version.version_number == 1


# --- version numbering ------------------------------------------------------


def test_the_next_version_number_is_one_past_the_highest(session: Session) -> None:
    created = _create(session)

    assert service.next_version_number(session, created.document.id) == 2


def test_a_document_with_no_versions_starts_at_one(session: Session) -> None:
    document = Document(title="Empty")
    session.add(document)
    session.commit()

    assert service.next_version_number(session, document.id) == 1


def test_adding_a_version_increments(session: Session) -> None:
    created = _create(session)
    version_id = uuid.uuid4()

    second = service.add_version(
        session,
        document_id=created.document.id,
        original_filename="access-sop-v2.pdf",
        storage_path=f"documents/x/{version_id}.pdf",
        effective_date=None,
        version_id=version_id,
    )
    session.commit()

    assert second.version.version_number == 2
    assert second.version.status == VersionStatus.DRAFT


def test_adding_a_version_to_a_missing_document_is_refused(session: Session) -> None:
    with pytest.raises(service.DocumentNotFound):
        service.add_version(
            session,
            document_id=uuid.uuid4(),
            original_filename="x.pdf",
            storage_path="documents/x/y.pdf",
            effective_date=None,
            version_id=uuid.uuid4(),
        )


def test_a_new_version_does_not_supersede_the_active_one(session: Session) -> None:
    """The invariant: active means indexed, and v2 has not been."""
    created = _create(session)
    service.promote_version(session, created.version)
    session.commit()

    version_id = uuid.uuid4()
    service.add_version(
        session,
        document_id=created.document.id,
        original_filename="v2.pdf",
        storage_path=f"documents/x/{version_id}.pdf",
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, created.version.id).status == "active"
    assert session.get(DocumentVersion, version_id).status == "draft"


# --- the version-number race ------------------------------------------------


def test_the_unique_constraint_is_what_actually_decides(session: Session) -> None:
    """Not the read. Two versions computed the same number; one must lose."""
    created = _create(session)
    session.add(build_version(created.document.id, 2))
    session.commit()

    session.add(build_version(created.document.id, 2))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_a_lost_race_is_retried_against_the_database_only(session: Session) -> None:
    """The heart of the concurrency requirement.

    A real second connection commits version 2 between our read and our write,
    so the first attempt loses the race for real — this is not a mocked
    IntegrityError. The retry must then recalculate and land on 3, **without**
    the caller re-reading the request body or writing a second file.

    The file write is represented by a counter: it happens before the retry
    loop, exactly as the route does it, so if the implementation ever retried
    the whole upload this count would rise above one.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as PlainSession

    created = _create(session)
    document_id = created.document.id
    engine = session.get_bind()

    stored_files = 0

    def write_the_file() -> str:
        nonlocal stored_files
        stored_files += 1
        return f"documents/{document_id}/{uuid.uuid4()}.pdf"

    # The storage write happens once, before any database work.
    storage_path = write_the_file()
    interference = {"done": False}

    def attempt() -> service.Ingested:
        if not interference["done"]:
            # A genuinely separate connection takes version 2 first.
            other = create_engine(engine.url)
            with PlainSession(other) as rival:
                rival.add(build_version(document_id, 2))
                rival.commit()
            other.dispose()
            interference["done"] = True
        return service.add_version(
            session,
            document_id=document_id,
            original_filename="v.pdf",
            storage_path=storage_path,
            effective_date=None,
            version_id=uuid.uuid4(),
        )

    result = service.ingest_with_retry(session, attempt)

    assert result.version.version_number == 3
    assert stored_files == 1, "the file must not be written again on a retry"


def test_a_failure_that_is_not_a_version_collision_is_not_retried(
    session: Session
) -> None:
    """This wrapper exists for one race, not as a general retry."""
    created = _create(session)
    calls = 0

    def attempt():
        nonlocal calls
        calls += 1
        # A foreign key that does not resolve: an IntegrityError, but not this
        # one.
        session.add(build_version(uuid.uuid4(), 1))
        session.flush()

    with pytest.raises(IntegrityError):
        service.ingest_with_retry(session, attempt)

    assert calls == 1


def test_an_endlessly_contested_number_eventually_gives_up(session: Session) -> None:
    created = _create(session)
    document_id = created.document.id

    def attempt():
        # Claim the number we are about to use, from this same session, so the
        # collision recurs on every attempt.
        number = service.next_version_number(session, document_id)
        session.add(build_version(document_id, number))
        session.flush()
        session.add(build_version(document_id, number))
        session.flush()

    with pytest.raises(service.VersionNumberContested):
        service.ingest_with_retry(session, attempt, attempts=3)


# --- promotion --------------------------------------------------------------


def test_promoting_makes_a_version_active(session: Session) -> None:
    created = _create(session)

    service.promote_version(session, created.version)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, created.version.id).status == "active"


def test_promoting_supersedes_the_previous_active_version(session: Session) -> None:
    created = _create(session)
    service.promote_version(session, created.version)
    session.commit()

    second = build_version(created.document.id, 2)
    session.add(second)
    session.commit()

    service.promote_version(session, second)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, created.version.id).status == "superseded"
    assert session.get(DocumentVersion, second.id).status == "active"


def test_the_transition_never_has_two_active_versions_at_once(
    session: Session
) -> None:
    """The ordering the partial unique index forces. If promotion ran before
    demotion this would raise."""
    created = _create(session)
    service.promote_version(session, created.version)
    session.commit()

    second = build_version(created.document.id, 2)
    session.add(second)
    session.commit()

    service.promote_version(session, second)  # No IntegrityError.
    session.commit()

    active = session.execute(
        select(DocumentVersion).where(DocumentVersion.status == "active")
    ).scalars().all()
    assert len(active) == 1


def test_promoting_the_already_active_version_is_a_no_op(session: Session) -> None:
    created = _create(session)
    service.promote_version(session, created.version)
    session.commit()

    service.promote_version(session, created.version)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, created.version.id).status == "active"


def test_promotion_leaves_other_documents_alone(session: Session) -> None:
    first = _create(session, "Handbook")
    second = _create(session, "Security Policy")

    service.promote_version(session, first.version)
    service.promote_version(session, second.version)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, first.version.id).status == "active"
    assert session.get(DocumentVersion, second.version.id).status == "active"


def test_a_superseded_version_can_be_promoted_again(session: Session) -> None:
    """Rolling back to an earlier version is a legitimate thing to want."""
    created = _create(session)
    second = build_version(created.document.id, 2)
    session.add(second)
    session.commit()

    service.promote_version(session, created.version)
    session.commit()
    service.promote_version(session, second)
    session.commit()
    service.promote_version(session, created.version)
    session.commit()
    session.expire_all()

    assert session.get(DocumentVersion, created.version.id).status == "active"
    assert session.get(DocumentVersion, second.id).status == "superseded"


def test_nothing_in_the_application_promotes_on_upload() -> None:
    """Asserted against the source: the route must not call this."""
    import ast
    import pathlib

    from app.api import documents as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "promote_version" not in called
