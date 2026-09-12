"""Turning a validated upload into rows.

Two operations and one invariant.

`ingest` writes a document, a version and a queued job. All three go in one
transaction, because a job pointing at a version that does not exist is not a
recoverable state — and because the caller has already been told nothing yet.

`promote_version` is the active/superseded transition. It is **not** called by
upload. See its docstring for why.

The invariant, which the rest of the system will depend on:

    a version is `active` only if its content has been indexed

Milestone 2 upholds it by never activating anything.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Document,
    DocumentVersion,
    IngestionJob,
    JobStatus,
    VersionStatus,
)

logger = logging.getLogger(__name__)

# How many times a version-number collision is retried before giving up. A
# collision means another request committed the same number between our read
# and our write; one retry clears the overwhelming majority, and a bound stops
# a pathological loop.
VERSION_NUMBER_ATTEMPTS = 5


class DocumentNotFound(LookupError):
    """No document with that id."""


class VersionNumberContested(RuntimeError):
    """A version number could not be claimed after several attempts."""


@dataclass(frozen=True)
class Ingested:
    """What one upload produced."""

    document: Document
    version: DocumentVersion
    job: IngestionJob


def storage_key(document_id: uuid.UUID, version_id: uuid.UUID, extension: str) -> str:
    """Where a version's bytes live, relative to the storage root.

    Built entirely from identifiers this application generated. The name the
    uploader chose never appears — it is metadata, and a filename that reached
    the filesystem would be a filename that could aim at it.
    """
    return f"documents/{document_id}/{version_id}{extension}"


def create_document_with_version(
    session: Session,
    *,
    document_id: uuid.UUID,
    title: str,
    department: str | None,
    category: str | None,
    tags: list[str],
    original_filename: str,
    storage_path: str,
    effective_date: date | None,
    version_id: uuid.UUID,
) -> Ingested:
    """A new document, its first version, and a queued job.

    One transaction. The caller commits, so a storage failure afterwards can
    still be compensated before anything is visible.

    `document_id` is supplied rather than generated here because the file has
    already been stored under a key built from it. Letting the database pick a
    different one would leave every stored file sitting in a directory named
    after a document that does not exist.
    """
    document = Document(
        id=document_id,
        title=title,
        department=department,
        category=category,
        tags=tags,
    )
    session.add(document)
    session.flush()

    version = _new_version(
        session,
        document_id=document.id,
        version_number=1,
        original_filename=original_filename,
        storage_path=storage_path,
        effective_date=effective_date,
        version_id=version_id,
    )
    job = _queue_job(session, version)
    return Ingested(document=document, version=version, job=job)


def add_version(
    session: Session,
    *,
    document_id: uuid.UUID,
    original_filename: str,
    storage_path: str,
    effective_date: date | None,
    version_id: uuid.UUID,
) -> Ingested:
    """The next version of an existing document, and a queued job for it.

    The new version is a `draft`. It does **not** supersede the current active
    version: superseding one that has been indexed in favour of one that has
    not would take a working answer away and put nothing in its place.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise DocumentNotFound(str(document_id))

    version = _new_version(
        session,
        document_id=document.id,
        version_number=next_version_number(session, document.id),
        original_filename=original_filename,
        storage_path=storage_path,
        effective_date=effective_date,
        version_id=version_id,
    )
    job = _queue_job(session, version)
    return Ingested(document=document, version=version, job=job)


def next_version_number(session: Session, document_id: uuid.UUID) -> int:
    """One past the highest version this document has.

    Read inside the caller's transaction, and **not** trusted: two requests
    can read the same maximum, and the unique constraint on
    `(document_id, version_number)` is what actually decides. See
    `ingest_with_retry` for what happens when it does.
    """
    highest = session.execute(
        select(func.max(DocumentVersion.version_number)).where(
            DocumentVersion.document_id == document_id
        )
    ).scalar_one_or_none()
    return (highest or 0) + 1


def ingest_with_retry(session: Session, operation, attempts: int = VERSION_NUMBER_ATTEMPTS):
    """Run a version-creating operation, retrying a lost version-number race.

    The retry is **database-only and deliberately so**. The file has already
    been written to storage under a key built from the version UUID, which
    does not change between attempts — so a retry re-reads the maximum version
    number and re-inserts the rows, and never re-reads the request body, never
    writes a second file, and never re-runs validation.

    Anything other than a version-number collision is re-raised untouched:
    this exists for one race, not as a general retry wrapper.
    """
    for attempt in range(1, attempts + 1):
        try:
            result = operation()
            session.commit()
            return result
        except IntegrityError as exc:
            session.rollback()
            if not _is_version_number_collision(exc):
                raise
            if attempt == attempts:
                raise VersionNumberContested(
                    "another upload kept claiming the same version number"
                ) from exc
            logger.info(
                "Version number collision on attempt %d; recalculating.", attempt
            )
    raise AssertionError("unreachable")


def promote_version(session: Session, version: DocumentVersion) -> DocumentVersion:
    """Make this version the active one, superseding whichever was.

    **The caller is responsible for invoking this only after the version's
    content has been successfully indexed.** Nothing in milestone 2 calls it,
    and upload must not: a version becomes active when it can answer a
    question, not when its bytes arrive. Promoting an unparsed version would
    supersede one that works and leave default retrieval — which filters on
    `status = 'active'` — with a version that has no chunks behind it.

    The order is forced by the partial unique index on `(document_id) WHERE
    status = 'active'`: the incumbent is demoted and flushed first, so there
    is never an instant with two active versions. Both statements are in the
    caller's transaction, so either both land or neither does.

    Promoting the version that is already active is a no-op.
    """
    if version.status == VersionStatus.ACTIVE:
        return version

    incumbent = session.execute(
        select(DocumentVersion).where(
            DocumentVersion.document_id == version.document_id,
            DocumentVersion.status == VersionStatus.ACTIVE,
        )
    ).scalar_one_or_none()

    if incumbent is not None:
        incumbent.status = VersionStatus.SUPERSEDED
        # Before the promotion below, so the index never sees two.
        session.flush()

    version.status = VersionStatus.ACTIVE
    session.flush()
    return version


# --- internals --------------------------------------------------------------


def _new_version(
    session: Session,
    *,
    document_id: uuid.UUID,
    version_number: int,
    original_filename: str,
    storage_path: str,
    effective_date: date | None,
    version_id: uuid.UUID,
) -> DocumentVersion:
    """A draft version.

    `checksum` and `page_count` are left null on purpose. The checksum is
    defined over normalized text and the page count comes from parsing, and
    neither exists until milestone 3. A value invented here would be a
    plausible-looking lie.
    """
    version = DocumentVersion(
        id=version_id,
        document_id=document_id,
        version_number=version_number,
        status=VersionStatus.DRAFT,
        original_filename=original_filename,
        storage_path=storage_path,
        effective_date=effective_date,
    )
    session.add(version)
    session.flush()
    return version


def _queue_job(session: Session, version: DocumentVersion) -> IngestionJob:
    """The record that this version is waiting.

    Created `queued` and left there. Nothing in this milestone advances it —
    the runner is milestone 3 — so a queued job with no runner is the correct
    and intended state.
    """
    job = IngestionJob(document_version_id=version.id, status=JobStatus.QUEUED)
    session.add(job)
    session.flush()
    return job


def _is_version_number_collision(exc: IntegrityError) -> bool:
    return "uq_document_versions_document_id_version_number" in str(exc.orig)


__all__ = [
    "VERSION_NUMBER_ATTEMPTS",
    "DocumentNotFound",
    "Ingested",
    "VersionNumberContested",
    "add_version",
    "create_document_with_version",
    "ingest_with_retry",
    "next_version_number",
    "promote_version",
    "storage_key",
]
