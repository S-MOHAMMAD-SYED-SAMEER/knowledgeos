"""Uploading documents, and reading back what was uploaded.

The order of an upload is the interesting part, and it is deliberate:

1. validate, reading the body in bounded chunks into a temporary file;
2. move that file into storage, atomically, under a key made of UUIDs;
3. write the document, version and job in **one** database transaction;
4. on any database failure, delete the file that step 2 wrote.

Storage first, database last, because the two possible residues are not
equally bad. A file with no row is invisible, costs disk, and can be removed.
A row with no file is a version this system believes it has, and it breaks the
moment a parser opens it. So the ordering guarantees the one that can be
cleaned up rather than the one that cannot.

A crash between step 2 and step 3 leaves an orphan file. That window is real,
it is not swept in this milestone, and the README says so.

Nothing in the upload path promotes a version. A version becomes `active`
when its content has been indexed, which happens in the indexing stage and
nowhere else.

`POST /documents/{id}/reindex` queues the active version to be run again. It
creates no new version: re-running rewrites the same chunks with the same
identifiers, which is what makes re-indexing idempotent.
"""

import logging
import tempfile
import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import (
    DocumentDetail,
    DocumentOut,
    DocumentPage,
    JobOut,
    UploadAccepted,
    VersionOut,
)
from app.config import Settings, get_settings
from app.db.session import get_session
from app.ingestion import service
from app.ingestion.validation import UploadRejected, validate
from app.models import (
    TERMINAL_STATUSES,
    Document,
    DocumentVersion,
    IngestionJob,
    JobStatus,
    VersionStatus,
)
from app.storage import Storage, StorageError, get_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# The most documents one list request will return. A constant rather than a
# setting: it bounds a response, and nothing about a deployment should need to
# change it.
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 50


@router.post(
    "/documents",
    response_model=UploadAccepted,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    response: Response,
    file: UploadFile,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    storage: Annotated[Storage, Depends(get_storage)],
    title: Annotated[str | None, Form()] = None,
    department: Annotated[str | None, Form()] = None,
    category: Annotated[str | None, Form()] = None,
    tags: Annotated[list[str] | None, Form()] = None,
    effective_date: Annotated[date | None, Form()] = None,
) -> UploadAccepted:
    """A new document, its first version, and a queued ingestion job."""
    del response

    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    validated, key = _store(
        file, document_id, version_id, settings=settings, storage=storage
    )

    def attempt() -> service.Ingested:
        return service.create_document_with_version(
            session,
            document_id=document_id,
            title=title or validated.filename,
            department=department,
            category=category,
            tags=list(tags or []),
            original_filename=validated.filename,
            storage_path=key,
            effective_date=effective_date,
            version_id=version_id,
        )

    result = _commit_or_clean_up(session, attempt, storage=storage, key=key)
    return _accepted(result)


@router.post(
    "/documents/{document_id}/versions",
    response_model=UploadAccepted,
    status_code=status.HTTP_201_CREATED,
)
def upload_version(
    document_id: uuid.UUID,
    file: UploadFile,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    storage: Annotated[Storage, Depends(get_storage)],
    effective_date: Annotated[date | None, Form()] = None,
) -> UploadAccepted:
    """The next version of a document, as a draft, with its own queued job.

    The current active version is left alone. It is the one that has been
    indexed and can answer questions; this one has not been.
    """
    if session.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="No such document")

    version_id = uuid.uuid4()
    validated, key = _store(
        file, document_id, version_id, settings=settings, storage=storage
    )

    def attempt() -> service.Ingested:
        return service.add_version(
            session,
            document_id=document_id,
            original_filename=validated.filename,
            storage_path=key,
            effective_date=effective_date,
            version_id=version_id,
        )

    try:
        result = _commit_or_clean_up(session, attempt, storage=storage, key=key)
    except service.DocumentNotFound as exc:
        # Deleted between the check above and the insert.
        storage.delete(key)
        raise HTTPException(status_code=404, detail="No such document") from exc
    return _accepted(result)


@router.post(
    "/documents/{document_id}/reindex",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def reindex_document(
    document_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> UploadAccepted:
    """Queue the active version of a document to be parsed and indexed again.

    Targets **only the active version**, and deliberately. The active version
    is the one answering questions; a draft that never finished indexing is
    the upload path's business, not this endpoint's, and a superseded version
    is history. So a document with no active version is a conflict rather than
    a silent no-op.

    No new document version is created — the same version is re-run. Because
    `chunk_uid` is derived from the version, the sequence and the text, the
    rows written are identical to the ones they replace: re-indexing produces
    the same chunks rather than duplicates, which is the specification's
    acceptance gate for this feature.

    A version already being re-run is a conflict too. Two jobs racing on one
    version would have them deleting and rewriting each other's chunks.
    """
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document")

    version = session.execute(
        select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.status == VersionStatus.ACTIVE,
        )
    ).scalar_one_or_none()
    if version is None:
        raise HTTPException(
            status_code=409,
            detail="This document has no active version to re-index",
        )

    outstanding = session.execute(
        select(IngestionJob).where(
            IngestionJob.document_version_id == version.id,
            IngestionJob.status.notin_(TERMINAL_STATUSES),
        )
    ).first()
    if outstanding is not None:
        raise HTTPException(
            status_code=409,
            detail="This version is already being ingested",
        )

    job = IngestionJob(document_version_id=version.id, status=JobStatus.QUEUED)
    session.add(job)
    session.commit()

    return UploadAccepted(
        document=DocumentOut.model_validate(document),
        version=VersionOut.model_validate(version),
        job=JobOut.model_validate(job),
    )


@router.get("/documents", response_model=DocumentPage)
def list_documents(
    session: Annotated[Session, Depends(get_session)],
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
) -> DocumentPage:
    """Newest first. Filtering by metadata belongs with retrieval."""
    limit = max(1, min(limit, MAX_PAGE_SIZE))
    offset = max(0, offset)

    documents = (
        session.execute(
            select(Document)
            .order_by(Document.created_at.desc(), Document.id)
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )
    return DocumentPage(
        documents=[DocumentOut.model_validate(row) for row in documents],
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> DocumentDetail:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document")

    versions = (
        session.execute(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number)
        )
        .scalars()
        .all()
    )
    return DocumentDetail(
        document=DocumentOut.model_validate(document),
        versions=[VersionOut.model_validate(row) for row in versions],
    )


# --- internals --------------------------------------------------------------


def _store(
    file: UploadFile,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    *,
    settings: Settings,
    storage: Storage,
) -> tuple[object, str]:
    """Validate an upload and put it in storage. Nothing is written on refusal.

    The body goes to a temporary file while it is checked, because the checks
    need the size and the signature before anything is committed to a real
    key, and because holding an upload of unknown length in memory is how a
    request becomes an outage.
    """
    with tempfile.TemporaryFile() as scratch:
        try:
            validated = validate(
                filename=file.filename,
                declared_type=file.content_type,
                source=file.file,
                destination=scratch,
                allowed_extensions=tuple(settings.allowed_extensions),
                max_bytes=settings.max_upload_bytes,
            )
        except UploadRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc

        key = service.storage_key(document_id, version_id, validated.extension)
        scratch.seek(0)
        try:
            storage.write(key, scratch)
        except StorageError as exc:
            # The path and the cause go to the log, never to the caller.
            logger.exception("Could not store an upload for version %s", version_id)
            raise HTTPException(
                status_code=500, detail="The file could not be stored"
            ) from exc

    return validated, key


def _commit_or_clean_up(
    session: Session, attempt, *, storage: Storage, key: str
) -> service.Ingested:
    """Commit the rows, or take the file back out of storage.

    The compensation is what keeps the ordering honest: writing the file first
    is only safe if a database failure removes it again.
    """
    try:
        return service.ingest_with_retry(session, attempt)
    except service.DocumentNotFound:
        raise
    except Exception as exc:
        storage.delete(key)
        if isinstance(exc, HTTPException):
            raise
        logger.exception("Could not record an upload; its file has been removed")
        raise HTTPException(
            status_code=500, detail="The upload could not be recorded"
        ) from exc


def _accepted(result: service.Ingested) -> UploadAccepted:
    return UploadAccepted(
        document=DocumentOut.model_validate(result.document),
        version=VersionOut.model_validate(result.version),
        job=JobOut.model_validate(result.job),
    )


__all__ = ["DEFAULT_PAGE_SIZE", "MAX_PAGE_SIZE", "router"]
