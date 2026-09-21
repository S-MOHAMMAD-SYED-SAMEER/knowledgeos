"""Building a small, really-indexed corpus, for retrieval tests.

Retrieval tests need chunks with real vectors and a real `tsv` — produced by
running the actual milestone 3/4 pipeline functions (parse, chunk, index),
not inserted directly. Inserting rows by hand would test the query against a
corpus the application could never actually produce; running the real
pipeline against `FakeEmbeddingProvider` gives deterministic vectors from
real rows, which is what a retrieval test should search.
"""

import io
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion import service
from app.ingestion.pipeline import run_indexing, run_job
from app.models import Document, DocumentVersion
from app.providers import FakeEmbeddingProvider
from app.storage import LocalStorage


@dataclass(frozen=True)
class SeededVersion:
    document: Document
    version: DocumentVersion


def seed_active_version(
    session: Session,
    storage: LocalStorage,
    *,
    text: str,
    title: str = "doc.txt",
    department: str | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    document_id: uuid.UUID | None = None,
    embeddings: FakeEmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> SeededVersion:
    """Upload, parse, chunk and index `text`, leaving one `active` version.

    Calls the pipeline functions directly rather than going through the job
    runner or HTTP: retrieval tests need the resulting rows, not a
    demonstration that the runner drains a queue — that is milestone 3 and
    4's own test suite.
    """
    settings = settings or Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5
    )
    embeddings = embeddings or FakeEmbeddingProvider()
    document_id = document_id or uuid.uuid4()
    version_id = uuid.uuid4()

    key = service.storage_key(document_id, version_id, ".txt")
    storage.write(key, io.BytesIO(text.encode()))

    result = service.create_document_with_version(
        session,
        document_id=document_id,
        title=title,
        department=department,
        category=category,
        tags=tags or [],
        original_filename=title,
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()

    run_job(session, result.job, storage, settings)
    session.commit()
    run_indexing(session, result.job, embeddings)
    session.commit()
    session.refresh(result.version)

    return SeededVersion(document=result.document, version=result.version)


def seed_additional_active_version(
    session: Session,
    storage: LocalStorage,
    *,
    document_id: uuid.UUID,
    text: str,
    embeddings: FakeEmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> SeededVersion:
    """A second (or third, ...) version of an existing document, indexed all
    the way to `active` — which, through the real `run_indexing` pipeline,
    supersedes whichever version was active before it.
    """
    settings = settings or Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5
    )
    embeddings = embeddings or FakeEmbeddingProvider()
    version_id = uuid.uuid4()
    key = service.storage_key(document_id, version_id, ".txt")
    storage.write(key, io.BytesIO(text.encode()))

    result = service.add_version(
        session,
        document_id=document_id,
        original_filename="version.txt",
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()

    run_job(session, result.job, storage, settings)
    session.commit()
    run_indexing(session, result.job, embeddings)
    session.commit()
    session.refresh(result.version)

    return SeededVersion(document=result.document, version=result.version)


def add_draft_version(
    session: Session,
    storage: LocalStorage,
    *,
    document_id: uuid.UUID,
    text: str,
    settings: Settings | None = None,
) -> SeededVersion:
    """A second version of an existing document, parsed and chunked but
    left at `indexing` — never embedded, never promoted. For proving a
    draft is excluded from retrieval."""
    settings = settings or Settings(
        _env_file=None, chunk_size_tokens=50, chunk_overlap_tokens=5
    )
    version_id = uuid.uuid4()
    key = service.storage_key(document_id, version_id, ".txt")
    storage.write(key, io.BytesIO(text.encode()))

    result = service.add_version(
        session,
        document_id=document_id,
        original_filename="draft.txt",
        storage_path=key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    run_job(session, result.job, storage, settings)
    session.commit()
    session.refresh(result.version)

    return SeededVersion(document=result.document, version=result.version)


__all__ = [
    "SeededVersion",
    "add_draft_version",
    "seed_active_version",
    "seed_additional_active_version",
]
