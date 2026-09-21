"""Seeding the evaluation fixture corpus through the real ingestion
pipeline.

Every chunk this corpus produces must come from the actual parser and
chunker — the same code path a real upload takes — because
`expected_chunk_uids` in the question set can only ever match chunk_uids
the real pipeline actually produced. `evals/fixtures/knowledge_base/
manifest.yaml` pins the document and version UUIDs that make this
reproducible: `chunk_uid` is derived from `document_version_id`
(`app/chunking/uid.py`), so a document seeded under a fresh random id every
run would produce a fresh, unmatchable set of chunk_uids every time.

Idempotent by construction, not by a special-cased flag: a version whose
row already exists is left alone rather than re-seeded, and seeding a
version that does not yet exist writes its chunks the same way a
production re-index does — deterministically, by `document_version_id` —
so running this module twice never produces a duplicate chunk.

This module calls a provider and writes to the database. Unlike
`app/retrieval/`, it is not bound by the specification's "no provider
calls" layering rule — the same relationship `app/ingestion/pipeline.py`
has to `app/providers/embeddings.py`.
"""

import io
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion import service
from app.ingestion.pipeline import run_indexing, run_job
from app.models import Chunk, Document, DocumentVersion, VersionStatus
from app.providers.embeddings import EmbeddingProvider
from app.storage import Storage

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent / "fixtures" / "knowledge_base" / "manifest.yaml"
)

# D9, locked: the chunk configuration every chunk_uid in the frozen question
# set was generated under. The manifest also states these two numbers, and
# is checked against these constants rather than blindly trusted — editing
# the manifest's numbers without understanding the ground-truth consequence
# is exactly the mistake this guard exists to catch.
PINNED_CHUNK_SIZE_TOKENS = 512
PINNED_CHUNK_OVERLAP_TOKENS = 64

# D6, locked: the corpus must produce meaningfully more active chunks than
# the 20-candidate retrieval window, or Recall@10/@20-style saturation makes
# the metrics measure nothing.
MINIMUM_ACTIVE_CHUNKS = 20


class CorpusError(RuntimeError):
    """The fixture corpus could not be seeded or verified.

    Never carries document content — only structural facts (a count, a
    missing key, a file name).
    """


@dataclass(frozen=True)
class SeededDocument:
    """One manifest document's outcome: which version ended up active, and
    how many chunks it holds."""

    key: str
    document_id: uuid.UUID
    active_version_id: uuid.UUID
    chunk_count: int


@dataclass(frozen=True)
class CorpusSeedResult:
    documents: list[SeededDocument]
    total_active_chunks: int
    skipped_existing: int
    newly_seeded: int


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "documents" not in raw:
        raise CorpusError(
            f"{path.name}: manifest must be a mapping with a 'documents' list"
        )

    for key in ("chunk_size_tokens", "chunk_overlap_tokens"):
        if key not in raw:
            raise CorpusError(f"{path.name}: manifest is missing '{key}'")

    if raw["chunk_size_tokens"] != PINNED_CHUNK_SIZE_TOKENS:
        raise CorpusError(
            f"{path.name}: chunk_size_tokens is {raw['chunk_size_tokens']}, "
            f"but the frozen question set was generated at "
            f"{PINNED_CHUNK_SIZE_TOKENS} — changing it would silently "
            f"invalidate every expected_chunk_uid"
        )
    if raw["chunk_overlap_tokens"] != PINNED_CHUNK_OVERLAP_TOKENS:
        raise CorpusError(
            f"{path.name}: chunk_overlap_tokens is "
            f"{raw['chunk_overlap_tokens']}, but the frozen question set "
            f"was generated at {PINNED_CHUNK_OVERLAP_TOKENS}"
        )

    return raw


def ensure_corpus_seeded(
    session: Session,
    storage: Storage,
    embeddings: EmbeddingProvider,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> CorpusSeedResult:
    """Seed every document/version in the manifest that is not already
    present, through the real ingestion pipeline. Safe to call repeatedly:
    a version already in the database is left untouched rather than
    re-seeded.
    """
    manifest = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    settings = Settings(
        _env_file=None,
        chunk_size_tokens=manifest["chunk_size_tokens"],
        chunk_overlap_tokens=manifest["chunk_overlap_tokens"],
    )

    seeded_documents: list[SeededDocument] = []
    skipped = 0
    newly_seeded = 0

    for entry in manifest["documents"]:
        document_id = uuid.UUID(str(entry["document_id"]))
        versions = entry.get("versions") or []
        if not versions:
            raise CorpusError(f"{entry['key']}: manifest entry has no versions")

        for index, version_entry in enumerate(versions):
            version_id = uuid.UUID(str(version_entry["version_id"]))

            if session.get(DocumentVersion, version_id) is not None:
                skipped += 1
                continue

            content = (base_dir / version_entry["file"]).read_bytes()
            storage_key = service.storage_key(document_id, version_id, ".md")
            storage.write(storage_key, io.BytesIO(content))

            if index == 0:
                result = service.create_document_with_version(
                    session,
                    document_id=document_id,
                    title=entry["title"],
                    department=entry.get("department"),
                    category=entry.get("category"),
                    tags=entry.get("tags") or [],
                    original_filename=version_entry["file"],
                    storage_path=storage_key,
                    effective_date=None,
                    version_id=version_id,
                )
            else:
                result = service.add_version(
                    session,
                    document_id=document_id,
                    original_filename=version_entry["file"],
                    storage_path=storage_key,
                    effective_date=None,
                    version_id=version_id,
                )
            session.commit()

            run_job(session, result.job, storage, settings)
            session.commit()
            run_indexing(session, result.job, embeddings)
            session.commit()
            newly_seeded += 1

        session.expire_all()
        active_version = session.execute(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.status == VersionStatus.ACTIVE,
            )
        ).scalar_one()
        chunk_count = session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_version_id == active_version.id)
        ).scalar_one()
        seeded_documents.append(
            SeededDocument(
                key=entry["key"],
                document_id=document_id,
                active_version_id=active_version.id,
                chunk_count=chunk_count,
            )
        )

    return CorpusSeedResult(
        documents=seeded_documents,
        total_active_chunks=sum(doc.chunk_count for doc in seeded_documents),
        skipped_existing=skipped,
        newly_seeded=newly_seeded,
    )


def verify_corpus_size(result: CorpusSeedResult) -> None:
    """Refuse to evaluate over a corpus too small to test retrieval depth."""
    if result.total_active_chunks <= MINIMUM_ACTIVE_CHUNKS:
        raise CorpusError(
            f"the corpus has {result.total_active_chunks} active chunk(s); "
            f"the retrieval window is {MINIMUM_ACTIVE_CHUNKS}, and the "
            f"corpus must be meaningfully larger than that or retrieval "
            f"metrics cannot discriminate a good ranking from a bad one"
        )


__all__ = [
    "DEFAULT_MANIFEST",
    "MINIMUM_ACTIVE_CHUNKS",
    "PINNED_CHUNK_OVERLAP_TOKENS",
    "PINNED_CHUNK_SIZE_TOKENS",
    "CorpusError",
    "CorpusSeedResult",
    "SeededDocument",
    "ensure_corpus_seeded",
    "load_manifest",
    "verify_corpus_size",
]
