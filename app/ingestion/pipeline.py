"""Turning one stored file into chunks.

The stages, and where this milestone stops:

    queued → parsing → chunking → indexing
                  ↓ (either stage)
                failed, with stage_error

`indexing` is where milestone 3 stopped and where milestone 4 picks up.
`run_job` takes a job from `queued` through parsing and chunking and leaves it
at `indexing`; `run_indexing` takes it from there to `ready`, and promotes its
version to `active`. They are separate claims, so a document takes two passes
of the runner.

A version becomes `active` only in `run_indexing`, and only once its vectors
are written — a version that can be retrieved is a version whose content is
really there.

**The normalized document is produced once** and reused for all three things
that depend on it — the checksum, the chunk text, and the character offsets.
Normalizing separately for each would let them disagree, and an offset that
does not index the text the checksum was taken over is worse than no offset.

Errors carry the stage and the failure class and nothing else. The
specification requires stage errors to be stored "without raw document
content", and a stage error is the one string from this pipeline that a person
will read.
"""

import logging
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.chunking import Chunk as ChunkData
from app.chunking import chunk_document
from app.config import Settings
from app.ingestion.validation import extension_of
from app.models import Chunk, DocumentVersion, IngestionJob, JobStatus
from app.parsing import ParseError, normalize, parser_for
from app.parsing.base import Block
from app.providers.embeddings import EmbeddingProvider
from app.storage import Storage, StorageError

logger = logging.getLogger(__name__)

# The separator placed between parsed blocks when they are joined into the
# document text. A blank line, so block boundaries are paragraph boundaries
# and the chunker can prefer them.
BLOCK_SEPARATOR = "\n\n"


class StageFailed(Exception):
    """A stage could not complete.

    `stage` is one of the job statuses, and `detail` is safe to store: it
    names what went wrong structurally and never quotes the document.
    """

    def __init__(self, stage: JobStatus, detail: str) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class NormalizedDocument:
    """One document, normalized once, with its blocks located inside it."""

    text: str
    blocks: tuple[Block, ...]
    offsets: tuple[tuple[int, int], ...]
    page_count: int | None


def parse_and_normalize(
    version: DocumentVersion, storage: Storage
) -> NormalizedDocument:
    """Read the stored file and produce the canonical text for this version.

    Each block is normalized on its own and then joined, so that the offsets
    recorded for a block are offsets into the same string the chunks are cut
    from. Normalization is idempotent, so normalizing the pieces and then the
    join does not change the result a second pass would give.
    """
    extension = extension_of(version.original_filename)
    try:
        parser = parser_for(extension)
    except ParseError as exc:
        raise StageFailed(JobStatus.PARSING, str(exc)) from exc

    try:
        handle = storage.open(version.storage_path)
    except StorageError as exc:
        # The key is not included: it describes where this system keeps a
        # file, and a stage error is readable through the API.
        raise StageFailed(
            JobStatus.PARSING, "the stored file could not be opened"
        ) from exc

    try:
        with handle:
            parsed = parser.parse(handle)
    except ParseError as exc:
        raise StageFailed(JobStatus.PARSING, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a parser must never escape
        raise StageFailed(
            JobStatus.PARSING, f"the file could not be parsed ({type(exc).__name__})"
        ) from exc

    pieces: list[str] = []
    kept: list[Block] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for block in parsed.blocks:
        body = normalize(block.text)
        if not body:
            continue
        if pieces:
            cursor += len(BLOCK_SEPARATOR)
        pieces.append(body)
        offsets.append((cursor, cursor + len(body)))
        kept.append(block)
        cursor += len(body)

    text = BLOCK_SEPARATOR.join(pieces)

    if not text.strip():
        # A scanned PDF with no text layer lands here. There is no OCR in this
        # system, and a version with no text can never answer anything, so
        # calling this a success would be a lie told once and believed later.
        raise StageFailed(
            JobStatus.PARSING, "the document contains no extractable text"
        )

    return NormalizedDocument(
        text=text,
        blocks=tuple(kept),
        offsets=tuple(offsets),
        page_count=parsed.page_count,
    )


def build_chunks(
    version: DocumentVersion, document: NormalizedDocument, settings: Settings
) -> list[ChunkData]:
    """Split the normalized document. Pure; no database, no storage."""
    try:
        chunks = chunk_document(
            document_version_id=version.id,
            text=document.text,
            blocks=document.blocks,
            block_offsets=document.offsets,
            size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
        )
    except ValueError as exc:
        raise StageFailed(JobStatus.CHUNKING, str(exc)) from exc

    if not chunks:
        raise StageFailed(JobStatus.CHUNKING, "the document produced no chunks")
    return chunks


def replace_chunks(
    session: Session, version: DocumentVersion, chunks: list[ChunkData]
) -> None:
    """Write this version's chunks, replacing whatever was there.

    Delete-then-write, scoped to one `document_version_id`, which is what the
    specification asks of a re-run. Because `chunk_uid` is derived from the
    version, the sequence and the text, the rows written are identical to the
    ones they replaced — a re-index produces the same rows rather than
    duplicates.

    Both statements are in the caller's transaction, so a failure leaves the
    version with the chunks it had, not half of a new set.
    """
    session.execute(delete(Chunk).where(Chunk.document_version_id == version.id))
    session.flush()

    session.add_all(
        [
            Chunk(
                document_version_id=version.id,
                sequence=chunk.sequence,
                chunk_uid=chunk.chunk_uid,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ]
    )
    session.flush()


def run_job(
    session: Session, job: IngestionJob, storage: Storage, settings: Settings
) -> IngestionJob:
    """Take one job from `queued` to `indexing`.

    The caller owns the transaction and commits. Every stage's status is
    written as it is entered, so a job that dies mid-stage says where it was.
    """
    version = session.get(DocumentVersion, job.document_version_id)
    if version is None:
        raise StageFailed(JobStatus.PARSING, "the document version no longer exists")

    job.status = JobStatus.PARSING
    session.flush()
    document = parse_and_normalize(version, storage)

    # The checksum is of the complete normalized text, which is what the
    # specification defines it as, and it is taken from the same string the
    # chunks are cut from.
    version.checksum = _checksum(document.text)
    # Real pages for a paginated format; null for everything else. Never
    # invented.
    version.page_count = document.page_count

    job.status = JobStatus.CHUNKING
    session.flush()
    chunks = build_chunks(version, document, settings)
    replace_chunks(session, version, chunks)

    logger.info(
        "Version %s: %d chunk(s) from %d block(s).",
        version.id,
        len(chunks),
        len(document.blocks),
    )

    # The handoff. Parsing and chunking are done; indexing is milestone 4's,
    # and `ready` must not be claimed before it has happened.
    job.status = JobStatus.INDEXING
    session.flush()
    return job


def run_indexing(
    session: Session, job: IngestionJob, embeddings: EmbeddingProvider
) -> IngestionJob:
    """Take one job from `indexing` to `ready`, and its version to `active`.

    The ordering is the point, and it is deliberate:

    1. read the version and its chunk texts;
    2. **end that read transaction**, and only then generate the vectors —
       the specification requires index *writes* to be transactional, not
       inference, and a transaction held open across model inference holds a
       connection and a snapshot for no benefit;
    3. open the write transaction: write the vectors, flip the job to
       `ready`, promote the version;
    4. commit, by the caller.

    **Precondition: the session must have no unwritten changes.** Step 2 ends
    the read transaction by rolling it back, which would discard them. The
    runner satisfies this by committing the claim before any work begins; a
    caller that does not is refused below rather than quietly losing work.

    `tsv` is never written here. PostgreSQL maintains it from `text`, so it
    is already correct for every chunk and writing to a generated column is
    an error — one fewer thing that can succeed halfway.
    """
    from app.indexing import (
        IndexingFailed,
        chunks_to_embed,
        generate_embeddings,
        texts_of,
        write_index,
    )

    if session.new or session.dirty or session.deleted:
        raise StageFailed(
            JobStatus.INDEXING,
            "the indexing stage was entered with unwritten changes",
        )

    version = session.get(DocumentVersion, job.document_version_id)
    if version is None:
        raise StageFailed(
            JobStatus.INDEXING, "the document version no longer exists"
        )
    version_id = version.id

    # Plain strings, so nothing below can reach the database by lazily
    # refreshing a row.
    texts = texts_of(chunks_to_embed(session, version_id))

    # The read is done. Let the transaction go before the model runs.
    session.rollback()

    try:
        vectors = generate_embeddings(texts, embeddings)
    except IndexingFailed as exc:
        raise StageFailed(JobStatus.INDEXING, str(exc)) from exc

    # The write transaction opens here, with the rows read again inside it.
    version = session.get(DocumentVersion, version_id)
    if version is None:
        raise StageFailed(
            JobStatus.INDEXING, "the document version no longer exists"
        )
    chunks = chunks_to_embed(session, version_id)

    try:
        write_index(session, version, chunks, vectors)
    except IndexingFailed as exc:
        raise StageFailed(JobStatus.INDEXING, str(exc)) from exc

    job.status = JobStatus.READY
    session.flush()

    logger.info(
        "Version %s indexed: %d chunk(s) embedded with %s.",
        version_id,
        len(chunks),
        embeddings.model_name,
    )
    return job


def _checksum(text: str) -> str:
    from hashlib import sha256

    return sha256(text.encode("utf-8")).hexdigest()


def pending_chunk_count(session: Session, version_id) -> int:
    from sqlalchemy import func

    return int(
        session.execute(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_version_id == version_id
            )
        ).scalar_one()
    )


__all__ = [
    "BLOCK_SEPARATOR",
    "NormalizedDocument",
    "StageFailed",
    "build_chunks",
    "parse_and_normalize",
    "pending_chunk_count",
    "replace_chunks",
    "run_indexing",
    "run_job",
]
