"""Vector retrieval: pgvector cosine distance over `chunks.embedding`.

No embedding happens here. This module receives a vector that something else
already produced and never imports a provider — that is what the
specification's layering rule requires of `app/retrieval/`, and it is also
what makes this function callable over fixture data: give it any 384-float
list and a session, and it runs.

Cosine distance, not L2 or inner product, because the vectors milestone 4
stores are **not normalized to unit length** — the specification says
nothing about normalising them, and cosine distance is the one of the three
pgvector operators that stays correct regardless.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentVersion
from app.retrieval.filters import RetrievalFilters, predicates

# The specification's number: "Retrieve top 50 from each". Fixed, not a
# setting — a caller-tunable limit would make that sentence untrue on
# request, and nothing in the specification says it should be configurable.
CANDIDATE_LIMIT = 50


@dataclass(frozen=True)
class ChunkEvidence:
    """Everything a citation or an answer might need to point at one chunk.

    Carries the chunk's own fields and enough of its document and version to
    render and to filter on, without a second query. `storage_path` is
    deliberately not one of them — it describes where this system keeps a
    file, which is nobody outside it needs to know.
    """

    chunk_uid: str
    text: str
    sequence: int
    page: int | None
    section: str | None
    char_start: int
    char_end: int
    token_count: int
    document_id: uuid.UUID
    document_title: str
    department: str | None
    category: str | None
    tags: tuple[str, ...]
    version_id: uuid.UUID
    version_number: int
    version_status: str


@dataclass(frozen=True)
class Candidate:
    """One chunk found by one retrieval channel, at its rank in that channel.

    `rank` is 1-based, `score` is this channel's own raw number — cosine
    distance here, `ts_rank_cd` in the lexical channel. Fusion never compares
    the two directly: only rank position crosses the boundary between
    channels, which is the specification's whole reason for choosing RRF
    over score normalization.
    """

    evidence: ChunkEvidence
    rank: int
    score: float


def search(
    session: Session,
    *,
    query_vector: list[float],
    filters: RetrievalFilters,
    limit: int = CANDIDATE_LIMIT,
) -> list[Candidate]:
    """The nearest `limit` chunks to `query_vector`, filtered and ordered.

    `embedding IS NOT NULL` is written explicitly rather than relied on. The
    milestone 4 invariant (`active ⇒ indexed`) means it excludes nothing
    today for an active or superseded version, but a query that assumed the
    invariant instead of stating it would break silently if that ever
    stopped being true.
    """
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

    stmt = (
        select(Chunk, DocumentVersion, Document, distance)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Chunk.embedding.is_not(None))
        .where(*predicates(filters))
        # Ascending: 0 is identical, 2 is opposite. `chunk_uid` breaks ties so
        # the same corpus and the same query always come back in the same
        # order — required for the fixed final ranking retrieval promises.
        .order_by(distance.asc(), Chunk.chunk_uid.asc())
        .limit(limit)
    )

    rows = session.execute(stmt).all()
    return [
        Candidate(
            evidence=evidence_of(chunk, version, document),
            rank=index + 1,
            score=float(row_distance),
        )
        for index, (chunk, version, document, row_distance) in enumerate(rows)
    ]


def evidence_of(
    chunk: Chunk, version: DocumentVersion, document: Document
) -> ChunkEvidence:
    """Build one chunk's evidence record from its three joined rows.

    Shared with `lexical.py`, so both channels describe the same chunk
    identically regardless of which query found it.
    """
    return ChunkEvidence(
        chunk_uid=chunk.chunk_uid,
        text=chunk.text,
        sequence=chunk.sequence,
        page=chunk.page,
        section=chunk.section,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        token_count=chunk.token_count,
        document_id=document.id,
        document_title=document.title,
        department=document.department,
        category=document.category,
        tags=tuple(document.tags),
        version_id=version.id,
        version_number=version.version_number,
        version_status=str(version.status),
    )


__all__ = ["CANDIDATE_LIMIT", "Candidate", "ChunkEvidence", "evidence_of", "search"]
