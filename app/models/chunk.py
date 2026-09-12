"""A chunk: one retrievable piece of one version of a document.

Chunks are what retrieval will actually search, and what a citation will
eventually point at. Milestone 3 produced and stored them; milestone 4 makes
them findable, and adds the two columns that do it.

`embedding` is a 384-dimensional vector, nullable because a chunk exists from
the moment it is cut and is embedded a stage later — and because milestone 3
wrote chunks before embeddings existed at all.

`tsv` is **maintained by PostgreSQL**, not by this application: it is a
generated stored column over `text`. That means it cannot drift from the text
it describes and cannot fail independently of the row write, which is one
fewer half-indexed state to reason about. It is a Postgres full-text search
vector — **not BM25**, which is a different ranking model this system does not
implement and must not be described as.

Two unique constraints, both from the specification, and they do different
jobs. `(document_version_id, sequence)` keeps a version's chunks an ordered
sequence with no gaps or repeats. `chunk_uid` is globally unique and is what
makes re-indexing idempotent: because the identifier is derived from the
version, the position and the text, a re-run writes rows that are identical
to the ones it replaced rather than duplicating them.
"""

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.providers.embeddings import DIMENSIONS

# The expression PostgreSQL computes for every row. The two-argument form of
# `to_tsvector` is IMMUTABLE, which is what a generated column requires; the
# one-argument form depends on a session setting and is only STABLE.
TSV_EXPRESSION = "to_tsvector('english', text)"


class Chunk(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint("document_version_id", "sequence"),
        UniqueConstraint("chunk_uid"),
        CheckConstraint("sequence >= 0", name="sequence_non_negative"),
        CheckConstraint("token_count >= 0", name="token_count_non_negative"),
        CheckConstraint("char_end >= char_start", name="offsets_ordered"),
        CheckConstraint("page IS NULL OR page > 0", name="page_positive"),
        # GIN over the full-text vector. The specification names no index
        # type for it — GIN is the ordinary choice for a tsvector and is a
        # physical decision rather than a behavioural one.
        #
        # There is deliberately **no index on `embedding`**: the
        # specification says exact vector search is fast enough at v1 corpus
        # size and that no HNSW index may be added until a measured latency
        # number justifies it. No such number can exist before retrieval and
        # evaluation exist.
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        index=True,
    )
    # Position within the version, from zero.
    sequence: Mapped[int] = mapped_column(Integer)
    # sha256(version : sequence : text)[:32] — see app/chunking/uid.py.
    chunk_uid: Mapped[str] = mapped_column(String(32))

    text: Mapped[str] = mapped_column(Text)

    # The page the chunk **starts** on, for formats that have real pages, and
    # null for everything else. A chunk may run past the end of that page;
    # the specification gives it one page column and inventing the rest would
    # be worse than recording where it began.
    page: Mapped[int | None] = mapped_column(Integer)
    # The nearest preceding heading, where the format has headings. Null for
    # PDF and plain text, which structurally do not provide one.
    section: Mapped[str | None] = mapped_column(String(512))

    # Half-open `[char_start, char_end)` into the **complete normalized
    # document text**, so `document_text[char_start:char_end] == text`.
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # Whitespace-delimited words. See app/chunking/chunker.py on why.
    token_count: Mapped[int] = mapped_column(Integer)

    # Written at the indexing stage, so null for a chunk that has been cut
    # but not yet embedded. The width is the provider's, and a mismatch is
    # refused before it reaches the database.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(DIMENSIONS))

    # Maintained by PostgreSQL from `text`. Read-only from here: writing to a
    # generated column is an error, which is exactly the guarantee wanted.
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        sa.Computed(TSV_EXPRESSION, persisted=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    version: Mapped["DocumentVersion"] = relationship(  # noqa: F821
        back_populates="chunks"
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.sequence} of {self.document_version_id}>"


__all__ = ["TSV_EXPRESSION", "Chunk"]
