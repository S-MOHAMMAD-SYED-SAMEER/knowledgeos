"""A chunk: one retrievable piece of one version of a document.

Chunks are what retrieval will actually search, and what a citation will
eventually point at. This milestone produces and stores them; making them
findable — embeddings and a full-text vector — is milestone 4, so the
`embedding` and `tsv` columns the specification's row includes are
deliberately **not** here yet. They arrive with the code that can fill them.

Two unique constraints, both from the specification, and they do different
jobs. `(document_version_id, sequence)` keeps a version's chunks an ordered
sequence with no gaps or repeats. `chunk_uid` is globally unique and is what
makes re-indexing idempotent: because the identifier is derived from the
version, the position and the text, a re-run writes rows that are identical
to the ones it replaced rather than duplicating them.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Chunk(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint("document_version_id", "sequence"),
        UniqueConstraint("chunk_uid"),
        CheckConstraint("sequence >= 0", name="sequence_non_negative"),
        CheckConstraint("token_count >= 0", name="token_count_non_negative"),
        CheckConstraint("char_end >= char_start", name="offsets_ordered"),
        CheckConstraint("page IS NULL OR page > 0", name="page_positive"),
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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    version: Mapped["DocumentVersion"] = relationship(  # noqa: F821
        back_populates="chunks"
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.sequence} of {self.document_version_id}>"


__all__ = ["Chunk"]
