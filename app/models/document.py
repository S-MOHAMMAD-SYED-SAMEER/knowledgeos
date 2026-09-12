"""Documents and their versions.

A document is the thing people refer to — "the Access SOP" — and it does not
change. A version is one concrete artefact of it, with a file behind it, and
versions are what retrieval actually reaches. That split is what lets the
approval chain in v2 of a policy replace the one in v1 while v1 stays
answerable for an audit question about last quarter.

The rule that makes it work is a single partial unique index: at most one
version of a document may be `active` at a time. It is enforced by PostgreSQL
rather than by whichever code path happens to promote a version, because the
alternative is a race between two uploads and a knowledge base that answers
from two versions at once.

Milestone 1 writes none of these rows — there is no upload yet. What it does
is settle the shape they must have before anything writes them.
"""

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# The provenance of a document: where it came from, not what format it is in.
# Format belongs to the version, because v2 of a policy may arrive as DOCX
# when v1 was a PDF. Connectors (Drive, Notion) are explicitly not in v1, so
# "upload" is the only value there is.
SOURCE_TYPE_UPLOAD = "upload"
SOURCE_TYPES = (SOURCE_TYPE_UPLOAD,)

# sha256, lower-case hex.
CHECKSUM_PATTERN = "^[0-9a-f]{64}$"


class VersionStatus(enum.StrEnum):
    """Where a version sits in its life.

    `StrEnum` on the Python side, `VARCHAR` + `CHECK` on the PostgreSQL side,
    deliberately rather than a native `ENUM` type. A native enum makes the
    partial index predicate clumsier to write and is awkward to extend — a new
    value needs `ALTER TYPE`, which cannot be used in the same transaction
    that adds it. A check constraint is one line of a migration.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            f"source_type IN ({', '.join(repr(value) for value in SOURCE_TYPES)})",
            name="source_type",
        ),
        # An array, not an object: tags are a flat set of labels, and the
        # metadata filter that reads them in a later milestone is a `@>`
        # containment test against a GIN index.
        CheckConstraint("jsonb_typeof(tags) = 'array'", name="tags_is_array"),
        Index("ix_documents_tags", "tags", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    title: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[str] = mapped_column(
        String(32), default=SOURCE_TYPE_UPLOAD, server_default=SOURCE_TYPE_UPLOAD
    )

    # Both are filter columns, and both are the customer's vocabulary rather
    # than ours — so they are indexed but not constrained to a fixed set.
    # Nullable because not every document belongs to a department.
    department: Mapped[str | None] = mapped_column(String(128), index=True)
    category: Mapped[str | None] = mapped_column(String(128), index=True)

    # Never null. A nullable JSONB would put a COALESCE into every metadata
    # filter, and a filter that is wrong about an absent tag is exactly the
    # thing the metadata-correctness gate exists to catch.
    tags: Mapped[list[str]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Maintained by the ORM. Worth knowing: this does not fire for a bulk
    # `UPDATE` statement that bypasses the unit of work. Nothing does that
    # yet; the day something does, this becomes a trigger.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_number",
    )

    def __repr__(self) -> str:
        return f"<Document {self.title!r}>"


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    __table_args__ = (
        UniqueConstraint("document_id", "version_number"),
        CheckConstraint("version_number >= 1", name="version_number_positive"),
        CheckConstraint(
            "status IN ('draft', 'active', 'superseded')", name="status"
        ),
        # Only when present: the checksum is of normalized text, and
        # normalization happens at parsing, which is a later milestone.
        CheckConstraint(
            f"checksum IS NULL OR checksum ~ '{CHECKSUM_PATTERN}'",
            name="checksum_is_sha256_hex",
        ),
        CheckConstraint(
            "page_count IS NULL OR page_count > 0", name="page_count_positive"
        ),
        # The rule the whole versioning model rests on: one active version per
        # document, enforced by the database. Partial, so any number of drafts
        # and superseded versions may sit beside the active one.
        Index(
            "uq_document_versions_one_active",
            "document_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # Indexed explicitly: PostgreSQL does not index a foreign key for you, and
    # listing a document's version history is the commonest read there is.
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[VersionStatus] = mapped_column(
        String(16), default=VersionStatus.DRAFT, server_default=VersionStatus.DRAFT
    )

    # sha256 of the normalized text, which does not exist until the document
    # has been parsed. Null until then, and never invented in the meantime.
    checksum: Mapped[str | None] = mapped_column(String(64))

    # What the uploader called it, kept as metadata only. What it is stored
    # under is a path this application chose.
    original_filename: Mapped[str] = mapped_column(String(512))
    storage_path: Mapped[str] = mapped_column(String(1024))

    # A calendar day, not an instant: "effective from 1 April" is a date, and
    # storing it as a timestamp invents a time nobody specified. Null when the
    # document does not state one.
    effective_date: Mapped[date | None] = mapped_column(Date)
    # Known only after parsing, and meaningless for a plain text file.
    page_count: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="versions")
    jobs: Mapped[list["IngestionJob"]] = relationship(  # noqa: F821
        back_populates="version", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentVersion v{self.version_number} {self.status}>"


__all__ = [
    "CHECKSUM_PATTERN",
    "SOURCE_TYPES",
    "SOURCE_TYPE_UPLOAD",
    "Document",
    "DocumentVersion",
    "VersionStatus",
]
