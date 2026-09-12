"""The record of a version's journey from uploaded bytes to searchable text.

A job is created `queued` when a version is uploaded, and in this milestone
that is the only state it ever has: **the runner belongs to milestone 3.** The
row exists now because the boundary is what milestone 2 owes — something has
to say "this version is waiting", and `GET /ingestion/{job_id}` has to have
something to read.

The states are the specification's, and they are stages rather than a
lifecycle of the job object itself:

    queued → parsing → chunking → indexing → ready
                ↓ (from any stage)
              failed, with stage_error

`stage_error` records where it broke and why. It must never carry document
content — an error string is not a place to leak the text of a policy.

There is deliberately **no unique constraint** on `document_version_id`:
re-indexing a version creates a second job for it, which is a milestone-4
feature this table must not preclude.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class JobStatus(enum.StrEnum):
    """Where a job has got to.

    `StrEnum` in Python, `VARCHAR` + `CHECK` in PostgreSQL, matching the
    decision taken for version status in milestone 1: a check constraint is
    one line of a migration to change, where a native enum needs `ALTER TYPE`
    and cannot use the new value in the same transaction that adds it.
    """

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# Declared, not executed. Milestone 2 writes `queued` and stops; this is the
# map the milestone-3 runner has to satisfy, recorded here so it is a contract
# rather than something rediscovered later.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.PARSING, JobStatus.FAILED}),
    JobStatus.PARSING: frozenset({JobStatus.CHUNKING, JobStatus.FAILED}),
    JobStatus.CHUNKING: frozenset({JobStatus.INDEXING, JobStatus.FAILED}),
    JobStatus.INDEXING: frozenset({JobStatus.READY, JobStatus.FAILED}),
    JobStatus.READY: frozenset(),
    JobStatus.FAILED: frozenset({JobStatus.QUEUED}),
}

TERMINAL_STATUSES = frozenset({JobStatus.READY, JobStatus.FAILED})


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'parsing', 'chunking', 'indexing', "
            "'ready', 'failed')",
            name="status",
        ),
        CheckConstraint("attempts >= 0", name="attempts_non_negative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # Indexed because the runner will ask "what is outstanding for this
    # version", and because PostgreSQL does not index a foreign key for you.
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        String(16), default=JobStatus.QUEUED, server_default=JobStatus.QUEUED
    )
    # Stage context, never document content.
    stage_error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))

    # Both set by the runner, so both are null for every job this milestone
    # creates.
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    version: Mapped["DocumentVersion"] = relationship(  # noqa: F821
        back_populates="jobs"
    )

    def __repr__(self) -> str:
        return f"<IngestionJob {self.status}>"


__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATUSES",
    "IngestionJob",
    "JobStatus",
]
