"""add ingestion jobs

One table. A job records that a version is waiting to be turned into
searchable text, and — from milestone 3 onwards — how far that got.

Two things this table deliberately does not have.

**No unique constraint on `document_version_id`.** Re-indexing a version
creates a second job for it, which is a milestone-4 feature; a unique
constraint here would block it before it was written.

**No `created_at`.** `started_at` and `completed_at` are set by the runner, so
a job created here carries neither — the specification lists exactly these
columns and this migration adds exactly these columns.

Nothing in milestone 1's schema is touched.

Revision ID: 45c43eca68b2
Revises: 70c2f52e54ba
Create Date: 2026-09-12 16:07:43.075244
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "45c43eca68b2"
down_revision: str | Sequence[str] | None = "70c2f52e54ba"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_version_id", sa.UUID(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="queued", nullable=False
        ),
        # Stage context only. The specification is explicit that errors are
        # stored without raw document content.
        sa.Column("stage_error", sa.Text(), nullable=True),
        sa.Column(
            "attempts", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        # Both written by the milestone-3 runner; null for every job this
        # milestone creates.
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'parsing', 'chunking', 'indexing', "
            "'ready', 'failed')",
            name=op.f("ck_ingestion_jobs_status"),
        ),
        sa.CheckConstraint(
            "attempts >= 0", name=op.f("ck_ingestion_jobs_attempts_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name=op.f("fk_ingestion_jobs_document_version_id_document_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
    )
    op.create_index(
        op.f("ix_ingestion_jobs_document_version_id"),
        "ingestion_jobs",
        ["document_version_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_ingestion_jobs_document_version_id"), table_name="ingestion_jobs"
    )
    op.drop_table("ingestion_jobs")
