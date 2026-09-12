"""create documents and versions

Milestone 1's whole schema: the `vector` extension the retrieval half of this
project will be built on, and the two tables that hold what a document is and
which of its versions is current.

Two things here were written by hand rather than by autogenerate.

**The extension.** `CREATE EXTENSION IF NOT EXISTS vector` is idempotent, so
this migration is safe to run against a database where a superuser has already
installed it. That matters, because pgvector 0.6.0 is not a *trusted*
extension: an unprivileged role cannot create it, and the documented bootstrap
(see the README) runs it once as a superuser. Where the role does have the
privilege, this line does the work by itself.

**The downgrade leaves the extension alone.** Dropping `vector` would remove
it for everything else in that database, and `DROP EXTENSION` cascades to any
column that uses its type. Downgrading this migration means "remove
KnowledgeOS's two tables", not "uninstall a shared PostgreSQL feature". The
`IF NOT EXISTS` above makes a re-upgrade a no-op either way.

Revision ID: 70c2f52e54ba
Revises:
Create Date: 2026-09-12 13:18:44.348960
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "70c2f52e54ba"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column(
            "source_type",
            sa.String(length=32),
            server_default="upload",
            nullable=False,
        ),
        sa.Column("department", sa.String(length=128), nullable=True),
        sa.Column("category", sa.String(length=128), nullable=True),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(tags) = 'array'", name=op.f("ck_documents_tags_is_array")
        ),
        sa.CheckConstraint(
            "source_type IN ('upload')", name=op.f("ck_documents_source_type")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
    )
    op.create_index(
        op.f("ix_documents_category"), "documents", ["category"], unique=False
    )
    op.create_index(
        op.f("ix_documents_department"), "documents", ["department"], unique=False
    )
    # GIN, because the metadata filter that reads tags in a later milestone is
    # a containment test rather than an equality one.
    op.create_index(
        "ix_documents_tags", "documents", ["tags"], unique=False, postgresql_using="gin"
    )

    op.create_table(
        "document_versions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="draft", nullable=False
        ),
        sa.Column("checksum", sa.String(length=64), nullable=True),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "checksum IS NULL OR checksum ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_document_versions_checksum_is_sha256_hex"),
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'active', 'superseded')",
            name=op.f("ck_document_versions_status"),
        ),
        sa.CheckConstraint(
            "page_count IS NULL OR page_count > 0",
            name=op.f("ck_document_versions_page_count_positive"),
        ),
        sa.CheckConstraint(
            "version_number >= 1",
            name=op.f("ck_document_versions_version_number_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_document_versions_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_versions")),
        sa.UniqueConstraint(
            "document_id",
            "version_number",
            name=op.f("uq_document_versions_document_id_version_number"),
        ),
    )
    op.create_index(
        op.f("ix_document_versions_document_id"),
        "document_versions",
        ["document_id"],
        unique=False,
    )
    # At most one active version per document. Partial, so any number of
    # drafts and superseded versions may sit beside the active one — which is
    # what keeps history queryable for audit.
    op.create_index(
        "uq_document_versions_one_active",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_document_versions_one_active",
        table_name="document_versions",
        postgresql_where=sa.text("status = 'active'"),
    )
    op.drop_index(
        op.f("ix_document_versions_document_id"), table_name="document_versions"
    )
    op.drop_table("document_versions")
    op.drop_index("ix_documents_tags", table_name="documents", postgresql_using="gin")
    op.drop_index(op.f("ix_documents_department"), table_name="documents")
    op.drop_index(op.f("ix_documents_category"), table_name="documents")
    op.drop_table("documents")
    # The `vector` extension is deliberately not dropped: see the module
    # docstring.
