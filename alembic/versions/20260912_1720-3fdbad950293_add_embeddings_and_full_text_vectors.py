"""add embeddings and full text vectors

The two columns that make a chunk findable. Milestone 3 cut chunks; this is
what milestone 4 adds so they can be searched.

**`embedding vector(384)`, nullable.** Nullable for two reasons that both
matter: a chunk exists from the moment it is cut and is embedded a stage
later, and milestone 3 already wrote chunks with no embeddings at all. Those
rows are **not backfilled here** — a migration that ran model inference could
not run offline or deterministically, and would need a model this environment
cannot fetch. They are filled by the indexing workflow instead.

**`tsv tsvector`, generated and stored.** PostgreSQL maintains it from `text`,
so it cannot drift from the text it describes and cannot fail independently of
the row write. The two-argument `to_tsvector(regconfig, text)` is IMMUTABLE,
which is what a generated column requires; the one-argument form reads a
session setting, is only STABLE, and would be rejected outright. Adding the
column computes it for every existing row, so every chunk milestone 3 wrote
becomes full-text searchable immediately.

This is Postgres full-text search. It is **not BM25** — a different ranking
model that this system does not implement.

**A GIN index on `tsv`, and deliberately none on `embedding`.** The
specification states that at v1 corpus size exact vector search is fast enough
and exact, and that no HNSW index may be added until a measured latency number
justifies it — and no such number can exist before retrieval and evaluation
do. It names no index type for the full-text vector; GIN is the ordinary
choice for a tsvector and is a physical decision rather than a behavioural one.

Nothing in milestones 1 to 3 is altered.

Revision ID: 3fdbad950293
Revises: 2e809bd3b68b
Create Date: 2026-09-12 17:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "3fdbad950293"
down_revision: str | Sequence[str] | None = "2e809bd3b68b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Kept beside the model's own copy so the two cannot silently diverge; a test
# asserts they are the same string.
TSV_EXPRESSION = "to_tsvector('english', text)"


def upgrade() -> None:
    op.add_column("chunks", sa.Column("embedding", Vector(384), nullable=True))
    op.add_column(
        "chunks",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(TSV_EXPRESSION, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_column("chunks", "tsv")
    op.drop_column("chunks", "embedding")
