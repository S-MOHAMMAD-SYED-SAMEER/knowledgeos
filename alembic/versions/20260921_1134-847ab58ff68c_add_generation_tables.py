"""add generation tables

The specification's own three tables (§5), all three deferred since
milestone 5's locked decision and created together here because milestone
8 is the first milestone that has anything to write into them.

**`queries`** — the query as typed and normalized, its filters, and what
milestone 8 genuinely produces: `model`, `prompt_version`, `input_tokens`,
`output_tokens`. The four `*_ms` timing columns and `cost_usd` are created
nullable and stay `NULL` in every row this milestone writes — the
specification requires unknown pricing to raise a config error rather than
silently become zero (§14), and this milestone measures no stage latency at
all. They exist now only so milestone 9's observability work does not need
a second migration to fill in what the specification already named here.

**`retrieved_chunks`** — the retrieval and reranking snapshot for one query,
one row per chunk, composite primary key `(query_id, chunk_id)` rather than
a surrogate id: the specification gives this table no id column of its own,
and the natural key already is a query's chunk, never two rows for the
same pair. `chunk_id` is a real foreign key to `chunks.id`, resolved from
`chunk_uid` at write time rather than carried on milestone 5's own
`ChunkEvidence`, which is locked and untouched.

**`answers`** — one row per query, enforced by a unique constraint on
`query_id`: the specification states no explicit uniqueness rule for this
table, but "one answer belongs to the query" is the actual invariant this
project's own locked decisions state, and a retried request writing a
second row would silently turn "the answer" into "an answer, arbitrarily
chosen" for anything that reads it back. Every row this milestone writes
has `citation_valid = True` by construction: an answer whose citations do
not validate is a rejected generation, never a persisted one (see
`app/generation/generator.py`).

Nothing in migrations 0001-0005 is altered.

Revision ID: 847ab58ff68c
Revises: 57746e5929c3
Create Date: 2026-09-21 11:34:30.200581
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "847ab58ff68c"
down_revision: str | Sequence[str] | None = "57746e5929c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "queries",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("query_text", sa.String(length=4096), nullable=False),
        sa.Column("normalized_text", sa.String(length=4096), nullable=False),
        sa.Column(
            "filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        # Milestone 9's own columns, nullable and unfilled here.
        sa.Column("retrieval_ms", sa.Float(), nullable=True),
        sa.Column("rerank_ms", sa.Float(), nullable=True),
        sa.Column("llm_ms", sa.Float(), nullable=True),
        sa.Column("total_ms", sa.Float(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_queries")),
    )
    op.create_table(
        "answers",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("query_id", sa.UUID(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column(
            "citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("citation_valid", sa.Boolean(), nullable=False),
        sa.Column("grounded", sa.Boolean(), nullable=False),
        sa.Column(
            "grounding_detail",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name=op.f("fk_answers_query_id_queries"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_answers")),
        sa.UniqueConstraint("query_id", name=op.f("uq_answers_query_id")),
    )
    op.create_table(
        "retrieved_chunks",
        sa.Column("query_id", sa.UUID(), nullable=False),
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("lexical_rank", sa.Integer(), nullable=True),
        sa.Column("vector_rank", sa.Integer(), nullable=True),
        sa.Column("rrf_score", sa.Float(), nullable=False),
        sa.Column("rerank_score", sa.Float(), nullable=False),
        sa.Column("final_rank", sa.Integer(), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["chunks.id"],
            name=op.f("fk_retrieved_chunks_chunk_id_chunks"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["query_id"],
            ["queries.id"],
            name=op.f("fk_retrieved_chunks_query_id_queries"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "query_id", "chunk_id", name=op.f("pk_retrieved_chunks")
        ),
    )


def downgrade() -> None:
    op.drop_table("retrieved_chunks")
    op.drop_table("answers")
    op.drop_table("queries")
