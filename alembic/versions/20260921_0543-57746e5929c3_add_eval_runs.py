"""add eval runs

The specification's own table, unchanged from its definition: `id, suite,
prompt_version, config (JSONB), metrics (JSONB), created_at`. Milestone 7 is
the first to write to it — one row per run of `python -m evals.run --suite
retrieval` that actually produces measured numbers.

`prompt_version` is nullable because the retrieval suite has no prompt; a
generation suite (milestone 9) is what will give it a value. `config` and
`metrics` are JSONB rather than fixed columns because their shape belongs to
whichever suite wrote them, and a suite that does not exist yet cannot have
its columns designed today.

No foreign key: unlike `queries`/`retrieved_chunks` (still deferred — see
milestone 5's locked decision), this table has nothing to reference and
nothing yet references it, so it can exist standalone.

Nothing in milestones 1 to 6 is altered.

Revision ID: 57746e5929c3
Revises: 3fdbad950293
Create Date: 2026-09-21 05:43:36.448292
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "57746e5929c3"
down_revision: str | Sequence[str] | None = "3fdbad950293"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "eval_runs",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("suite", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=True),
        sa.Column(
            "config", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("suite <> ''", name=op.f("ck_eval_runs_suite_not_empty")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_runs")),
    )


def downgrade() -> None:
    op.drop_table("eval_runs")
