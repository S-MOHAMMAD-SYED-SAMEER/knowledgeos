"""add feedback

The specification's own last deferred table (§5): `feedback — id, answer_id
FK, rating (helpful|not_helpful), reason, created_at`. Milestone 10 is the
first milestone with a feedback endpoint to write it from
(`POST /answers/{id}/feedback`, §11).

Append-only, not one-per-answer: the specification states no uniqueness
rule, and a second opinion on the same answer is not an error, so
`answer_id` is an indexed foreign key rather than a unique one.

`rating` is constrained to the specification's own two values by a `CHECK`
constraint — the same discipline `document_versions.status` and
`documents.source_type` already use for their own fixed vocabularies,
rather than a native `ENUM` type that would need `ALTER TYPE` to extend.

Nothing in migrations 0001-0006 is altered.

Revision ID: 9176dca9861f
Revises: 847ab58ff68c
Create Date: 2026-09-21 15:39:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9176dca9861f"
down_revision: str | Sequence[str] | None = "847ab58ff68c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("answer_id", sa.UUID(), nullable=False),
        sa.Column("rating", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.String(length=2000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating IN ('helpful', 'not_helpful')", name=op.f("ck_feedback_rating")
        ),
        sa.ForeignKeyConstraint(
            ["answer_id"],
            ["answers.id"],
            name=op.f("fk_feedback_answer_id_answers"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
    )
    op.create_index(
        op.f("ix_feedback_answer_id"), "feedback", ["answer_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_answer_id"), table_name="feedback")
    op.drop_table("feedback")
