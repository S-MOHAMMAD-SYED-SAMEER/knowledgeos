"""add chunks

The table milestone 3 fills: one row per retrievable piece of one version of a
document, with the text, its position, and whatever page and section the
parser genuinely knew.

Two columns from the specification's row are deliberately **absent**:
`embedding vector(384)` and `tsv tsvector`. Both belong to milestone 4, which
owns the embedding provider and tsvector maintenance — a vector column this
milestone cannot populate would be a promise the code has not made. They
arrive with the migration that can fill them.

The two unique constraints do different jobs. `(document_version_id,
sequence)` keeps a version's chunks a gapless ordered sequence. `chunk_uid` is
globally unique and is what makes re-indexing idempotent: the identifier is
derived from the version, the position and the text, so re-running a job
rewrites identical rows rather than duplicating them.

Nothing in milestone 1 or 2's schema is touched.

Revision ID: 2e809bd3b68b
Revises: 45c43eca68b2
Create Date: 2026-09-12 16:49:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2e809bd3b68b"
down_revision: str | Sequence[str] | None = "45c43eca68b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('chunks',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('document_version_id', sa.UUID(), nullable=False),
    sa.Column('sequence', sa.Integer(), nullable=False),
    sa.Column('chunk_uid', sa.String(length=32), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('page', sa.Integer(), nullable=True),
    sa.Column('section', sa.String(length=512), nullable=True),
    sa.Column('char_start', sa.Integer(), nullable=False),
    sa.Column('char_end', sa.Integer(), nullable=False),
    sa.Column('token_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('char_end >= char_start', name=op.f('ck_chunks_offsets_ordered')),
    sa.CheckConstraint('page IS NULL OR page > 0', name=op.f('ck_chunks_page_positive')),
    sa.CheckConstraint('sequence >= 0', name=op.f('ck_chunks_sequence_non_negative')),
    sa.CheckConstraint('token_count >= 0', name=op.f('ck_chunks_token_count_non_negative')),
    sa.ForeignKeyConstraint(['document_version_id'], ['document_versions.id'], name=op.f('fk_chunks_document_version_id_document_versions'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_chunks')),
    sa.UniqueConstraint('chunk_uid', name=op.f('uq_chunks_chunk_uid')),
    sa.UniqueConstraint('document_version_id', 'sequence', name=op.f('uq_chunks_document_version_id_sequence'))
    )
    op.create_index(op.f('ix_chunks_document_version_id'), 'chunks', ['document_version_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chunks_document_version_id'), table_name='chunks')
    op.drop_table('chunks')
