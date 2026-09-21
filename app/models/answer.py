"""One generated answer for one query.

`answers` is the specification's own table (§5). Unique on `query_id`: the
specification's column list states no explicit uniqueness rule, but "one
answer per query" is the actual invariant this project's locked decisions
state, and a retried request producing two answer rows for the same query
would silently turn `GET /queries/{id}`'s "the answer" into "an answer,
arbitrarily chosen."

`citations` is JSONB, storing the same flat `list[chunk_uid]` the model
declared and `app/generation/citations.py` already validated — not the
inline marker positions, which are recoverable from `answer_text` itself
and are not the specification's own field to persist a second time.

Every row this milestone writes has `citation_valid = True` — see
`app/generation/generator.py`'s module docstring on why an invalid citation
is a rejected generation, never a persisted one.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Answer(Base):
    __tablename__ = "answers"

    __table_args__ = (UniqueConstraint("query_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE")
    )

    answer_text: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(Boolean)
    citations: Mapped[list[str]] = mapped_column(JSONB)
    citation_valid: Mapped[bool] = mapped_column(Boolean)
    grounded: Mapped[bool] = mapped_column(Boolean)
    grounding_detail: Mapped[dict] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Answer {self.id} query={self.query_id}>"


__all__ = ["Answer"]
