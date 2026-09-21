"""Feedback on one generated answer.

`feedback` is the specification's own table (§5): `id, answer_id FK, rating
(helpful|not_helpful), reason, created_at`. Append-only — the specification
states no uniqueness rule, and a second opinion on the same answer, from the
same or a different reader, is not an error.

`rating` is validated at two layers, deliberately: the pydantic `Literal`
type at the API boundary (`app/api/schemas.py::FeedbackIn`), and the
database `CHECK` constraint here — so a write that bypassed the API (a
script against the database, a future second caller) still cannot produce
a value outside the specification's two.
"""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

RATING_HELPFUL = "helpful"
RATING_NOT_HELPFUL = "not_helpful"
RATINGS = (RATING_HELPFUL, RATING_NOT_HELPFUL)

# The specification names no limit. This exists for the same reason
# `query_max_length` does (`app/config.py`): an explicit bound rather than
# an unbounded `Text` column that lets one reader's free-text answer grow
# without limit.
MAX_REASON_LENGTH = 2000


class Feedback(Base):
    __tablename__ = "feedback"

    __table_args__ = (
        CheckConstraint(
            f"rating IN ({', '.join(repr(value) for value in RATINGS)})",
            name="rating",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # Indexed explicitly, the same rule `document_versions.document_id`
    # already follows: PostgreSQL does not index a foreign key on its own,
    # and reading an answer's feedback back is the commonest query this
    # table will ever serve.
    answer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("answers.id", ondelete="CASCADE"),
        index=True,
    )
    rating: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(MAX_REASON_LENGTH))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Feedback {self.rating} answer={self.answer_id}>"


__all__ = [
    "MAX_REASON_LENGTH",
    "RATINGS",
    "RATING_HELPFUL",
    "RATING_NOT_HELPFUL",
    "Feedback",
]
