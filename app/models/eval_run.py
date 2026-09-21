"""One recorded evaluation run: what was measured, and with what config.

`eval_runs` is the specification's own table, unchanged from its
definition: `id, suite, prompt_version, config (JSONB), metrics (JSONB),
created_at`. Milestone 7 is the first milestone to write to it, and the
first suite it records is `"retrieval"` — `prompt_version` stays null for
that suite, since retrieval evaluation has no prompt; a generation suite
(milestone 9) is what will eventually give this column a value.

Every AI-generated number that ends up in a document is required to trace
back to a row here (§18) — that is the whole reason this table exists rather
than a number typed once and never checked. `config` and `metrics` are both
JSONB rather than a fixed set of columns because the specification gives
neither a schema: what a "config" or a "metrics" object contains differs by
suite, and a suite that does not exist yet cannot have its columns designed
today.
"""

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# The only suite milestone 7 ever writes. Not enforced as a CHECK against a
# fixed list of every suite name: milestone 9's answer suite arrives with
# its own name later, and a list here would have to be edited for every
# future milestone that adds one. The one rule the database enforces is
# that the column is never empty.
RETRIEVAL_SUITE = "retrieval"


class EvalRun(Base):
    __tablename__ = "eval_runs"

    __table_args__ = (CheckConstraint("suite <> ''", name="suite_not_empty"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # Which evaluation this row is. "retrieval" is the only value this
    # milestone ever writes.
    suite: Mapped[str] = mapped_column(String(64))
    # Null for the retrieval suite, which has no prompt. A generation suite
    # sets this to record which prompt version it ran.
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    # What was run: model names, chunk configuration, candidate limits,
    # question-set size — whatever this suite's own run needed pinned down
    # to be reproducible. The shape belongs to the suite, not to this table.
    config: Mapped[dict] = mapped_column(JSONB)
    # The measured numbers this run actually produced. Written only after
    # the harness produces them (specification, §13) — never before, and
    # never fabricated.
    metrics: Mapped[dict] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<EvalRun {self.suite} {self.created_at}>"


__all__ = ["RETRIEVAL_SUITE", "EvalRun"]
