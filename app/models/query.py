"""One user query and what it took to answer it.

`queries` is the specification's own table (§5): the query as typed, its
normalized form, the filters it ran under, which model and prompt version
answered it, per-stage timings, token counts, and cost. Milestone 8 is the
first to write a row here — and writes only what it genuinely produces:
`query_text` through `output_tokens`. The four `*_ms` timing columns and
`cost_usd` stay `NULL` in every row this milestone writes: the
specification requires unknown pricing to raise a config error rather than
silently become zero (§14), and this milestone measures no stage latency at
all — that is milestone 9's observability work, not invented here to fill
a column early.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, Numeric, String, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    query_text: Mapped[str] = mapped_column(String(4096))
    normalized_text: Mapped[str] = mapped_column(String(4096))
    # The same shape `QueryFiltersIn` serializes to — stored so a query is
    # reproducible from its own row.
    filters: Mapped[dict] = mapped_column(JSONB)

    # Milestone 8's own columns: which provider and which prompt answered
    # this query, and what it cost in tokens. `model`/`prompt_version` are
    # null only for a pre-LLM abstention, where no provider call was ever
    # made.
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)

    # Milestone 9's own columns. Always NULL as of this milestone — see the
    # module docstring.
    retrieval_ms: Mapped[float | None] = mapped_column(Float)
    rerank_ms: Mapped[float | None] = mapped_column(Float)
    llm_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float | None] = mapped_column(Float)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(12, 6))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Query {self.id}>"


__all__ = ["Query"]
