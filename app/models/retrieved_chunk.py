"""One chunk retrieved for one query — the retrieval and reranking
snapshot, and whether it was selected to reach the model.

`retrieved_chunks` is the specification's own table (§5). No surrogate
`id`: the specification gives this table no primary key of its own beyond
the two foreign keys, and `(query_id, chunk_id)` is exactly the natural key
a retrieval snapshot has — one row per chunk per query, never two.

`chunk_id` is resolved from the retrieval evidence's `chunk_uid` at
persistence time (`app/generation/persistence.py`), by one indexed lookup
against `chunks.chunk_uid`'s existing unique constraint. It is not carried
on `ChunkEvidence` itself — milestone 5's own type, locked, and never
widened to serve this milestone's need.
"""

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RetrievedChunk(Base):
    __tablename__ = "retrieved_chunks"

    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )

    lexical_rank: Mapped[int | None] = mapped_column(Integer)
    vector_rank: Mapped[int | None] = mapped_column(Integer)
    rrf_score: Mapped[float] = mapped_column(Float)
    rerank_score: Mapped[float] = mapped_column(Float)
    final_rank: Mapped[int] = mapped_column(Integer)
    selected: Mapped[bool] = mapped_column(Boolean)

    def __repr__(self) -> str:
        return (
            f"<RetrievedChunk query={self.query_id} chunk={self.chunk_id} "
            f"selected={self.selected}>"
        )


__all__ = ["RetrievedChunk"]
