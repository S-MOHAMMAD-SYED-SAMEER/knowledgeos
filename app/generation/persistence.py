"""Writing one query's `queries` / `retrieved_chunks` / `answers` rows, in
a single transaction, after generation has already succeeded.

No provider call happens here. By the time this module runs, the LLM call
(if any) has already completed and the answer has already passed citation
validation — this module never persists a query whose generation failed,
matching locked decision 14: no partial `queries`/`answers` persistence on
a generation, parsing, or validation failure.

`chunk_uid -> chunks.id` is resolved by one indexed lookup: milestone 5's
`ChunkEvidence` never carries the surrogate id, and this is what lets that
stay true without widening a locked type (see `app/models/retrieved_chunk.py`).

**Milestone 9 (D5).** The five observability columns
(`retrieval_ms`/`rerank_ms`/`llm_ms`/`total_ms`/`cost_usd`) are all
optional keyword-only parameters, defaulting to `None` -- a measured value
from the caller's own stage timing (`app.observability.timing.stage_timer`)
and cost calculation (`app.observability.pricing.compute_cost_usd`), never
computed here. Both `POST /query` and `evals/answers/suite.py` call this
same function; a caller that has not measured a stage, or could not price
it, passes `None` rather than a fabricated number.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.generation.generator import GeneratedAnswer
from app.models import Answer, Chunk, Query, RetrievedChunk
from app.reranking.pipeline import RerankedChunk


class PersistenceError(RuntimeError):
    """A retrieved chunk_uid had no corresponding row in `chunks` at
    persistence time.

    Should not happen in ordinary operation — a chunk cannot be retrieved
    unless it already exists — and is reported structurally rather than
    silently dropping the row or guessing at one.
    """


def persist_query(
    session: Session,
    *,
    query_text: str,
    normalized_text: str,
    filters: dict,
    candidates: list[RerankedChunk],
    answer: GeneratedAnswer,
    retrieval_ms: float | None = None,
    rerank_ms: float | None = None,
    llm_ms: float | None = None,
    total_ms: float | None = None,
    cost_usd: float | None = None,
) -> tuple[Query, Answer]:
    """Write `queries`, `retrieved_chunks`, and `answers` for one query, in
    one transaction, and commit.

    Callers must have already produced a successfully validated
    `GeneratedAnswer` — a generation failure never reaches this function at
    all, so there is no failure path here that would leave a partial row
    behind.

    The five observability parameters (milestone 9, D5) are the caller's
    own measured values, written through unchanged; a caller with nothing
    measured leaves them at their `None` default rather than this function
    guessing at zero.
    """
    query_row = Query(
        query_text=query_text,
        normalized_text=normalized_text,
        filters=filters,
        model=answer.model_name,
        prompt_version=answer.prompt_version,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        cost_usd=cost_usd,
    )
    session.add(query_row)
    session.flush()  # assigns query_row.id without ending the transaction

    if candidates:
        chunk_ids = _resolve_chunk_ids(session, candidates)
        selected_uids = frozenset(
            item.candidate.evidence.chunk_uid for item in answer.selected if item.selected
        )
        for candidate in candidates:
            session.add(
                RetrievedChunk(
                    query_id=query_row.id,
                    chunk_id=chunk_ids[candidate.evidence.chunk_uid],
                    lexical_rank=candidate.lexical_rank,
                    vector_rank=candidate.vector_rank,
                    rrf_score=candidate.rrf_score,
                    rerank_score=candidate.rerank_score,
                    final_rank=candidate.final_rank,
                    selected=candidate.evidence.chunk_uid in selected_uids,
                )
            )

    answer_row = Answer(
        query_id=query_row.id,
        answer_text=answer.answer_text,
        abstained=answer.abstained,
        citations=answer.citations,
        citation_valid=answer.citation_valid,
        grounded=answer.grounded,
        grounding_detail=answer.grounding_detail,
    )
    session.add(answer_row)

    session.commit()
    return query_row, answer_row


def _resolve_chunk_ids(
    session: Session, candidates: list[RerankedChunk]
) -> dict[str, uuid.UUID]:
    uids = [candidate.evidence.chunk_uid for candidate in candidates]
    rows = session.execute(
        select(Chunk.chunk_uid, Chunk.id).where(Chunk.chunk_uid.in_(uids))
    ).all()
    mapping = {uid: chunk_id for uid, chunk_id in rows}
    missing = set(uids) - set(mapping)
    if missing:
        raise PersistenceError(
            f"{len(missing)} retrieved chunk_uid(s) have no matching row in "
            "chunks at persistence time"
        )
    return mapping


__all__ = ["PersistenceError", "persist_query"]
