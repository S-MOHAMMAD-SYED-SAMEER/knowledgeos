"""`POST /query` — hybrid retrieval, evidence only.

No LLM call happens anywhere in this module, and none of the milestone 6-9
concepts do either: no reranking, no citation, no generation, no persisted
query record. Milestone 5 stops exactly where the specification's row for it
stops — "`/query` returning evidence only, no answer".

The query is embedded **here**, not inside `app/retrieval/`: that package's
layering rule forbids provider calls, and this is the one place above it
that is allowed to make one. The provider is milestone 4's real, cached
`BgeEmbeddingProvider`, reached through the same accessor the ingestion
runner uses — nothing here loads a second copy of the model, and nothing
here ever substitutes the fake. A test overrides `embedding_provider`, the
one FastAPI dependency this module exposes for that purpose.
"""

import logging
from typing import Annotated

import sqlalchemy.exc
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import (
    QueryCandidateOut,
    QueryCounts,
    QueryDocumentOut,
    QueryRequest,
    QueryResponse,
    QueryVersionOut,
)
from app.config import Settings, get_settings
from app.db.session import get_session
from app.ingestion.runner import get_embedding_provider
from app.parsing import normalize
from app.providers.embeddings import EmbeddingError, EmbeddingProvider
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieval"])


def embedding_provider() -> EmbeddingProvider:
    """The embedding provider this endpoint uses.

    A thin wrapper around milestone 4's cached accessor, so `/query` and the
    ingestion runner share one loaded model rather than each holding a copy.
    A test overrides this dependency to inject `FakeEmbeddingProvider`; the
    application itself never calls anything but this.
    """
    return get_embedding_provider()


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(embedding_provider)],
) -> QueryResponse:
    """Normalize, embed, retrieve. `200` even when nothing matches — an
    empty evidence set is a real answer at this layer, not a failure."""
    raw_query = request.query

    if len(raw_query) > settings.query_max_length:
        raise HTTPException(
            status_code=422,
            detail=(
                f"query exceeds the maximum length of "
                f"{settings.query_max_length} characters"
            ),
        )

    # `normalize()` strips outer whitespace, so a whitespace-only query
    # normalizes to the empty string and is caught by the same check as a
    # query that was empty to begin with. One rule, not two.
    normalized_query = normalize(raw_query)
    if not normalized_query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        query_vector = embeddings.embed([normalized_query])[0]
    except EmbeddingError as exc:
        # Never the query text, never a traceback: this is exactly the
        # boundary the specification's error-response rule exists for.
        logger.warning("Query embedding failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="the embedding model is unavailable"
        ) from exc

    filters = RetrievalFilters(
        document_id=request.filters.document_id,
        department=request.filters.department,
        category=request.filters.category,
        tags=tuple(request.filters.tags),
        include_superseded=request.filters.include_superseded,
    )

    try:
        result = retrieve(
            session,
            normalized_text=normalized_query,
            query_vector=query_vector,
            filters=filters,
            rrf_k=settings.rrf_k,
        )
    except sqlalchemy.exc.SQLAlchemyError as exc:
        logger.warning("Retrieval failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="the database is unavailable"
        ) from exc

    return QueryResponse(
        query=raw_query,
        normalized_query=normalized_query,
        filters=request.filters,
        candidates=[_candidate_out(candidate) for candidate in result.candidates],
        counts=QueryCounts(
            vector=result.vector_candidate_count,
            lexical=result.lexical_candidate_count,
            fused=result.fused_candidate_count,
            returned=len(result.candidates),
        ),
    )


def _candidate_out(candidate: RetrievedChunk) -> QueryCandidateOut:
    evidence = candidate.evidence
    return QueryCandidateOut(
        chunk_uid=evidence.chunk_uid,
        text=evidence.text,
        sequence=evidence.sequence,
        page=evidence.page,
        section=evidence.section,
        char_start=evidence.char_start,
        char_end=evidence.char_end,
        document=QueryDocumentOut(
            id=evidence.document_id,
            title=evidence.document_title,
            department=evidence.department,
            category=evidence.category,
            tags=list(evidence.tags),
        ),
        version=QueryVersionOut(
            id=evidence.version_id,
            version_number=evidence.version_number,
            status=evidence.version_status,
        ),
        lexical_rank=candidate.lexical_rank,
        vector_rank=candidate.vector_rank,
        rrf_score=candidate.rrf_score,
        final_rank=candidate.final_rank,
    )


__all__ = ["embedding_provider", "router"]
