"""`POST /query` — hybrid retrieval and reranking, evidence only.

No LLM call happens anywhere in this module, and none of the milestone 8-9
concepts do either: no citation, no generation, no persisted query record.
Milestone 6 stops exactly where the specification's row for it stops —
rerank interface, local cross-encoder, passthrough, evidence selection, full
query pipeline wiring, and explicitly **no generation**.

Two providers are embedded/called **here**, not inside `app/retrieval/` or
`app/reranking/`'s own interface module: `app/retrieval/`'s layering rule
forbids provider calls, and this is the one place above it that is allowed
to make one for embeddings. Reranking's orchestration
(`app/reranking/pipeline.py`) is itself allowed to call a provider — the
layering rule does not name `app/reranking/` — but the provider instance
still arrives from here, the same dependency-injection pattern the embedding
provider already uses, so both are equally overridable in tests and equally
never substituted by the application itself.

The embedding provider is milestone 4's real, cached `BgeEmbeddingProvider`.
The reranking provider is this milestone's real, cached
`CrossEncoderRerankProvider`. Neither is ever replaced by a fake or a
passthrough outside a test — a test overrides `embedding_provider` and
`rerank_provider`, the two FastAPI dependencies this module exposes for that
purpose.
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
from app.providers.reranker import RerankError, RerankProvider
from app.reranking import get_rerank_provider
from app.reranking.pipeline import RerankedChunk, rerank
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve

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


def rerank_provider() -> RerankProvider:
    """The reranking provider this endpoint uses.

    A thin wrapper around this milestone's cached accessor
    (`app.reranking.get_rerank_provider`), mirroring `embedding_provider`
    exactly. A test overrides this dependency to inject the passthrough or
    the deterministic fake; the application itself never calls anything but
    this, which always resolves to the real local cross-encoder.
    """
    return get_rerank_provider()


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(embedding_provider)],
    reranker: Annotated[RerankProvider, Depends(rerank_provider)],
) -> QueryResponse:
    """Normalize, embed, retrieve, rerank. `200` even when nothing matches —
    an empty evidence set is a real answer at this layer, not a failure."""
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

    # Reranking is skipped, not attempted, when there is nothing to score —
    # calling the provider on an empty list would mean loading the real
    # cross-encoder for no reason. `rerank()` itself already short-circuits
    # on empty input; this early return additionally avoids constructing a
    # `QueryResponse` around a call that could only ever return `[]`.
    if not result.candidates:
        return QueryResponse(
            query=raw_query,
            normalized_query=normalized_query,
            filters=request.filters,
            candidates=[],
            counts=QueryCounts(
                vector=result.vector_candidate_count,
                lexical=result.lexical_candidate_count,
                fused=result.fused_candidate_count,
                returned=0,
            ),
        )

    try:
        reranked = rerank(normalized_query, result.candidates, reranker)
    except RerankError as exc:
        # Never the query text, never the chunk text, never a traceback:
        # the same boundary the embedding failure above observes.
        logger.warning("Reranking failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="the reranking model is unavailable"
        ) from exc

    return QueryResponse(
        query=raw_query,
        normalized_query=normalized_query,
        filters=request.filters,
        candidates=[_candidate_out(candidate) for candidate in reranked],
        counts=QueryCounts(
            vector=result.vector_candidate_count,
            lexical=result.lexical_candidate_count,
            fused=result.fused_candidate_count,
            returned=len(reranked),
        ),
    )


def _candidate_out(candidate: RerankedChunk) -> QueryCandidateOut:
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
        fusion_rank=candidate.fusion_rank,
        rerank_score=candidate.rerank_score,
        final_rank=candidate.final_rank,
    )


__all__ = ["embedding_provider", "rerank_provider", "router"]
