"""`POST /query` — retrieval, reranking, and (as of milestone 8) generation.

`/query` normalizes, embeds, retrieves, reranks, selects evidence, and
generates a cited, grounded answer — or an honest abstention — from it. No
generation-only or evidence-only mode exists: the specification's v1 API
(§11) lists no second query endpoint, and milestone 5's own scope line
("`/query` returning evidence only, no answer") describes what this
endpoint used to do before milestone 8 arrived to finish it, not a mode
that stays selectable afterward.

Three providers are injected here, not inside `app/retrieval/`,
`app/reranking/`, or `app/generation/citations.py`: those three packages'
layering rule (§4) forbids provider calls, and this is the one place above
all of them allowed to make one. `app/reranking/pipeline.py` and
`app/generation/generator.py` are themselves allowed to call a provider —
the layering rule does not name either — but the provider instance always
arrives from here, the same dependency-injection pattern every provider in
this module already uses, so all three are equally overridable in tests and
equally never substituted by the application itself.

The embedding provider is milestone 4's real, cached `BgeEmbeddingProvider`.
The reranking provider is milestone 6's real, cached
`CrossEncoderRerankProvider`. The generation provider is milestone 8's real,
cached `GeminiLLMProvider` — Google Gemini, not Anthropic; see the README's
milestone 8 section for why. None of the three is ever replaced by a fake or
a passthrough outside a test — a test overrides `embedding_provider`,
`rerank_provider`, and `llm_provider`, the three FastAPI dependencies this
module exposes for that purpose.

**Milestone 9 observability (D5).** Retrieval (embed + retrieve, timed as
one stage), reranking, and generation are each timed with
`app.observability.timing.stage_timer`, and the three durations plus their
sum are passed to `persist_query` unchanged — never recomputed or rounded
here. Cost is computed from the answer's own measured token counts via
`app.observability.pricing.compute_cost_usd`; an unconfigured or unknown
model's pricing is a `PricingError`, which is caught here and turns into a
`None` `cost_usd` rather than a `503` — pricing being unset must never break
a live query, only leave its cost unmeasured. No timing or cost value is
ever fabricated: every one is either a real measurement or `None`.
"""

import logging
import uuid
from typing import Annotated

import sqlalchemy.exc
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnswerDetailOut,
    QueryCandidateOut,
    QueryCounts,
    QueryDetailOut,
    QueryDocumentOut,
    QueryRequest,
    QueryResponse,
    QueryVersionOut,
    RetrievedChunkOut,
)
from app.config import Settings, get_settings
from app.db.session import get_session
from app.generation import get_llm_provider
from app.generation.citations import CitationError
from app.generation.generator import GenerationError, generate_answer
from app.generation.persistence import PersistenceError, persist_query
from app.ingestion.runner import get_embedding_provider
from app.models import Answer, Chunk, Query, RetrievedChunk
from app.observability.pricing import PricingError, compute_cost_usd
from app.observability.timing import stage_timer
from app.parsing import normalize
from app.providers.embeddings import EmbeddingError, EmbeddingProvider
from app.providers.llm import LLMError, LLMProvider
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

    A thin wrapper around milestone 6's cached accessor
    (`app.reranking.get_rerank_provider`), mirroring `embedding_provider`
    exactly. A test overrides this dependency to inject the passthrough or
    the deterministic fake; the application itself never calls anything but
    this, which always resolves to the real local cross-encoder.
    """
    return get_rerank_provider()


def llm_provider() -> LLMProvider:
    """The generation provider this endpoint uses.

    A thin wrapper around milestone 8's cached accessor
    (`app.generation.get_llm_provider`), mirroring the two providers above
    exactly. A test overrides this dependency to inject
    `FakeLLMProvider`; the application itself never calls anything but
    this, which always resolves to the real Gemini adapter.
    """
    return get_llm_provider()


@router.post("/query", response_model=QueryResponse)
def run_query(
    request: QueryRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingProvider, Depends(embedding_provider)],
    reranker: Annotated[RerankProvider, Depends(rerank_provider)],
    llm: Annotated[LLMProvider, Depends(llm_provider)],
) -> QueryResponse:
    """Normalize, embed, retrieve, rerank, select evidence, generate,
    validate, persist. `200` even for an abstained answer — insufficient
    evidence is a real, valid answer at this layer, never a failure."""
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

    with stage_timer() as retrieval_timer:
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

    # `rerank()` short-circuits on empty input without calling the
    # provider — safe to call unconditionally, including when nothing was
    # retrieved at all, which is exactly the zero-candidate case
    # `generate_answer` below turns into a pre-LLM abstention rather than a
    # separate branch here.
    with stage_timer() as rerank_timer:
        try:
            reranked = rerank(normalized_query, result.candidates, reranker)
        except RerankError as exc:
            # Never the query text, never the chunk text, never a traceback:
            # the same boundary the embedding failure above observes.
            logger.warning("Reranking failed (%s).", type(exc).__name__)
            raise HTTPException(
                status_code=503, detail="the reranking model is unavailable"
            ) from exc

    with stage_timer() as llm_timer:
        try:
            answer = generate_answer(
                query_text=normalized_query,
                candidates=reranked,
                llm=llm,
                abstention_threshold=settings.abstention_rerank_threshold,
                max_tokens=settings.llm_max_output_tokens,
            )
        except LLMError as exc:
            logger.warning("Generation failed (%s).", type(exc).__name__)
            raise HTTPException(
                status_code=503, detail="the language model is unavailable"
            ) from exc
        except (GenerationError, CitationError) as exc:
            # The specification's own words for an invalid citation: "answer
            # rejected." Never the raw model output, never the evidence, never
            # a traceback.
            logger.warning(
                "Generated answer failed validation (%s).", type(exc).__name__
            )
            raise HTTPException(
                status_code=502, detail="the model's output could not be validated"
            ) from exc

    # `answer.model_name is None` marks a pre-LLM abstention (zero
    # candidates, or the score threshold) -- generate_answer never called
    # the provider, so there is no real generation latency or cost to
    # report, and `llm_timer.ms` (a few microseconds of Python) would
    # misrepresent both.
    llm_ms = llm_timer.ms if answer.model_name is not None else None
    total_ms = retrieval_timer.ms + rerank_timer.ms + (llm_ms or 0.0)

    cost_usd = None
    if answer.model_name is not None:
        try:
            cost_usd = compute_cost_usd(
                model=answer.model_name,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                table=settings.llm_pricing_usd_per_million_tokens,
            )
        except PricingError:
            # Unconfigured pricing must never fail a live query (D7) --
            # only leave this one query's cost unmeasured.
            cost_usd = None

    try:
        query_row, answer_row = persist_query(
            session,
            query_text=raw_query,
            normalized_text=normalized_query,
            filters=request.filters.model_dump(mode="json"),
            candidates=reranked,
            answer=answer,
            retrieval_ms=retrieval_timer.ms,
            rerank_ms=rerank_timer.ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            cost_usd=cost_usd,
        )
    except (PersistenceError, sqlalchemy.exc.SQLAlchemyError) as exc:
        logger.warning("Persisting the query failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="the database is unavailable"
        ) from exc

    selected_uids = {
        item.candidate.evidence.chunk_uid for item in answer.selected if item.selected
    }

    return QueryResponse(
        query=raw_query,
        normalized_query=normalized_query,
        filters=request.filters,
        candidates=[_candidate_out(candidate, selected_uids) for candidate in reranked],
        counts=QueryCounts(
            vector=result.vector_candidate_count,
            lexical=result.lexical_candidate_count,
            fused=result.fused_candidate_count,
            returned=len(reranked),
        ),
        query_id=query_row.id,
        answer_id=answer_row.id,
        answer=answer.answer_text,
        abstained=answer.abstained,
        citations=answer.citations,
        citation_valid=answer.citation_valid,
        grounded=answer.grounded,
        grounding_detail=answer.grounding_detail,
        model=answer.model_name,
        prompt_version=answer.prompt_version,
    )


@router.get("/queries/{query_id}", response_model=QueryDetailOut)
def get_query(
    query_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> QueryDetailOut:
    """The persisted query, its retrieval/reranking snapshot, and its
    answer — read back from `queries`/`retrieved_chunks`/`answers` rather
    than recomputed. `404` for an id that does not exist, the same
    convention `GET /documents/{id}` already uses."""
    try:
        parsed_id = uuid.UUID(query_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="No such query") from exc

    query_row = session.get(Query, parsed_id)
    if query_row is None:
        raise HTTPException(status_code=404, detail="No such query")

    chunk_rows = session.execute(
        select(RetrievedChunk)
        .where(RetrievedChunk.query_id == parsed_id)
        .order_by(RetrievedChunk.final_rank)
    ).scalars().all()

    chunk_uid_by_id = dict(
        session.execute(
            select(Chunk.id, Chunk.chunk_uid).where(
                Chunk.id.in_([row.chunk_id for row in chunk_rows])
            )
        ).all()
    )

    answer_row = session.execute(
        select(Answer).where(Answer.query_id == parsed_id)
    ).scalar_one_or_none()

    return QueryDetailOut(
        id=query_row.id,
        query_text=query_row.query_text,
        normalized_text=query_row.normalized_text,
        filters=query_row.filters,
        model=query_row.model,
        prompt_version=query_row.prompt_version,
        input_tokens=query_row.input_tokens,
        output_tokens=query_row.output_tokens,
        created_at=query_row.created_at,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_uid=chunk_uid_by_id[row.chunk_id],
                lexical_rank=row.lexical_rank,
                vector_rank=row.vector_rank,
                rrf_score=row.rrf_score,
                rerank_score=row.rerank_score,
                final_rank=row.final_rank,
                selected=row.selected,
            )
            for row in chunk_rows
        ],
        answer=AnswerDetailOut.model_validate(answer_row) if answer_row else None,
    )


def _candidate_out(
    candidate: RerankedChunk, selected_uids: set[str]
) -> QueryCandidateOut:
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
        selected=evidence.chunk_uid in selected_uids,
    )


__all__ = ["embedding_provider", "get_query", "llm_provider", "rerank_provider", "router"]
