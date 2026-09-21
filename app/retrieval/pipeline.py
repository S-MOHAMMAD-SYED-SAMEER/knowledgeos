"""Wiring the stages together: two channels, fused, deduplicated, truncated.

This is the one function in `app/retrieval/` an API layer calls. It takes a
query that has already been normalized and already embedded — this package
never calls a provider, per the specification's layering rule — runs both
channels, fuses their rankings, attaches evidence once per chunk, and returns
the top 20. Nothing here reranks, generates an answer, or persists a query
record: those are milestones 6, 8 and (if ever) 5's own persistence decision,
none of them this one.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.retrieval import lexical, vector
from app.retrieval.dedupe import dedupe_evidence
from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import DEFAULT_RRF_K, fuse
from app.retrieval.vector import ChunkEvidence

# The specification's number: "take top 20 into reranking". Fixed for the
# same reason the per-channel limits are — a caller-tunable final count
# would make that sentence untrue on request.
FINAL_CANDIDATE_LIMIT = 20


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk in the final, truncated, ranked result."""

    evidence: ChunkEvidence
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float
    final_rank: int


@dataclass(frozen=True)
class RetrievalResult:
    """What one query produced, plus enough about each stage to explain it."""

    candidates: list[RetrievedChunk]
    vector_candidate_count: int
    lexical_candidate_count: int
    fused_candidate_count: int


def retrieve(
    session: Session,
    *,
    normalized_text: str,
    query_vector: list[float],
    filters: RetrievalFilters,
    rrf_k: int = DEFAULT_RRF_K,
) -> RetrievalResult:
    """Run both channels against one query and return its top 20 evidence.

    1. vector retrieval (top 50, cosine distance);
    2. lexical retrieval (top 50, `ts_rank_cd`);
    3. RRF fusion of the two rank lists;
    4. deduplication of evidence by `chunk_uid`;
    5. truncation to the top 20;
    6. `final_rank` assigned 1..N over what survives.

    Both channel queries run sequentially on `session`, under its ordinary
    `READ COMMITTED` behaviour — no elevated isolation level. A version
    committing between the two statements could in principle put one of its
    chunks in only one channel's result; RRF already handles a chunk found by
    only one channel, so nothing here needs to notice or guard against it.
    """
    vector_candidates = vector.search(
        session, query_vector=query_vector, filters=filters
    )
    lexical_candidates = lexical.search(
        session, normalized_text=normalized_text, filters=filters
    )

    fused = fuse(
        vector_ranking=[c.evidence.chunk_uid for c in vector_candidates],
        lexical_ranking=[c.evidence.chunk_uid for c in lexical_candidates],
        k=rrf_k,
    )
    evidence_by_uid = dedupe_evidence(vector_candidates, lexical_candidates)

    top = fused[:FINAL_CANDIDATE_LIMIT]
    candidates = [
        RetrievedChunk(
            evidence=evidence_by_uid[entry.chunk_uid],
            lexical_rank=entry.lexical_rank,
            vector_rank=entry.vector_rank,
            rrf_score=entry.rrf_score,
            final_rank=position + 1,
        )
        for position, entry in enumerate(top)
    ]

    return RetrievalResult(
        candidates=candidates,
        vector_candidate_count=len(vector_candidates),
        lexical_candidate_count=len(lexical_candidates),
        fused_candidate_count=len(fused),
    )


__all__ = ["FINAL_CANDIDATE_LIMIT", "RetrievalResult", "RetrievedChunk", "retrieve"]
