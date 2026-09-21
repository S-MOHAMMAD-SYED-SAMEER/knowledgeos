"""Wiring milestone 5's retrieval into the reranking stage.

Takes M5's `RetrievalResult.candidates` — already fetched, fused, deduplicated
and truncated to the top 20 — and a `RerankProvider`, and produces the final
reranked evidence: one score per candidate, sorted by score descending,
`chunk_uid` ascending as the deterministic tie-break, `final_rank` assigned
1..N over whatever survived.

This is the one place that bridges `app.retrieval`'s types and
`app.providers.reranker`'s types. Neither package imports the other; this
module imports both, which is what "full query pipeline wiring" — the
specification's own words for this milestone's scope — means. It is also why
this module, unlike anything under `app/retrieval/`, is allowed to call a
provider at all: the layering rule that forbids provider calls names
`app/retrieval/`, `app/chunking/` and `app/generation/citations.py`
specifically, and `app/reranking/` is not one of them.
"""

from dataclasses import dataclass

from app.providers.reranker import Candidate, RerankProvider
from app.retrieval.pipeline import RetrievedChunk
from app.retrieval.vector import ChunkEvidence


@dataclass(frozen=True)
class RerankedChunk:
    """One chunk in the final, reranked result.

    `fusion_rank` is what milestone 5 called `final_rank` — the chunk's
    position after RRF fusion, before reranking. `final_rank` here is the
    position after reranking, which is what "final" now means once a
    reranking stage exists: the rank a caller actually sees last.
    """

    evidence: ChunkEvidence
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float
    fusion_rank: int
    rerank_score: float
    final_rank: int


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    provider: RerankProvider,
) -> list[RerankedChunk]:
    """Score, sort and rank milestone 5's candidates.

    Empty input never reaches the provider: there is nothing to score, and
    calling a provider for nothing would mean loading a model — possibly the
    expensive real one — for no reason. `RetrievedChunk`'s own `final_rank`
    (renamed `fusion_rank` here) is read but never written to; nothing this
    function does can change milestone 5's retrieval result.
    """
    if not candidates:
        return []

    provider_candidates = [
        Candidate(id=candidate.evidence.chunk_uid, text=candidate.evidence.text)
        for candidate in candidates
    ]
    scored = provider.rerank(query, provider_candidates)
    score_by_id = {item.id: item.score for item in scored}

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -score_by_id[candidate.evidence.chunk_uid],
            candidate.evidence.chunk_uid,
        ),
    )

    return [
        RerankedChunk(
            evidence=candidate.evidence,
            lexical_rank=candidate.lexical_rank,
            vector_rank=candidate.vector_rank,
            rrf_score=candidate.rrf_score,
            fusion_rank=candidate.final_rank,
            rerank_score=score_by_id[candidate.evidence.chunk_uid],
            final_rank=position + 1,
        )
        for position, candidate in enumerate(ordered)
    ]


__all__ = ["RerankedChunk", "rerank"]
