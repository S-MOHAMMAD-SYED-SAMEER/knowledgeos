"""Hybrid retrieval: lexical and vector search, fused by RRF.

No provider calls and no I/O beyond the database anywhere in this package —
the specification's layering rule, because these are the parts that get
evaluated and must be callable as functions over fixture data. A query is
already normalized text and an already-computed vector by the time it
reaches `pipeline.retrieve()`; embedding the query happens one layer up, in
the API.

Generation, citations, reranking and evaluation are later milestones. This
package returns evidence, never an answer.
"""

from app.retrieval.dedupe import dedupe_evidence
from app.retrieval.filters import RetrievalFilters, predicates, version_statuses
from app.retrieval.fusion import DEFAULT_RRF_K, FusedRank, fuse
from app.retrieval.pipeline import (
    FINAL_CANDIDATE_LIMIT,
    RetrievalResult,
    RetrievedChunk,
    retrieve,
)
from app.retrieval.vector import Candidate, ChunkEvidence

__all__ = [
    "DEFAULT_RRF_K",
    "FINAL_CANDIDATE_LIMIT",
    "Candidate",
    "ChunkEvidence",
    "FusedRank",
    "RetrievalFilters",
    "RetrievalResult",
    "RetrievedChunk",
    "dedupe_evidence",
    "fuse",
    "predicates",
    "retrieve",
    "version_statuses",
]
