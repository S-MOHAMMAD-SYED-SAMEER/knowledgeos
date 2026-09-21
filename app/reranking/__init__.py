"""Reranking: scoring milestone 5's top-20 candidates with a cross-encoder.

`app/providers/reranker.py` defines the interface and the identity types;
`app/providers/cross_encoder.py`, `app/providers/passthrough_reranker.py` and
`app/providers/fake_reranker.py` are the adapters. This package is the
orchestration that bridges `app.retrieval`'s output to that provider
interface and back — the specification's own words for this milestone's
scope are "full query pipeline wiring", and this is that wiring.

No generation, no citations, no persistence: this package returns evidence,
never an answer.
"""

from app.reranking.pipeline import RerankedChunk, rerank
from app.reranking.provider import get_rerank_provider, reset_rerank_provider

__all__ = [
    "RerankedChunk",
    "get_rerank_provider",
    "rerank",
    "reset_rerank_provider",
]
