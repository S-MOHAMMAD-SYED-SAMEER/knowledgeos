"""The reranking provider this process uses, built once.

Mirrors `app.ingestion.runner.get_embedding_provider` exactly: an
`@lru_cache`'d accessor so the model — expensive to load — is built once and
shared, with an explicit reset hook for tests. Kept here, in
`app/reranking/`, rather than folded into the embedding provider's cache in
`app/ingestion/runner.py`: milestone 4 and 5 are locked, and sharing one
generic cache across two unrelated providers would mean touching locked code
for a tidiness milestone 6 does not need.

The real local cross-encoder, always — nothing in the application ever
selects the passthrough or the deterministic test double through this
accessor. A caller that wants either constructs it directly, the same way a
test constructs `FakeEmbeddingProvider` directly instead of going through
`get_embedding_provider()`.
"""

from functools import lru_cache

from app.providers.reranker import RerankProvider


@lru_cache
def get_rerank_provider() -> RerankProvider:
    """The reranking provider this process uses.

    The real local model. Nothing in the application ever selects the
    passthrough or the fake here: only a test, or a future evaluation run,
    does — by constructing one and passing it in explicitly.
    """
    from app.providers.cross_encoder import CrossEncoderRerankProvider

    return CrossEncoderRerankProvider()


def reset_rerank_provider() -> None:
    """Drop the cached provider. For tests."""
    get_rerank_provider.cache_clear()


__all__ = ["get_rerank_provider", "reset_rerank_provider"]
