"""The generation provider this process uses, built once.

Mirrors `app.reranking.provider.get_rerank_provider` exactly: an
`@lru_cache`'d accessor so the client is built once and shared, with an
explicit reset hook for tests. The real `GeminiLLMProvider`, always —
nothing in the application ever selects the scripted fake through this
accessor. A caller that wants the fake constructs it directly, the same way
a test constructs `FakeRerankProvider` directly instead of going through
`get_rerank_provider()`.
"""

from functools import lru_cache

from app.config import get_settings
from app.providers.llm import LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    """The generation provider this process uses.

    The real Gemini adapter. Nothing in the application ever selects the
    fake here: only a test does, by constructing one and passing it in
    explicitly through the `llm_provider` FastAPI dependency override.
    """
    from app.providers.gemini_llm import GeminiLLMProvider

    return GeminiLLMProvider(get_settings().llm_model)


def reset_llm_provider() -> None:
    """Drop the cached provider. For tests."""
    get_llm_provider.cache_clear()


__all__ = ["get_llm_provider", "reset_llm_provider"]
