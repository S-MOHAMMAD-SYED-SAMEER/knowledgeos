"""The demo entrypoint: the real FastAPI application
(`app.main.create_app()`), with the three provider dependencies
overridden to demo-only implementations.

Not a second query pipeline. `POST /query`, `POST /ui/query`, and every
other route this application serves are `app.main.create_app()`'s own,
completely unmodified -- this module only ever touches
`app.dependency_overrides`, the exact mechanism `tests/test_query_api.py`
and `tests/test_ui.py` already use to inject test doubles. Nothing here
reimplements retrieval, reranking, or generation; a demo request runs
through `app/api/query.py::run_query` (and, through it, the whole
frozen M5/M6/M8 pipeline) exactly as production does, against demo-only
providers instead of the real BGE/cross-encoder/Gemini ones.

Keyless, offline, no model download, no credential. `DemoEmbeddingProvider`,
`DemoRerankProvider`, and `DemoLLMProvider` (`demo/providers.py`,
`demo/reranking.py`, `demo/llm.py`) each replay real, precomputed model
output from a fixture committed to this repository -- none of the three
ever reaches HuggingFace, Gemini, or any other network endpoint.

Run with, e.g., `uvicorn demo.app:app` -- the database still has to be a
real, reachable PostgreSQL+pgvector instance, migrated to head and seeded
with `demo.seed.seed_demo_corpus`; only the three model/generation
credentials are what this module removes the need for.
"""

from fastapi import FastAPI

from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.main import create_app
from demo.llm import DemoLLMProvider
from demo.providers import DemoEmbeddingProvider
from demo.reranking import DemoRerankProvider


def create_demo_app() -> FastAPI:
    """The real application, with demo-only providers substituted at the
    same FastAPI dependency seam a test already uses.

    One instance of each provider, built once and named here explicitly
    -- deterministic construction, not a fresh instance re-loading its
    fixture on every request the way returning the bare class itself as
    the override would.
    """
    app = create_app()

    demo_embeddings = DemoEmbeddingProvider()
    demo_reranker = DemoRerankProvider()
    demo_llm = DemoLLMProvider()

    app.dependency_overrides[embedding_provider] = lambda: demo_embeddings
    app.dependency_overrides[rerank_provider] = lambda: demo_reranker
    app.dependency_overrides[llm_provider] = lambda: demo_llm

    return app


app = create_demo_app()


__all__ = ["app", "create_demo_app"]
