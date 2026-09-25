"""The demo entrypoint: the real FastAPI application
(`app.main.create_app()`), with the three provider dependencies and the
existing mutation guard overridden to demo-only implementations.

Not a second query pipeline. `POST /query`, `POST /ui/query`, and every
other route this application serves are `app.main.create_app()`'s own,
completely unmodified -- this module only ever touches
`app.dependency_overrides`, the exact mechanism `tests/test_query_api.py`
and `tests/test_ui.py` already use to inject test doubles. Nothing here
reimplements retrieval, reranking, generation, or mutation-blocking; a
demo request runs through `app/api/query.py::run_query` (and, through it,
the whole frozen M5/M6/M8 pipeline) exactly as production does, against
demo-only providers instead of the real BGE/cross-encoder/Gemini ones,
and a demo mutation attempt runs through the same
`app/api/demo_guard.py::require_mutation_allowed` every other deployment
already depends on -- see `_deny_mutations` below for why it is reached
through an override rather than `Settings.demo_mode`.

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

from app.api.demo_guard import require_mutation_allowed
from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.config import Settings
from app.main import create_app
from demo.llm import DemoLLMProvider
from demo.providers import DemoEmbeddingProvider
from demo.reranking import DemoRerankProvider


def _deny_mutations() -> None:
    """Refuse every mutation, unconditionally -- by calling the *real*
    `require_mutation_allowed` directly, against a `Settings` instance
    built here and nowhere else, never the shared, cached `get_settings()`
    object `app.main`'s lifespan and every other dependency read.

    Not a second guard: this is the identical function `POST /documents`
    and the rest already depend on, given the identical answer it would
    give the integrated M1-M4 demo (`demo_mode=True`) -- but without ever
    setting the *application's own* `Settings.demo_mode`, which is what
    `app.main._lifespan` also reads to decide whether to seed the M1-M4
    corpus through the real BGE provider. This deployment must never do
    that -- see this module's own docstring -- so the flag stays `False`
    everywhere except inside this one, throwaway `Settings(...)` call.
    """
    require_mutation_allowed(Settings(demo_mode=True))


def create_demo_app() -> FastAPI:
    """The real application, with demo-only providers substituted at the
    same FastAPI dependency seam a test already uses.

    One instance of each provider, built once and named here explicitly
    -- deterministic construction, not a fresh instance re-loading its
    fixture on every request the way returning the bare class itself as
    the override would.

    **No document upload, re-indexing, or feedback.** A fourth override,
    alongside the three provider ones: `require_mutation_allowed` (the
    same dependency all five mutating routes already declare -- three in
    `app/api/documents.py`, one in `app/api/feedback.py`, and one in
    `app/ui/routes.py`) always refuses here, regardless of the real,
    shared `Settings.demo_mode` -- which this
    deployment leaves at its default `False` the entire time. That
    matters beyond mutation-blocking: `app.main._lifespan` also reads
    `Settings.demo_mode`, unconditionally, to decide whether to seed the
    M1-M4 corpus through the real, local BGE model -- a step this
    deployment must never reach, since it ships with no cached model
    weights at all. Leaving the flag off is what keeps that path closed;
    see `_deny_mutations` above for how the guard still engages without it.
    """
    app = create_app()

    demo_embeddings = DemoEmbeddingProvider()
    demo_reranker = DemoRerankProvider()
    demo_llm = DemoLLMProvider()

    app.dependency_overrides[embedding_provider] = lambda: demo_embeddings
    app.dependency_overrides[rerank_provider] = lambda: demo_reranker
    app.dependency_overrides[llm_provider] = lambda: demo_llm
    app.dependency_overrides[require_mutation_allowed] = _deny_mutations

    return app


app = create_demo_app()


__all__ = ["app", "create_demo_app"]
