"""`demo.app.create_demo_app()`: the FastAPI dependency-override seam
that wires the real application to demo-only providers.

Focused on the integration seam itself -- proving `create_app()` is
reused unmodified, all three dependencies are overridden, the production
dependency functions are untouched, and a demo request reaches the real
`/query` pipeline with no real BGE/CrossEncoder/Gemini provider ever
invoked. The five flagship scenarios' end-to-end correctness is a later
step's job, not this one's.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.main import create_app
from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.providers.gemini_llm import GeminiLLMProvider
from app.storage import LocalStorage
from demo.app import create_demo_app
from demo.generate_embeddings import FLAGSHIP_QUERIES
from demo.llm import DemoLLMProvider
from demo.providers import DemoEmbeddingProvider
from demo.reranking import DemoRerankProvider
from demo.seed import seed_demo_corpus

FLAGSHIP_QUERY_TEXT = dict(FLAGSHIP_QUERIES)


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


# --- create_app() is the underlying production factory --------------------


def test_create_demo_app_reuses_the_production_application_factory() -> None:
    """No second pipeline: the demo app is built by the identical
    `create_app()` production uses, so every route it serves -- health,
    documents, ingestion, /query, feedback, the UI -- is the real one."""
    production_app = create_app()
    demo_app = create_demo_app()

    # The same technique `tests/test_ui.py` already uses to prove a route
    # set: FastAPI's own generated OpenAPI schema, not route introspection
    # that varies across FastAPI's own internal route-wrapper types.
    assert demo_app.openapi()["paths"] == production_app.openapi()["paths"]
    assert demo_app.title == production_app.title
    assert demo_app.version == production_app.version


# --- all three dependencies are overridden --------------------------------


def test_all_three_provider_dependencies_are_overridden() -> None:
    demo_app = create_demo_app()

    assert embedding_provider in demo_app.dependency_overrides
    assert rerank_provider in demo_app.dependency_overrides
    assert llm_provider in demo_app.dependency_overrides

    assert isinstance(
        demo_app.dependency_overrides[embedding_provider](), DemoEmbeddingProvider
    )
    assert isinstance(
        demo_app.dependency_overrides[rerank_provider](), DemoRerankProvider
    )
    assert isinstance(demo_app.dependency_overrides[llm_provider](), DemoLLMProvider)


def test_the_same_provider_instance_is_reused_across_overrides() -> None:
    """Deterministic, explicit construction: one instance per provider,
    built once -- not a fresh instance (and a fresh fixture load) on
    every dependency resolution."""
    demo_app = create_demo_app()

    first = demo_app.dependency_overrides[embedding_provider]()
    second = demo_app.dependency_overrides[embedding_provider]()
    assert first is second


# --- the production dependency functions are unchanged ---------------------


def test_the_production_dependency_functions_still_resolve_the_real_providers() -> None:
    """Called directly -- bypassing any override -- the three functions
    `app/api/query.py` declares still resolve to the real providers,
    proving this step never touched that file. Every one of these three
    constructors is cheap and lazy (no model load, no network, no
    credential check) -- exactly the same real-provider construction
    `tests/test_reranking.py`/`tests/test_query_api.py` already exercise
    directly, without needing the model cached or a credential set."""
    assert isinstance(embedding_provider(), BgeEmbeddingProvider)
    assert isinstance(rerank_provider(), CrossEncoderRerankProvider)
    assert isinstance(llm_provider(), GeminiLLMProvider)


def test_creating_the_demo_app_does_not_mutate_app_api_query_module() -> None:
    """The override lives on the FastAPI *instance* this module builds,
    never on the shared dependency functions themselves -- so a second,
    independently-built production app is completely unaffected."""
    create_demo_app()
    production_app = create_app()

    assert embedding_provider not in production_app.dependency_overrides
    assert rerank_provider not in production_app.dependency_overrides
    assert llm_provider not in production_app.dependency_overrides


# --- a demo request reaches the real pipeline, with demo providers only ---


def _seed_and_client(migrated_engine: Engine, storage: LocalStorage) -> TestClient:
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)
    return TestClient(create_demo_app())


def test_a_demo_query_reaches_the_real_query_pipeline_and_answers(
    migrated_engine: Engine, storage: LocalStorage
) -> None:
    """`da001`, answered through `/query` -- the real, unmodified
    `run_query` -- with every provider construction the real
    BGE/CrossEncoder/Gemini classes patched to fail loudly if ever
    called, proving the demo providers alone served this request."""
    client = _seed_and_client(migrated_engine, storage)

    with (
        patch.object(BgeEmbeddingProvider, "embed", side_effect=AssertionError("real BGE called")),
        patch.object(
            CrossEncoderRerankProvider, "rerank", side_effect=AssertionError("real cross-encoder called")
        ),
        patch.object(GeminiLLMProvider, "complete", side_effect=AssertionError("real Gemini called")),
    ):
        response = client.post("/query", json={"query": FLAGSHIP_QUERY_TEXT["da001"]})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert "5a04c330c14efaf8f4da83ebf6d6854f" in body["citations"]
    assert body["model"] == DemoLLMProvider().model_name


def test_a_demo_query_can_also_abstain_through_the_real_pipeline(
    migrated_engine: Engine, storage: LocalStorage
) -> None:
    """`ie001` -- the post-LLM abstention path -- reaches the same real
    pipeline and the same demo providers, never the real ones."""
    client = _seed_and_client(migrated_engine, storage)

    with (
        patch.object(BgeEmbeddingProvider, "embed", side_effect=AssertionError("real BGE called")),
        patch.object(
            CrossEncoderRerankProvider, "rerank", side_effect=AssertionError("real cross-encoder called")
        ),
        patch.object(GeminiLLMProvider, "complete", side_effect=AssertionError("real Gemini called")),
    ):
        response = client.post("/query", json={"query": FLAGSHIP_QUERY_TEXT["ie001"]})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_no_credential_or_network_setting_is_required_to_build_the_demo_app(
    monkeypatch,
) -> None:
    """`GEMINI_API_KEY`/`GOOGLE_API_KEY`/`KNOWLEDGEOS_LLM_MODEL` all
    absent -- the same "runs with nothing set" guarantee
    `tests/test_config.py::test_no_credential_is_needed_to_configure_this_application`
    already proves for `Settings` itself -- and the demo app still
    builds and its overrides still resolve."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "KNOWLEDGEOS_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    demo_app = create_demo_app()

    assert isinstance(
        demo_app.dependency_overrides[embedding_provider](), DemoEmbeddingProvider
    )
    assert isinstance(demo_app.dependency_overrides[llm_provider](), DemoLLMProvider)
