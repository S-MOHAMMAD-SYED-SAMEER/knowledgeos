"""`demo.app.create_demo_app()`: the FastAPI dependency-override seam
that wires the real application to demo-only providers.

Focused on the integration seam itself -- proving `create_app()` is
reused unmodified, all three dependencies are overridden, the production
dependency functions are untouched, and a demo request reaches the real
`/query` pipeline with no real BGE/CrossEncoder/Gemini provider ever
invoked. The five flagship scenarios' end-to-end correctness is a later
step's job, not this one's.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.api.demo_guard import require_mutation_allowed
from app.api.query import embedding_provider, llm_provider, rerank_provider
from app.config import get_settings
from app.main import create_app
from app.models import Feedback
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


# --- the mutation guard is a fourth override, alongside the three providers


def test_the_mutation_guard_is_overridden() -> None:
    demo_app = create_demo_app()

    assert require_mutation_allowed in demo_app.dependency_overrides


def test_the_mutation_guard_override_always_refuses() -> None:
    """Calling the override directly -- exactly what FastAPI does when the
    dependency graph resolves it -- raises the same 403
    `require_mutation_allowed` itself raises for `demo_mode=True`, without
    this application's own, shared `Settings.demo_mode` ever being set
    (see the settings-isolation and lifespan-regression tests below)."""
    demo_app = create_demo_app()

    with pytest.raises(HTTPException) as excinfo:
        demo_app.dependency_overrides[require_mutation_allowed]()

    assert excinfo.value.status_code == 403


def test_production_app_does_not_inherit_the_mutation_guard_override() -> None:
    """The override lives on the FastAPI *instance* `create_demo_app()`
    builds, never on the shared `require_mutation_allowed` function
    itself -- mirrors `test_creating_the_demo_app_does_not_mutate_app_api_
    query_module` above, now for the fourth override."""
    create_demo_app()
    production_app = create_app()

    assert require_mutation_allowed not in production_app.dependency_overrides


def test_using_the_demo_app_does_not_leak_demo_mode_into_shared_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_deny_mutations` builds its own, throwaway `Settings(demo_mode=True)`
    -- this proves that instance is never confused with, or written back
    to, the real, shared, `lru_cache`d `get_settings()` object every other
    dependency in this application reads, including `app.main._lifespan`'s
    own `demo_mode` check."""
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_MODE", raising=False)
    get_settings.cache_clear()

    client = TestClient(create_demo_app())
    client.post("/documents", files={"file": ("note.txt", b"hello", "text/plain")})

    assert get_settings().demo_mode is False


# --- end-to-end: all five mutation routes refuse a standalone demo request -


def test_standalone_demo_mutation_routes_refuse_with_no_demo_mode_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `KNOWLEDGEOS_DEMO_MODE`, or any other special environment
    variable, is set anywhere in this test -- proving the guard holds by
    construction the moment `create_demo_app()` is called, not because an
    operator remembered to configure a flag correctly. All five routes
    `app/api/demo_guard.py`'s own module docstring names as the M4
    finding are checked, `POST /ui/answers/{id}/feedback` included: it
    now declares `require_mutation_allowed` as a route dependency the
    same way the other four already do (`app/ui/routes.py::
    ui_submit_feedback`), so `demo.app`'s override reaches it too. See
    `test_standalone_demo_refuses_ui_feedback_for_an_existing_answer`
    below for the stronger version of this proof, against a real,
    existing answer rather than a random id."""
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_MODE", raising=False)
    get_settings.cache_clear()

    client = TestClient(create_demo_app())
    random_id = uuid.uuid4()

    responses = {
        "POST /documents": client.post(
            "/documents", files={"file": ("note.txt", b"hello", "text/plain")}
        ),
        "POST /documents/{id}/versions": client.post(
            f"/documents/{random_id}/versions",
            files={"file": ("note.txt", b"hello", "text/plain")},
        ),
        "POST /documents/{id}/reindex": client.post(
            f"/documents/{random_id}/reindex"
        ),
        "POST /answers/{id}/feedback": client.post(
            f"/answers/{random_id}/feedback", json={"rating": "helpful"}
        ),
        "POST /ui/answers/{id}/feedback": client.post(
            f"/ui/answers/{random_id}/feedback", data={"rating": "helpful"}
        ),
    }

    for route, response in responses.items():
        assert response.status_code == 403, (
            f"{route} returned {response.status_code}, expected 403: "
            f"{response.text}"
        )


# --- CRITICAL: entering the real ASGI lifespan never reaches real BGE ------


def test_entering_the_demo_apps_lifespan_never_seeds_the_m1_m4_corpus(
    migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`with TestClient(...) as client:` -- unlike every other test in
    this file, which builds `TestClient(...)` directly -- actually runs
    the application's ASGI lifespan (`app.main._lifespan`), the same
    startup path a real `docker compose up` triggers. That path seeds the
    M1-M4 corpus through the real, local BGE model whenever
    `Settings.demo_mode` is true (`app/demo/seed.py::
    ensure_demo_corpus_seeded`) -- exactly the failure mode discovered
    during deployment inspection: this deployment's image carries no
    cached model weights, so reaching that path would crash startup.
    Patching `BgeEmbeddingProvider._load`/`embed` to raise turns "was the
    real model ever touched" into a hard failure rather than something
    that could pass by accident."""
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_MODE", raising=False)
    get_settings.cache_clear()

    with (
        patch.object(
            BgeEmbeddingProvider,
            "_load",
            side_effect=AssertionError("real BGE model loaded"),
        ),
        patch.object(
            BgeEmbeddingProvider, "embed", side_effect=AssertionError("real BGE called")
        ),
    ):
        with TestClient(create_demo_app()) as client:
            assert client.get("/health").status_code == 200


# --- the standalone demo's UI feedback route is guarded too ----------------


def test_standalone_demo_refuses_ui_feedback_for_an_existing_answer(
    migrated_engine: Engine, storage: LocalStorage
) -> None:
    """`POST /ui/answers/{id}/feedback` (`app/ui/routes.py`) now declares
    `require_mutation_allowed` as a route dependency, the same way the
    other four mutation routes already do, so the standalone demo's
    override (`demo/app.py::_deny_mutations`) reaches it too. This
    proves that against a real, existing `answer_id` obtained through
    the real query flow -- not an invented one a 404 could mask a
    still-broken guard behind -- and confirms no `Feedback` row is ever
    written for it."""
    client = _seed_and_client(migrated_engine, storage)

    answer_response = client.post(
        "/query", json={"query": FLAGSHIP_QUERY_TEXT["da001"]}
    )
    assert answer_response.status_code == 200
    answer_id = answer_response.json()["answer_id"]

    feedback_response = client.post(
        f"/ui/answers/{answer_id}/feedback", data={"rating": "helpful"}
    )

    assert feedback_response.status_code == 403

    with Session(migrated_engine) as session:
        rows = session.execute(
            select(Feedback).where(Feedback.answer_id == answer_id)
        ).scalars().all()
    assert rows == []
