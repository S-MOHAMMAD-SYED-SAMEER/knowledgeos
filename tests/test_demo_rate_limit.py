"""Rate limiting for the standalone demo's `POST /query`/`POST /ui/query`.

Uses the same `migrated_engine`/`storage`/`create_demo_app()` seam
`tests/test_demo_app.py` already establishes for this deployment -- no real
network, no real BGE/CrossEncoder/Gemini, and no real sleep: the limiter's
own clock and client-identity dependency are both injected, the same way
`demo.app.create_demo_app()` already injects the three provider
dependencies.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.rate_limit import (
    InMemoryRateLimiter,
    client_identity,
    get_rate_limiter,
    reset_rate_limiter,
)
from app.config import get_settings
from app.main import create_app
from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.storage import LocalStorage
from demo.app import create_demo_app
from demo.generate_embeddings import FLAGSHIP_QUERIES
from demo.llm import DemoLLMProvider
from demo.seed import seed_demo_corpus

FLAGSHIP_QUERY_TEXT = dict(FLAGSHIP_QUERIES)

# Comfortably above the default 10/minute, without running so many
# requests that this file becomes slow -- the point of each of these is
# "the limiter never engages", not "exhaustively prove no limiter ever
# would".
ABOVE_DEFAULT_LIMIT = 12


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture(autouse=True)
def _clean_shared_state() -> None:
    """Two kinds of process-wide state outlive any single test: the cached
    `Settings` (`app.config.get_settings`, an `lru_cache`) and the
    module-level rate limiter singleton `app.api.rate_limit` keeps. Both
    are cleared before and after every test here, the same way
    `tests/conftest.py::settings_env`/`migrated_engine` already clear the
    settings cache around themselves, so no test's environment variables
    or recorded request history leak into another's."""
    get_settings.cache_clear()
    reset_rate_limiter()
    yield
    get_settings.cache_clear()
    reset_rate_limiter()


class FakeClock:
    """A clock a test advances explicitly, so no test here ever sleeps."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _seeded_demo_client(migrated_engine: Engine, storage: LocalStorage) -> TestClient:
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)
    return TestClient(create_demo_app())


def _enable_rate_limit(monkeypatch: pytest.MonkeyPatch, *, per_minute: int = 10) -> None:
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_RATE_LIMIT_PER_MINUTE", str(per_minute))
    get_settings.cache_clear()


def _query_payload() -> dict:
    return {"query": FLAGSHIP_QUERY_TEXT["da001"]}


# --- 1: one request below the limit succeeds normally -----------------------


def test_a_single_request_below_the_limit_succeeds(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_rate_limit(monkeypatch, per_minute=10)
    client = _seeded_demo_client(migrated_engine, storage)

    response = client.post("/query", json=_query_payload())

    assert response.status_code == 200


# --- 2 & 3: exactly N succeed within the window, N+1 is refused -------------


def test_exactly_the_configured_limit_succeeds_then_the_next_is_refused(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = 3
    _enable_rate_limit(monkeypatch, per_minute=limit)
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)

    app = create_demo_app()
    limiter = InMemoryRateLimiter(clock=FakeClock())
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    client = TestClient(app)

    for _ in range(limit):
        response = client.post("/query", json=_query_payload())
        assert response.status_code == 200

    blocked = client.post("/query", json=_query_payload())

    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers
    assert int(blocked.headers["Retry-After"]) > 0


# --- 4: different client identities have independent limits -----------------


def test_different_clients_have_independent_limits(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_rate_limit(monkeypatch, per_minute=1)
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)

    # One limiter, shared by two demo apps whose only difference is the
    # `client_identity` override -- proving the limit is keyed on identity,
    # not merely on "one app instance sees one budget".
    shared_limiter = InMemoryRateLimiter(clock=FakeClock())

    app_a = create_demo_app()
    app_a.dependency_overrides[get_rate_limiter] = lambda: shared_limiter
    app_a.dependency_overrides[client_identity] = lambda: "visitor-a"
    client_a = TestClient(app_a)

    app_b = create_demo_app()
    app_b.dependency_overrides[get_rate_limiter] = lambda: shared_limiter
    app_b.dependency_overrides[client_identity] = lambda: "visitor-b"
    client_b = TestClient(app_b)

    assert client_a.post("/query", json=_query_payload()).status_code == 200
    # visitor-a is now at its limit of 1.
    assert client_a.post("/query", json=_query_payload()).status_code == 429
    # visitor-b, a distinct identity against the same shared limiter, has
    # its own, untouched budget.
    assert client_b.post("/query", json=_query_payload()).status_code == 200


# --- 5: advancing the injected clock frees a blocked client -----------------


def test_advancing_the_clock_allows_a_blocked_client_to_request_again(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_rate_limit(monkeypatch, per_minute=1)
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)

    clock = FakeClock()
    app = create_demo_app()
    limiter = InMemoryRateLimiter(clock=clock)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    client = TestClient(app)

    assert client.post("/query", json=_query_payload()).status_code == 200
    assert client.post("/query", json=_query_payload()).status_code == 429

    clock.advance(60.0)  # the full window -- never a real sleep

    assert client.post("/query", json=_query_payload()).status_code == 200


# --- 6: GET /health is unaffected by an exhausted query limit ---------------


def test_health_remains_available_after_exhausting_the_query_limit(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_rate_limit(monkeypatch, per_minute=1)
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)

    app = create_demo_app()
    limiter = InMemoryRateLimiter(clock=FakeClock())
    app.dependency_overrides[get_rate_limiter] = lambda: limiter
    client = TestClient(app)

    assert client.post("/query", json=_query_payload()).status_code == 200
    assert client.post("/query", json=_query_payload()).status_code == 429

    assert client.get("/health").status_code == 200


# --- 7 & 8: off by default, live/production behaviour is unaffected --------


def test_the_limiter_is_disabled_by_default(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `KNOWLEDGEOS_DEMO_RATE_LIMIT_ENABLED` set at all -- the default
    -- so well more than the default 10/minute still succeeds."""
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_RATE_LIMIT_ENABLED", raising=False)
    get_settings.cache_clear()
    client = _seeded_demo_client(migrated_engine, storage)

    for _ in range(ABOVE_DEFAULT_LIMIT):
        assert client.post("/query", json=_query_payload()).status_code == 200


def test_live_mode_query_is_unaffected_when_the_limiter_is_disabled(
    migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real production app (`app.main.create_app()`), not the demo
    app -- proving this feature, off by default, changes nothing about
    Live Mode's own `/query` dispatch. Real BGE/CrossEncoder are patched
    the same way `tests/test_query_api.py` already patches them, never
    invoked, so this needs no cached model and no network. The database
    is migrated but empty (no corpus seeded), so every call is a real,
    honest pre-LLM abstention -- 200, never a 429, `ABOVE_DEFAULT_LIMIT`
    times over."""
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("KNOWLEDGEOS_DEMO_MODE", raising=False)
    get_settings.cache_clear()

    client = TestClient(create_app())

    with (
        patch.object(BgeEmbeddingProvider, "embed", return_value=[[0.0] * 384]),
        patch.object(CrossEncoderRerankProvider, "rerank", return_value=[]),
    ):
        for _ in range(ABOVE_DEFAULT_LIMIT):
            response = client.post("/query", json=_query_payload())
            assert response.status_code == 200
            assert response.json()["abstained"] is True


# --- the mutation guard itself is `tests/test_demo_app.py`'s own scope,
# alongside the other three `demo.app.create_demo_app()` overrides -- see
# that file for the guard-override and lifespan-regression tests.


# --- 10: a successful response's shape is unchanged with the limiter on ----


def test_a_successful_demo_query_response_is_unchanged_with_the_limiter_enabled(
    migrated_engine: Engine, storage: LocalStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same assertions `tests/test_demo_app.py`'s own passing-query test
    makes for `da001`. `rate_limit_demo_query` either raises before
    `run_query` is ever called, or returns `None` and the route proceeds
    exactly as it would without it -- it never touches the response
    `run_query` builds."""
    _enable_rate_limit(monkeypatch, per_minute=10)
    client = _seeded_demo_client(migrated_engine, storage)

    response = client.post("/query", json=_query_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert "5a04c330c14efaf8f4da83ebf6d6854f" in body["citations"]
    assert body["model"] == DemoLLMProvider().model_name
