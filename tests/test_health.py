"""Liveness, and the HTTP surface milestone 1 is allowed to have.

The second test is the contract: `/health` must answer while the database is
unreachable. It is not an optimisation — a liveness probe that failed on a
dependency would have an orchestrator restart a process that was working.
"""

import pytest
from fastapi.testclient import TestClient

from app import __version__


def test_health_answers(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["app"] == "KnowledgeOS"
    assert body["version"] == __version__
    assert set(body) == {"status", "app", "version", "environment"}


def test_health_answers_with_no_database_at_all(monkeypatch) -> None:
    """No engine is built, so there is nothing for a broken database to break.

    The URL points at a port nothing listens on. If `/health` touched the
    database in any way, this would fail.
    """
    from app.config import Settings, get_settings
    from app.db.session import reset_engine
    from app.main import create_app

    monkeypatch.setenv(
        "KNOWLEDGEOS_DATABASE_URL",
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/nowhere",
    )
    get_settings.cache_clear()
    reset_engine()
    try:
        response = TestClient(create_app(Settings(_env_file=None))).get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    finally:
        get_settings.cache_clear()
        reset_engine()


def test_health_makes_no_query() -> None:
    """Read from the code, because the assertion above cannot see inside it."""
    import ast
    import pathlib

    from app.api import health as module

    source = pathlib.Path(module.__file__).read_text()
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.db" not in imported
    assert "app.db.session" not in imported
    assert "app.models" not in imported


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_neither_probe_needs_authentication(client: TestClient, path: str) -> None:
    assert client.get(path).status_code in (200, 503)


def test_the_http_surface_is_exactly_the_two_probes(client: TestClient) -> None:
    """A scope guard. Documents, ingestion and query belong to later
    milestones, and this fails loudly if one of them arrives early."""
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {"/health", "/ready"}
