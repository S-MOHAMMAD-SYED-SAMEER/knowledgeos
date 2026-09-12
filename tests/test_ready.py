"""Readiness: three checks, and what each says when it fails."""

import pathlib

from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.api import ready as module


def test_ready_is_200_on_a_migrated_database(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_reports_all_three_checks(
    client: TestClient, migrated_engine: Engine
) -> None:
    body = client.get("/ready").json()

    assert body["database"]["ok"] is True
    assert body["migrations"]["ok"] is True
    assert "at head" in body["migrations"]["detail"]
    assert body["extension"]["ok"] is True
    assert "vector" in body["extension"]["detail"]


def test_ready_is_503_when_the_database_is_unreachable(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        module,
        "_database",
        lambda: module.Check(ok=False, detail="unreachable (OperationalError)"),
    )
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not ready"
    assert response.json()["database"]["ok"] is False


def test_the_other_checks_are_skipped_when_the_database_is_down(
    client: TestClient, monkeypatch
) -> None:
    """There is nothing to read a revision or an extension list out of."""
    monkeypatch.setattr(
        module, "_database", lambda: module.Check(ok=False, detail="down")
    )
    body = client.get("/ready").json()

    assert body["migrations"]["ok"] is False
    assert "not checked" in body["migrations"]["detail"]
    assert body["extension"]["ok"] is False
    assert "not checked" in body["extension"]["detail"]


def test_an_unmigrated_database_is_not_ready(
    client: TestClient, database_url: str, alembic_config, monkeypatch
) -> None:
    """Measured against a real empty database, not a stubbed check."""
    from alembic import command

    from app.config import get_settings
    from app.db.session import reset_engine

    monkeypatch.setenv("KNOWLEDGEOS_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()
    command.downgrade(alembic_config, "base")
    try:
        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["migrations"]["ok"] is False
        assert "expects" in response.json()["migrations"]["detail"]
    finally:
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


def test_a_missing_extension_is_not_ready(
    client: TestClient, migrated_engine: Engine, monkeypatch
) -> None:
    """The foundation milestone 1 exists to establish."""
    monkeypatch.setattr(
        module,
        "_extension",
        lambda: module.Check(ok=False, detail="vector is not installed"),
    )
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["extension"]["ok"] is False


def test_readiness_never_names_the_connection_string() -> None:
    """It carries a password."""
    source = pathlib.Path(module.__file__).read_text()

    assert "database_url" not in source


def test_readiness_calls_no_provider() -> None:
    """There are none yet, and there must be none here when there are."""
    import ast

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not any(name.startswith("app.providers") for name in imported)
