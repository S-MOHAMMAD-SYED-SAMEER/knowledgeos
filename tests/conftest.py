"""Fixtures.

Two choices worth stating.

The schema under test is built by **running the real migration**, not by
`Base.metadata.create_all`. The one-active-version rule is a partial unique
index that lives in the migration; a schema built any other way would not have
it, and the tests that matter most would pass against a database that could
not enforce anything.

Tests needing PostgreSQL **skip** rather than fail when no server answers, so
`pytest` is still meaningful on a machine without one.
"""

import os
from collections.abc import Iterator

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import reset_engine
from app.main import create_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Never the development database. The suite migrates this one down to nothing
# on every run, so it must not be anything anybody wants to keep.
TEST_DATABASE_URL = os.environ.get(
    "KNOWLEDGEOS_TEST_DATABASE_URL",
    "postgresql+psycopg://knowledgeos:knowledgeos@localhost:5432/knowledgeos_test",
)


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A clean settings and engine cache, restored afterwards."""
    monkeypatch.setenv("KNOWLEDGEOS_ENVIRONMENT", "test")
    get_settings.cache_clear()
    reset_engine()
    try:
        yield
    finally:
        get_settings.cache_clear()
        reset_engine()


@pytest.fixture
def client(settings_env: None) -> Iterator[TestClient]:
    yield TestClient(create_app())


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test database URL, skipping the test if no server answers."""
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL at {TEST_DATABASE_URL}: {exc}")
    finally:
        engine.dispose()
    return TEST_DATABASE_URL


@pytest.fixture
def alembic_config(database_url: str) -> AlembicConfig:
    """Alembic pointed explicitly at the test database.

    The URL is set here rather than left to `app.config`, because a migration
    fixture that inherited the ambient configuration would migrate — and
    downgrade — whichever database the developer happened to be pointing at.
    `alembic/env.py` honours a URL a caller has already set for this reason.
    """
    if not database_url.rstrip("/").endswith("_test"):
        raise RuntimeError(
            f"refusing to migrate {database_url!r}: the suite downgrades to "
            "base, so it runs only against a database whose name ends in "
            "'_test'."
        )

    config = AlembicConfig(os.path.join(PROJECT_ROOT, "alembic.ini"))
    config.set_main_option("script_location", os.path.join(PROJECT_ROOT, "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def migrated_engine(
    database_url: str,
    alembic_config: AlembicConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Engine]:
    """A database at head, torn back down to empty afterwards."""
    monkeypatch.setenv("KNOWLEDGEOS_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine) as db_session:
        yield db_session


# --- document fixtures ------------------------------------------------------


@pytest.fixture
def document(session: Session):
    """One document, saved."""
    from app.models import Document

    row = Document(title="Engineering Access SOP", department="Engineering")
    session.add(row)
    session.commit()
    return row


def version(document_id, number: int = 1, status: str = "draft", **overrides):
    """A version of a document, with the columns an upload would set.

    Milestone 1 has no upload, so these are the values milestone 2 will be
    responsible for producing.
    """
    from app.models import DocumentVersion

    fields = {
        "document_id": document_id,
        "version_number": number,
        "status": status,
        "original_filename": f"access-sop-v{number}.pdf",
        "storage_path": f"documents/{document_id}/{number}/source.pdf",
    }
    fields.update(overrides)
    return DocumentVersion(**fields)
