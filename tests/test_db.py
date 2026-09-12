"""The engine, the session, and the extension underneath them."""

import os
import pathlib

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.db.session import get_engine, get_sessionmaker, reset_engine
from app.models import Document


def test_importing_the_application_opens_no_connection() -> None:
    """Otherwise collecting this suite on a machine with no PostgreSQL fails
    before a single test runs.

    Run in a fresh interpreter against a URL nothing listens on: if any import
    on the way to `app.main` built an engine and connected, this would raise
    rather than print.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import app.main; print(app.main.app.title)"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": os.environ["PATH"],
            "KNOWLEDGEOS_DATABASE_URL": (
                "postgresql+psycopg://nobody:nothing@127.0.0.1:1/nowhere"
            ),
        },
        cwd=pathlib.Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "KnowledgeOS"


def test_the_engine_is_built_once(settings_env) -> None:
    assert get_engine() is get_engine()
    assert get_sessionmaker() is get_sessionmaker()


def test_resetting_drops_the_cached_engine(settings_env) -> None:
    get_engine()
    reset_engine()

    assert get_engine.cache_info().currsize == 0


def test_the_engine_speaks_psycopg(settings_env) -> None:
    """The specification's `postgresql+psycopg://`, not psycopg2."""
    assert get_engine().dialect.driver == "psycopg"


def test_a_session_round_trips_a_row(migrated_engine: Engine) -> None:
    with Session(migrated_engine) as session:
        session.add(Document(title="Handbook"))
        session.commit()

    with Session(migrated_engine) as session:
        assert session.query(Document).count() == 1


def test_the_vector_extension_is_installed(migrated_engine: Engine) -> None:
    """The foundation the retrieval half of this project is built on. No
    column uses it yet; milestone 1 exists to prove it is really there."""
    with migrated_engine.connect() as connection:
        version = connection.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()

    assert version is not None


def test_a_vector_value_is_usable(migrated_engine: Engine) -> None:
    """Installed and working, not merely listed."""
    with migrated_engine.connect() as connection:
        distance = connection.execute(
            text("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
        ).scalar_one()

    assert distance == 1.0
