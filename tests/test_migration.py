"""The migration: up, down, up again, and agreeing with the models."""

import logging
import pathlib

from alembic import command
from sqlalchemy import Engine, create_engine, inspect, text

VERSIONS = pathlib.Path(__file__).resolve().parent.parent / "alembic" / "versions"
TABLES = {
    "documents",
    "document_versions",
    "ingestion_jobs",
    "chunks",
    "eval_runs",
    "queries",
    "retrieved_chunks",
    "answers",
    "feedback",
}


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_there_are_exactly_seven_migrations() -> None:
    assert len(list(VERSIONS.glob("*.py"))) == 7


def test_upgrade_creates_every_table(migrated_engine: Engine, database_url) -> None:
    assert TABLES <= _tables(database_url)


def test_downgrade_removes_every_table(alembic_config, database_url) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")

    assert _tables(database_url) & TABLES == set()


def test_downgrade_leaves_the_shared_extension_alone(
    alembic_config, database_url
) -> None:
    """Dropping it would remove a PostgreSQL feature from the whole database,
    and cascade to anything using its type."""
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            installed = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    finally:
        engine.dispose()

    assert installed == 1


def test_a_clean_re_upgrade_works(alembic_config, database_url) -> None:
    """Up, down, up. The extension line has to be idempotent for this to pass."""
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    try:
        assert TABLES <= _tables(database_url)
    finally:
        command.downgrade(alembic_config, "base")


def test_the_migration_and_the_models_agree(
    migrated_engine: Engine, alembic_config
) -> None:
    """`alembic check`: the models describe the schema the migration builds.

    Without the naming convention on `Base.metadata` this fails immediately,
    because PostgreSQL and SQLAlchemy would name the same constraint
    differently.
    """
    command.check(alembic_config)


def test_the_partial_unique_index_is_in_the_database(
    migrated_engine: Engine
) -> None:
    """Built by the migration, so a schema made any other way would lack it."""
    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'uq_document_versions_one_active'"
            )
        ).scalar_one()

    # PostgreSQL renders the predicate with the casts it inferred, so the
    # assertion is on its parts rather than on the text as written.
    assert "UNIQUE" in definition
    assert "document_versions" in definition
    assert "(document_id)" in definition
    assert "WHERE" in definition
    assert "status" in definition and "'active'" in definition


# --- running a migration must not silence the application's logging ---------------

_PROBE_LOGGER = logging.getLogger("app.migration_logging_probe")  # created before any migration runs


def test_running_a_migration_in_process_does_not_disable_existing_loggers(migrated_engine) -> None:
    """`alembic/env.py` configures logging from `alembic.ini`. By default that disables every
    logger that already exists, which in a process that also runs the application (the test
    suite) means later warnings, such as "the model's reply was rejected", are never emitted."""
    assert _PROBE_LOGGER.disabled is False
