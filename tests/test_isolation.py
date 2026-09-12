"""KnowledgeOS stands on its own.

This repository holds three unrelated applications. They share a git history
and nothing else: no package, no virtualenv, no database, no configuration
prefix. These tests are what keeps an import from one of them drifting into
this one — which would be invisible until the day somebody tried to deploy
this project by itself.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app"
SIBLINGS = ("docintel", "voicedesk")


def _modules() -> list[pathlib.Path]:
    return sorted(APP.rglob("*.py"))


def _imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("sibling", SIBLINGS)
def test_no_module_imports_a_sibling_project(sibling: str) -> None:
    for module in _modules():
        for name in _imports(module.read_text()):
            assert name.split(".")[0] != sibling, f"{module.name} imports {sibling}"


@pytest.mark.parametrize("sibling", SIBLINGS)
def test_no_module_mentions_a_sibling_project(sibling: str) -> None:
    """Not in a path, a default, a comment or a docstring either."""
    for module in _modules():
        assert sibling not in module.read_text().lower(), module.name


def test_the_configuration_prefix_is_our_own() -> None:
    """Sharing an environment prefix would mean sharing a database by
    accident, which is the one mistake that is hard to notice."""
    from app.config import Settings

    assert Settings.model_config["env_prefix"] == "KNOWLEDGEOS_"


def test_the_default_database_is_our_own() -> None:
    from app.config import DEFAULT_DATABASE_URL

    assert DEFAULT_DATABASE_URL.endswith("/knowledgeos")
    for sibling in SIBLINGS:
        assert sibling not in DEFAULT_DATABASE_URL


def test_this_project_owns_its_own_dependencies() -> None:
    """Its own pyproject, its own virtualenv, nothing borrowed."""
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "alembic.ini").is_file()
    assert (ROOT / ".venv").is_dir()


def test_milestone_one_installs_nothing_from_a_later_milestone() -> None:
    """Each of these arrives with the milestone that first imports it."""
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    names = {__import__("re").split(r"[><=\[]", item)[0].strip() for item in declared}

    assert names == {
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "sqlalchemy",
        "alembic",
        "psycopg",
        # Milestone 2: FastAPI refuses to define an upload route without it.
        "python-multipart",
        # Milestone 3: PDF text and page count, and the runner that polls
        # for queued ingestion jobs.
        "pypdf",
        "APScheduler",
        # Milestone 4: the local embedding model, and the SQLAlchemy
        # integration for PostgreSQL's vector type.
        "sentence-transformers",
        "pgvector",
    }


@pytest.mark.parametrize(
    "deferred",
    [
        "anthropic",
        # Parsers milestone 3 deliberately did not need: DOCX is read with
        # the standard library, and a signature check needs no library.
        "pypdfium2",
        "docx",
        "magic",
        "filetype",
        # Brokers the specification forbids outright.
        "redis",
        "celery",
    ],
)
def test_a_later_milestones_library_is_not_installed(deferred: str) -> None:
    """Installing one early would make this milestone's install dishonest
    about what the code actually uses."""
    import importlib.util

    assert importlib.util.find_spec(deferred) is None


def test_no_later_milestones_library_is_declared() -> None:
    """Jinja2 needs the weaker check: it is importable from milestone 4
    onwards because **torch pulls it in**, not because this project asked
    for it. What matters is that it is not a dependency of ours — the user
    interface it will eventually render is milestone 10's.
    """
    import re
    import tomllib

    declared = {
        re.split(r"[><=\[]", item)[0].strip().lower()
        for item in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
    }

    assert "jinja2" not in declared
    assert "anthropic" not in declared


def test_the_schema_is_the_four_tables_built_so_far() -> None:
    """queries, answers and the rest arrive with the milestones that write
    them."""
    from app.db.base import Base
    import app.models  # noqa: F401

    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "ingestion_jobs",
        "chunks",
    }
