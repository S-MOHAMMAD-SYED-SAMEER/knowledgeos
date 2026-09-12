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
    }


@pytest.mark.parametrize(
    "deferred",
    [
        "sentence_transformers",
        "anthropic",
        "pgvector",
        "jinja2",
        "apscheduler",
        # Parsing libraries: milestone 3 reads document content, not this one.
        "pypdf",
        "pypdfium2",
        "docx",
        "magic",
        "filetype",
    ],
)
def test_a_later_milestones_library_is_not_installed(deferred: str) -> None:
    """Installing one early would make this milestone's install dishonest
    about what the code actually uses."""
    import importlib.util

    assert importlib.util.find_spec(deferred) is None


def test_the_schema_is_the_three_tables_built_so_far() -> None:
    """chunks, queries, answers and the rest arrive with the milestones that
    write them."""
    from app.db.base import Base
    import app.models  # noqa: F401

    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "ingestion_jobs",
    }
