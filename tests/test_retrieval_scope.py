"""Scope guards for the retrieval and reranking pipeline (milestones 5-6).

Reranking (milestone 6) has arrived and is asserted present, correctly
wired, and correctly bounded. Generation, citations, abstention, evaluation,
query persistence and an approximate vector index all still belong to later
milestones (or, for `queries`/`retrieved_chunks`, are a locked project
decision to keep deferring). These tests fail loudly if one of them arrives
early.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = ROOT / "app"


def _imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_reranking_package_exists_and_is_bounded() -> None:
    """Milestone 6's own package — present, and calling no generation
    provider. A docstring may still *mention* "citations" or "abstention"
    while explaining that those belong elsewhere (this package's own
    `__init__.py` does), so this checks imports, not prose."""
    package = APP / "reranking"
    assert package.exists()

    for path in package.glob("*.py"):
        for name in _imports(path.read_text()):
            assert name != "anthropic" and not name.startswith("anthropic."), (
                f"{path.name} imports {name}"
            )


def test_retrieval_package_still_imports_no_reranking_provider() -> None:
    """Milestone 6 must not have put a provider call into `app/retrieval/`
    to wire reranking in — the layering rule names `app/retrieval/`
    specifically, and `app/reranking/` exists precisely so retrieval never
    has to import it."""
    forbidden_prefixes = ("app.reranking", "app.providers.reranker",
                           "app.providers.cross_encoder",
                           "app.providers.passthrough_reranker",
                           "app.providers.fake_reranker")
    for module in (APP / "retrieval").glob("*.py"):
        for name in _imports(module.read_text()):
            assert not name.startswith(forbidden_prefixes), (
                f"{module.name} imports {name}"
            )


def test_no_generation_package_exists() -> None:
    assert not (APP / "generation").exists()


def test_no_evals_package_exists() -> None:
    assert not (ROOT / "evals").exists()


def test_no_anthropic_import_anywhere_in_the_application() -> None:
    for module in APP.rglob("*.py"):
        for name in _imports(module.read_text()):
            assert name != "anthropic" and not name.startswith("anthropic."), module.name


def test_no_query_persistence_tables() -> None:
    """`queries` and `retrieved_chunks` are deferred — a locked project
    decision, not silently invented because the specification's v1 API
    section mentions `GET /queries/{id}`."""
    import app.models  # noqa: F401 - registers every model on Base.metadata
    from app.db.base import Base

    assert "queries" not in Base.metadata.tables
    assert "retrieved_chunks" not in Base.metadata.tables


def test_the_schema_is_still_the_four_tables_built_so_far() -> None:
    """Milestone 5 needs no migration: everything it reads already exists."""
    import app.models  # noqa: F401
    from app.db.base import Base

    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "ingestion_jobs",
        "chunks",
    }


def test_there_are_still_exactly_four_migrations() -> None:
    versions = (ROOT / "alembic" / "versions").glob("*.py")
    assert len(list(versions)) == 4


def test_no_get_queries_endpoint_exists() -> None:
    from app.main import create_app

    paths = create_app().openapi()["paths"]
    assert not any(path.startswith("/queries") for path in paths)


def test_no_answer_or_feedback_endpoints_exist() -> None:
    from app.main import create_app

    paths = create_app().openapi()["paths"]
    assert not any("/answers" in path for path in paths)
    assert not any("/feedback" in path for path in paths)


def test_the_query_response_has_no_generation_or_selection_fields() -> None:
    """`rerank_score` and `final_rank` (post-rerank) are milestone 6's own,
    legitimate fields. What must still be absent is anything belonging to
    milestone 8: an answer, citations, or an invented `selected` boolean —
    the specification names "evidence selection" as a concept but gives no
    rule for it, and milestone 6 does not invent one (see D6)."""
    from app.api.schemas import QueryCandidateOut, QueryResponse

    assert "rerank_score" in QueryCandidateOut.model_fields
    assert "final_rank" in QueryCandidateOut.model_fields
    assert "fusion_rank" in QueryCandidateOut.model_fields

    for forbidden in ("answer", "citations", "selected", "sufficient_evidence"):
        assert forbidden not in QueryCandidateOut.model_fields
        assert forbidden not in QueryResponse.model_fields


def test_no_hnsw_or_ivfflat_index_is_created_by_any_migration() -> None:
    """A docstring may *mention* HNSW while explaining why it is deferred
    (migration 0004 does); no migration may actually create one."""
    for path in (ROOT / "alembic" / "versions").glob("*.py"):
        source = path.read_text().lower()
        assert "using hnsw" not in source
        assert "using ivfflat" not in source
        assert "postgresql_using=\"hnsw\"" not in source
        assert "postgresql_using=\"ivfflat\"" not in source


def test_no_hnsw_or_ivfflat_mentioned_at_all_in_retrieval_code() -> None:
    """Unlike the migration docstring, `app/retrieval/` has no reason to
    mention either — it only ever runs the exact search the specification
    requires at v1 corpus size."""
    for path in (APP / "retrieval").glob("*.py"):
        source = path.read_text().lower()
        assert "hnsw" not in source
        assert "ivfflat" not in source


def test_no_redis_or_celery_anywhere() -> None:
    for module in APP.rglob("*.py"):
        for name in _imports(module.read_text()):
            assert not name.startswith("redis")
            assert not name.startswith("celery")


def test_no_second_datastore_is_declared() -> None:
    """The specification's whole data layer is PostgreSQL + pgvector."""
    import tomllib

    declared = {
        __import__("re").split(r"[><=\[]", item)[0].strip().lower()
        for item in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
    }
    for forbidden in ("redis", "pymongo", "elasticsearch", "pinecone", "weaviate", "qdrant"):
        assert forbidden not in declared


def test_milestone_five_added_no_dependency() -> None:
    """The inspection concluded no new dependency was required."""
    import tomllib

    declared = {
        __import__("re").split(r"[><=\[]", item)[0].strip()
        for item in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
    }

    assert declared == {
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-settings",
        "sqlalchemy",
        "alembic",
        "psycopg",
        "python-multipart",
        "pypdf",
        "APScheduler",
        "sentence-transformers",
        "pgvector",
    }


def test_bm25_never_describes_postgres_full_text_search() -> None:
    """The specification's naming rule. A disclaiming mention ("not BM25",
    "never ... BM25") is fine; a bare claim that Postgres FTS *is* BM25 is
    not — checked over a window of nearby text rather than one line, since
    prose wraps a "not" onto the line before "BM25" itself."""
    negations = ("not", "never", "isn't", "n't")
    window = 80
    targets = list((APP / "retrieval").glob("*.py")) + [ROOT / "README.md"]
    for path in targets:
        text = " ".join(path.read_text().split())
        lowered = text.lower()
        start = 0
        while (index := lowered.find("bm25", start)) != -1:
            nearby = lowered[max(0, index - window) : index]
            assert any(word in nearby for word in negations), (
                f"{path.name}: ...{text[max(0, index - window):index + 20]}..."
            )
            start = index + 4
