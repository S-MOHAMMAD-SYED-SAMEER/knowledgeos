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


def test_the_generation_package_exists_and_citations_stays_provider_free() -> None:
    """Milestone 8's own package — present, and `citations.py` calling no
    provider at all, per the specification's layering rule (§4), which
    names it explicitly alongside `app/retrieval/` and `app/chunking/`."""
    package = APP / "generation"
    assert package.exists()

    citations_source = (package / "citations.py").read_text()
    for name in _imports(citations_source):
        assert not name.startswith("app.providers"), name
        assert not name.startswith("sqlalchemy"), name


def test_the_evals_package_exists_and_is_bounded() -> None:
    """Milestone 7's own package — present, and calling no generation
    provider. A docstring may still *mention* generation while explaining
    that it belongs elsewhere, so this checks imports, not prose."""
    package = ROOT / "evals"
    assert package.exists()
    assert not (package / "answers").exists()

    for path in package.rglob("*.py"):
        for name in _imports(path.read_text()):
            assert name != "anthropic" and not name.startswith("anthropic."), (
                f"{path.name} imports {name}"
            )


def test_no_anthropic_import_anywhere_in_the_application() -> None:
    for module in APP.rglob("*.py"):
        for name in _imports(module.read_text()):
            assert name != "anthropic" and not name.startswith("anthropic."), module.name


def test_query_persistence_tables_now_exist() -> None:
    """`queries` and `retrieved_chunks` were deferred through milestones
    5-7 (a locked project decision); milestone 8 is what finally writes to
    them, and this inverts the guard that used to keep them out."""
    import app.models  # noqa: F401 - registers every model on Base.metadata
    from app.db.base import Base

    assert "queries" in Base.metadata.tables
    assert "retrieved_chunks" in Base.metadata.tables
    assert "answers" in Base.metadata.tables


def test_the_schema_is_the_eight_tables_built_so_far() -> None:
    """Milestone 8 adds `queries`, `retrieved_chunks` and `answers` — the
    specification's own three generation tables (§5), all deferred until
    now. `feedback` remains milestone 10's."""
    import app.models  # noqa: F401
    from app.db.base import Base

    assert set(Base.metadata.tables) == {
        "documents",
        "document_versions",
        "ingestion_jobs",
        "chunks",
        "eval_runs",
        "queries",
        "retrieved_chunks",
        "answers",
    }


def test_there_are_exactly_six_migrations() -> None:
    versions = (ROOT / "alembic" / "versions").glob("*.py")
    assert len(list(versions)) == 6


def test_get_queries_endpoint_now_exists() -> None:
    """Milestone 8 implements it — §11 lists it in the v1 API and nothing
    assigns it to a specific milestone; see the README's milestone 8
    section for the locked decision to build it now rather than leave it
    unreachable."""
    from app.main import create_app

    paths = create_app().openapi()["paths"]
    assert "/queries/{query_id}" in paths


def test_no_answer_or_feedback_endpoints_exist() -> None:
    """`POST /answers/{id}/feedback` remains milestone 10's — a route
    under `/answers` is still absent even though the `answers` table now
    exists."""
    from app.main import create_app

    paths = create_app().openapi()["paths"]
    assert not any("/answers" in path for path in paths)
    assert not any("/feedback" in path for path in paths)


def test_the_query_response_now_carries_generation_and_selection_fields() -> None:
    """`rerank_score` and `final_rank` (post-rerank) are milestone 6's own,
    legitimate fields, still present. Milestone 8 is what finally adds an
    answer, citations, and the `selected` decision the specification's
    `retrieved_chunks.selected` column always implied but milestone 6
    deliberately left unbuilt (no calibration data existed yet) — this
    inverts the guard that used to keep them out."""
    from app.api.schemas import QueryCandidateOut, QueryResponse

    assert "rerank_score" in QueryCandidateOut.model_fields
    assert "final_rank" in QueryCandidateOut.model_fields
    assert "fusion_rank" in QueryCandidateOut.model_fields
    assert "selected" in QueryCandidateOut.model_fields

    for expected in ("answer", "citations", "abstained", "citation_valid", "grounded"):
        assert expected in QueryResponse.model_fields

    # Still absent: nothing belonging to milestone 9/10 -- cost, latency,
    # confidence, or a feedback rating are not this milestone's to add.
    for forbidden in ("cost_usd", "latency_ms", "confidence", "rating"):
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


def test_milestones_five_through_eight_added_only_the_dependencies_they_needed() -> None:
    """Milestone 5's inspection concluded no new dependency was required,
    and milestone 6's cross-encoder came from a package milestone 4 already
    declared. Milestone 7 added PyYAML for the specification's own
    `*.yaml` question files. Milestone 8 adds `google-genai` — the
    generation provider is Google Gemini, not Anthropic, a documented
    deviation from the specification's stack wording; see the README's
    milestone 8 section."""
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
        "PyYAML",
        "google-genai",
    }


def test_no_evals_run_http_endpoint_exists() -> None:
    """The specification's own command is `python -m evals.run --suite
    retrieval` — an offline CLI, never something the running application
    exposes. No router anywhere may add a path under `/evals`."""
    from app.main import create_app

    paths = create_app().openapi()["paths"]
    assert not any(path.startswith("/evals") for path in paths)


def test_no_api_module_imports_the_evals_package() -> None:
    """The application itself never calls into `evals/` — evaluation reads
    the application's code, not the other way around."""
    for module in APP.rglob("*.py"):
        for name in _imports(module.read_text()):
            assert not name.startswith("evals"), f"{module} imports {name}"


def test_evals_fixtures_contain_no_python() -> None:
    """Fixture data (`evals/fixtures/knowledge_base/*.md`,
    `evals/fixtures/questions/*.yaml`) must stay data — a `.py` file in
    either directory would mean a provider or the database could be reached
    from what is supposed to be static content."""
    fixtures = ROOT / "evals" / "fixtures"
    assert fixtures.exists()
    assert not list(fixtures.rglob("*.py"))


def test_evals_package_has_no_second_database_configured() -> None:
    """D11, locked: the evaluation harness uses the one configured
    KnowledgeOS database, never a `knowledgeos_evals`-style second one."""
    for path in (ROOT / "evals").rglob("*.py"):
        source = path.read_text()
        assert "knowledgeos_evals" not in source
        assert "create_engine" not in source


def test_readme_m7_section_has_no_fabricated_metric_numbers() -> None:
    """D14, locked: the README documents what milestone 7 measures, never a
    result it did not actually produce. A real result, had one been
    recorded, would look like the four-decimal numbers
    `evals/retrieval/report.py` prints (`0.7000`) or a percentage — neither
    pattern may appear in this section, and the section must say plainly
    that official evaluation has not been run here."""
    import re

    text = (ROOT / "README.md").read_text()
    start = text.index("## What milestone 7 built")
    end = text.index("\n## ", start + 1)
    section = text[start:end]

    assert "has not been run" in section
    assert not re.search(r"\b\d\.\d{2,4}\b", section)
    assert "%" not in section
    for forbidden in ("gates passed", "evaluation passed", "all gates pass"):
        assert forbidden not in section.lower()


def test_readme_m8_section_has_no_fabricated_answer_metric_numbers() -> None:
    """The same discipline, applied to milestone 8: it implements
    generation, it does not run milestone 9's answer eval suite, and the
    README must not carry a grounded-answer rate, a citation precision/
    recall number, or an abstention accuracy figure it never measured."""
    import re

    text = (ROOT / "README.md").read_text()
    start = text.index("## What milestone 8 built")
    end = text.index("\n## ", start + 1)
    section = text[start:end]

    assert not re.search(r"\b\d\.\d{2,4}\b", section)
    assert "%" not in section
    for forbidden in ("gates passed", "evaluation passed", "all gates pass"):
        assert forbidden not in section.lower()


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
