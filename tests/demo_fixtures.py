"""Shared demo-mode test fixtures: a real, narrow, real-chunk_uid corpus
seed and a demo-mode `TestClient`, used by both `tests/test_demo_query_api.py`
(the JSON API, M2) and `tests/test_demo_ui.py` (the visitor UI, M3) so
neither duplicates the other's seeding logic.

See `tests/test_demo_query_api.py`'s own module docstring for why each
scenario is seeded from only the 1-2 real documents it actually cites,
rather than the full 10-document corpus.
"""

import io
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.query import embedding_provider, rerank_provider
from app.config import Settings, get_settings
from app.ingestion import service
from app.ingestion.pipeline import run_indexing, run_job
from app.main import create_app
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.passthrough_reranker import PassthroughRerankProvider
from app.storage import LocalStorage

MANIFEST = yaml.safe_load(
    Path("evals/fixtures/knowledge_base/manifest.yaml").read_text()
)
FIXTURE_DIR = Path("evals/fixtures/knowledge_base")

# The document(s) each non-ie001 scenario's real citations live in — the
# minimal real-fixture seed each scenario's test needs. Kept in one place so
# every scenario id in `DEMO_SCENARIOS` is covered, and a new scenario added
# there without an entry here fails loudly rather than silently skipping.
SCENARIO_DOCUMENTS: dict[str, list[str]] = {
    "da001": ["remote-work"],
    "md001": ["incident-response", "access-sop"],
    "cv001": ["access-sop"],
    "am001": ["access-sop", "vendor-access"],
    "mf001": ["access-sop"],
    "cs001": ["incident-response"],
    "adv003": ["access-sop"],
}


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def seed_demo_document(
    session: Session, storage: LocalStorage, embeddings: FakeEmbeddingProvider, key: str
) -> None:
    """Seed one manifest document's *current* (last-listed) version under
    its real `document_id`/`version_id`, through the real ingestion
    pipeline — so the chunk_uids it produces are the same real chunk_uids
    the curated demo scenarios cite."""
    entry = next(d for d in MANIFEST["documents"] if d["key"] == key)
    document_id = uuid.UUID(str(entry["document_id"]))
    version_entry = entry["versions"][-1]
    version_id = uuid.UUID(str(version_entry["version_id"]))
    text = (FIXTURE_DIR / version_entry["file"]).read_text()

    settings = Settings(
        _env_file=None,
        chunk_size_tokens=MANIFEST["chunk_size_tokens"],
        chunk_overlap_tokens=MANIFEST["chunk_overlap_tokens"],
    )
    storage_key = service.storage_key(document_id, version_id, ".md")
    storage.write(storage_key, io.BytesIO(text.encode()))
    result = service.create_document_with_version(
        session,
        document_id=document_id,
        title=entry["title"],
        department=entry.get("department"),
        category=entry.get("category"),
        tags=entry.get("tags") or [],
        original_filename=version_entry["file"],
        storage_path=storage_key,
        effective_date=None,
        version_id=version_id,
    )
    session.commit()
    run_job(session, result.job, storage, settings)
    session.commit()
    run_indexing(session, result.job, embeddings)
    session.commit()


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch, tmp_path_factory) -> Iterator[TestClient]:
    """`/query` (and, as of M3, `/demo`) with `Settings.demo_mode=True`,
    through the real `llm_provider` dependency function — never an
    override of it. Built the same way `tests/conftest.py::client` is
    (plain `TestClient(...)`, never entered as a context manager), so the
    application `_lifespan` (and its real-embedding-provider demo-corpus
    seeding) never runs here; each test seeds its own narrow, real-pipeline
    corpus instead."""
    monkeypatch.setenv("KNOWLEDGEOS_ENVIRONMENT", "test")
    monkeypatch.setenv("KNOWLEDGEOS_DEMO_MODE", "true")
    monkeypatch.setenv(
        "KNOWLEDGEOS_STORAGE_ROOT", str(tmp_path_factory.mktemp("storage"))
    )
    get_settings.cache_clear()
    try:
        yield TestClient(create_app())
    finally:
        get_settings.cache_clear()


@pytest.fixture
def fake_demo_client(
    demo_client: TestClient,
    embeddings: FakeEmbeddingProvider,
) -> Iterator[TestClient]:
    demo_client.app.dependency_overrides[embedding_provider] = lambda: embeddings
    demo_client.app.dependency_overrides[rerank_provider] = lambda: PassthroughRerankProvider()
    try:
        yield demo_client
    finally:
        demo_client.app.dependency_overrides.clear()


__all__ = [
    "FIXTURE_DIR",
    "MANIFEST",
    "SCENARIO_DOCUMENTS",
    "demo_client",
    "embeddings",
    "fake_demo_client",
    "seed_demo_document",
    "storage",
]
