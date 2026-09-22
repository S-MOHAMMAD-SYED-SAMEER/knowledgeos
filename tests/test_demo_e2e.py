"""P3 Step 5: the five flagship scenarios, end to end, through the real
HTTP route.

    HTTP POST /query
    -> app.api.query.run_query (unmodified)
    -> app.retrieval.pipeline.retrieve (fusion, unmodified)
    -> app.reranking.pipeline.rerank, scored by the real precomputed
       cross-encoder fixture replayed through DemoRerankProvider
    -> app.generation.generator.generate_answer, using the fixture
       answers replayed through DemoLLMProvider
    -> app.generation.citations.validate_citations (unmodified)
    -> app.generation.grounding.deterministic_grounding (unmodified)
    -> app.generation.persistence.persist_query (unmodified)
    -> the JSON QueryResponse

`demo.app.create_demo_app()` (P3 Step 4) is the only thing standing
between this and a live production query: the real
`app.main.create_app()`, with only the three provider dependencies
overridden. Every expected value below is read from this project's own
authoritative fixtures -- `demo/generate_embeddings.py::FLAGSHIP_QUERIES`,
`demo/fixtures/answers.yaml`, `demo/fixtures/reranker_scores.json` --
never invented here.
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.providers.bge import BgeEmbeddingProvider
from app.providers.cross_encoder import CrossEncoderRerankProvider
from app.providers.gemini_llm import GeminiLLMProvider
from app.storage import LocalStorage
from demo.app import create_demo_app
from demo.generate_embeddings import FLAGSHIP_QUERIES
from demo.llm import ANSWERS_FIXTURE_PATH
from demo.reranking import RERANKER_SCORES_FIXTURE_PATH
from demo.seed import seed_demo_corpus

FLAGSHIP_QUERY_TEXT = dict(FLAGSHIP_QUERIES)


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture(scope="module")
def answer_fixtures() -> dict[str, dict]:
    """The authoritative source for expected citations/abstention --
    `demo/fixtures/answers.yaml`, read once, never duplicated by hand."""
    raw = yaml.safe_load(ANSWERS_FIXTURE_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in raw}


@pytest.fixture(scope="module")
def reranker_fixture() -> dict:
    """The authoritative source for cs001's expected post-reranking
    ordering -- the real precomputed cross-encoder scores."""
    return json.loads(RERANKER_SCORES_FIXTURE_PATH.read_text(encoding="utf-8"))


@contextmanager
def _real_providers_never_called():
    """The same guard `tests/test_demo_app.py` already proved sufficient:
    the real BGE/cross-encoder/Gemini classes raise loudly if this
    request ever reaches them, so a passing test is proof the demo
    overrides -- not the real providers -- served it."""
    with (
        patch.object(BgeEmbeddingProvider, "embed", side_effect=AssertionError("real BGE called")),
        patch.object(
            CrossEncoderRerankProvider,
            "rerank",
            side_effect=AssertionError("real cross-encoder called"),
        ),
        patch.object(GeminiLLMProvider, "complete", side_effect=AssertionError("real Gemini called")),
    ):
        yield


def _demo_client(migrated_engine: Engine, storage: LocalStorage) -> TestClient:
    with Session(migrated_engine) as session:
        seed_demo_corpus(session, storage)
    return TestClient(create_demo_app())


def _post(client: TestClient, query_id: str):
    with _real_providers_never_called():
        return client.post("/query", json={"query": FLAGSHIP_QUERY_TEXT[query_id]})


# --- da001: retrieved correctly / answered from evidence -------------------


def test_da001_retrieved_correctly_and_answered_from_evidence(
    migrated_engine: Engine, storage: LocalStorage, answer_fixtures
) -> None:
    entry = answer_fixtures["da001"]
    golden = "5a04c330c14efaf8f4da83ebf6d6854f"
    client = _demo_client(migrated_engine, storage)

    response = _post(client, "da001")

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is False
    assert body["answer"]
    assert body["citations"] == entry["citations"] == [golden]
    assert body["citation_valid"] is True
    assert body["grounded"] is True

    candidate_uids = {c["chunk_uid"] for c in body["candidates"]}
    assert golden in candidate_uids
    # Every citation corresponds to evidence actually returned in this
    # response -- never a dangling reference.
    for uid in body["citations"]:
        assert uid in candidate_uids


# --- cs001: cited correctly, via the real reranking fixture -----------------


def test_cs001_cited_correctly_after_fixture_driven_reranking(
    migrated_engine: Engine, storage: LocalStorage, answer_fixtures, reranker_fixture
) -> None:
    entry = answer_fixtures["cs001"]
    golden = "f9cbee4e0939d52e95eebe93a063a85c"
    neighbor = "49446816bd2ba01147d70c4fcfeaa7fc"

    # The fixture is the authority that reranking moves the golden chunk
    # ahead of its same-document neighbor -- not an assumption invented
    # by this test. (Raw RRF fusion alone puts the neighbor first; this
    # is precisely what the real, precomputed cross-encoder scores
    # correct, per the P1/P3 triage.)
    fixture_entry = next(q for q in reranker_fixture["queries"] if q["id"] == "cs001")
    scores = {c["chunk_uid"]: c["rerank_score"] for c in fixture_entry["candidates"]}
    assert scores[golden] > scores[neighbor]

    client = _demo_client(migrated_engine, storage)
    response = _post(client, "cs001")

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is False
    assert body["citations"] == entry["citations"] == [golden]
    assert body["citation_valid"] is True
    assert body["grounded"] is True

    by_uid = {c["chunk_uid"]: c for c in body["candidates"]}
    assert golden in by_uid
    assert neighbor in by_uid
    # Not merely "both present": the golden chunk actually outranks its
    # same-document neighbor in the final, reranked response.
    assert by_uid[golden]["final_rank"] < by_uid[neighbor]["final_rank"]


# --- cv001: ranked correctly, current version only --------------------------


def test_cv001_ranked_correctly_current_version_only(
    migrated_engine: Engine, storage: LocalStorage, answer_fixtures
) -> None:
    entry = answer_fixtures["cv001"]
    active_v2 = "da1cdb5e0e87ad69728e9d2781239d13"
    superseded_v1 = "eef29ba299e4f0dd72c16e1741ac6928"

    client = _demo_client(migrated_engine, storage)
    response = _post(client, "cv001")

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is False
    assert body["citations"] == entry["citations"] == [active_v2]
    assert body["citation_valid"] is True
    assert body["grounded"] is True

    candidate_uids = {c["chunk_uid"] for c in body["candidates"]}
    assert active_v2 in candidate_uids
    assert superseded_v1 not in candidate_uids


# --- md001: answered from evidence across documents --------------------------


def test_md001_answered_from_evidence_across_documents(
    migrated_engine: Engine, storage: LocalStorage, answer_fixtures
) -> None:
    entry = answer_fixtures["md001"]
    incident_chunk = "f9cbee4e0939d52e95eebe93a063a85c"
    access_chunk = "da1cdb5e0e87ad69728e9d2781239d13"
    assert set(entry["citations"]) == {incident_chunk, access_chunk}  # the fixture's own set

    client = _demo_client(migrated_engine, storage)
    response = _post(client, "md001")

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is False
    assert set(body["citations"]) == set(entry["citations"])
    assert body["citation_valid"] is True
    assert body["grounded"] is True

    by_uid = {c["chunk_uid"]: c for c in body["candidates"]}
    assert incident_chunk in by_uid
    assert access_chunk in by_uid
    # The fixture specifies the citation set, not an ordering between the
    # two documents -- so this only asserts the signal PROJECT_PLAN.md
    # actually names: evidence drawn from two different documents.
    assert by_uid[incident_chunk]["document"]["title"] != by_uid[access_chunk]["document"]["title"]


# --- ie001: abstained when evidence is insufficient --------------------------


def test_ie001_abstains_when_evidence_is_insufficient(
    migrated_engine: Engine, storage: LocalStorage, answer_fixtures
) -> None:
    entry = answer_fixtures["ie001"]
    client = _demo_client(migrated_engine, storage)

    response = _post(client, "ie001")

    assert response.status_code == 200
    body = response.json()

    assert body["abstained"] is True
    assert body["citations"] == entry["citations"] == []
    assert body["citation_valid"] is True  # no citations, declared or inline: valid by construction
    assert body["grounded"] is True
    # No fabricated substantive answer: the frozen abstention contract's
    # own text, the fixture's own words, unchanged (DemoLLMProvider never
    # marks up an entry with no citations).
    assert body["answer"] == entry["answer_text"]
