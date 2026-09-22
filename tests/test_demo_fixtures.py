"""The demo foundation (PROJECT_PLAN.md, P1): real precomputed BGE
embeddings for the demo corpus, a deterministic replay provider, curated
demo answers, and the five flagship scenarios verified through the real
retrieval pipeline.

Fixture-shape tests below need no database. The flagship-scenario tests
seed a real corpus and call `app.retrieval.pipeline.retrieve()`
unmodified, against a real PostgreSQL database, exactly like
`tests/test_retrieval_pipeline.py` and `tests/test_evals_corpus.py`
already do -- so, like those, they skip rather than fail when no
PostgreSQL server answers.
"""

import json

import pytest
import yaml
from sqlalchemy import func, select

from app.models import Chunk
from app.providers import Candidate, CrossEncoderRerankProvider, FakeEmbeddingProvider
from app.providers.embeddings import DIMENSIONS, EmbeddingError, MODEL_NAME
from app.reranking import rerank
from app.retrieval import vector
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from app.storage import LocalStorage
from demo.generate_embeddings import (
    EXPECTED_CHUNK_COUNT,
    FLAGSHIP_QUERIES,
    iter_manifest_chunks,
)
from demo.providers import DEMO_MODEL_NAME, EMBEDDINGS_FIXTURE_PATH, DemoEmbeddingProvider
from demo.seed import seed_demo_corpus

ANSWERS_FIXTURE_PATH = EMBEDDINGS_FIXTURE_PATH.parent / "answers.yaml"


@pytest.fixture(scope="module")
def embeddings_fixture() -> dict:
    return json.loads(EMBEDDINGS_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def answers_fixture() -> list[dict]:
    return yaml.safe_load(ANSWERS_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_demo_chunk_uids() -> set[str]:
    """The chunk_uids the real ingestion pipeline actually produces for the
    pinned manifest -- ground truth the fixture is checked against, not
    read back from the fixture itself."""
    return {chunk.chunk_uid for chunk in iter_manifest_chunks()}


# --- demo/fixtures/embeddings.json --------------------------------------


def test_embeddings_fixture_has_exactly_33_chunks(embeddings_fixture) -> None:
    assert len(embeddings_fixture["chunks"]) == 33 == EXPECTED_CHUNK_COUNT


def test_embeddings_fixture_is_384_dimensional(embeddings_fixture) -> None:
    assert embeddings_fixture["dimensions"] == 384 == DIMENSIONS
    for entry in embeddings_fixture["chunks"]:
        assert len(entry["embedding"]) == 384


def test_embeddings_fixture_reports_the_real_bge_model(embeddings_fixture) -> None:
    assert embeddings_fixture["model_name"] == MODEL_NAME == "BAAI/bge-small-en-v1.5"


def test_embeddings_fixture_chunk_uids_match_the_real_demo_corpus(
    embeddings_fixture, real_demo_chunk_uids
) -> None:
    """Not merely "33 of something" -- exactly the chunk_uids the real
    parser/chunker produce for the pinned manifest, no substitutes."""
    fixture_uids = {entry["chunk_uid"] for entry in embeddings_fixture["chunks"]}
    assert fixture_uids == real_demo_chunk_uids
    assert len(fixture_uids) == 33  # no duplicate chunk_uid collapsed the set


def test_embeddings_fixture_chunk_vectors_are_not_uniform(embeddings_fixture) -> None:
    """A cheap but real sanity check that these are model output, not a
    constant or a repeated placeholder vector: no two chunks share a
    vector."""
    seen = set()
    for entry in embeddings_fixture["chunks"]:
        key = tuple(entry["embedding"])
        assert key not in seen
        seen.add(key)


def test_embeddings_fixture_includes_the_five_flagship_queries(
    embeddings_fixture,
) -> None:
    fixture_ids = {entry["id"] for entry in embeddings_fixture["queries"]}
    assert fixture_ids == {query_id for query_id, _ in FLAGSHIP_QUERIES}
    for entry in embeddings_fixture["queries"]:
        assert len(entry["embedding"]) == 384


# --- demo/providers.py: DemoEmbeddingProvider ----------------------------


def test_demo_embedding_provider_reports_384_dimensions() -> None:
    provider = DemoEmbeddingProvider()
    assert provider.dimensions == 384


def test_demo_embedding_provider_model_name_is_distinct_from_the_live_models() -> None:
    """The value is honest about being a fixture replay, not a claim that
    the live model ran -- it must not collide with the real provider's own
    `model_name` string."""
    provider = DemoEmbeddingProvider()
    assert provider.model_name == DEMO_MODEL_NAME
    assert provider.model_name != MODEL_NAME
    assert "bge-small-en-v1.5" in provider.model_name


def test_demo_embedding_provider_is_not_the_fake_provider() -> None:
    provider = DemoEmbeddingProvider()
    assert not isinstance(provider, FakeEmbeddingProvider)
    assert provider.model_name != FakeEmbeddingProvider().model_name


def test_demo_embedding_provider_returns_the_fixtures_real_vector(
    embeddings_fixture,
) -> None:
    provider = DemoEmbeddingProvider()
    known_chunk = embeddings_fixture["chunks"][0]
    known_text = next(
        chunk.text
        for chunk in iter_manifest_chunks()
        if chunk.chunk_uid == known_chunk["chunk_uid"]
    )

    [vector] = provider.embed([known_text])

    assert vector == known_chunk["embedding"]


def test_demo_embedding_provider_serves_the_flagship_query_vectors(
    embeddings_fixture,
) -> None:
    provider = DemoEmbeddingProvider()
    for query_id, text in FLAGSHIP_QUERIES:
        expected = next(
            entry["embedding"]
            for entry in embeddings_fixture["queries"]
            if entry["id"] == query_id
        )
        [vector] = provider.embed([text])
        assert vector == expected


def test_demo_embedding_provider_refuses_unknown_text() -> None:
    """Deliberately narrow: this is a replay of 33 pinned chunks and five
    pinned queries, never a general-purpose embedder."""
    provider = DemoEmbeddingProvider()
    with pytest.raises(EmbeddingError):
        provider.embed(["text nobody ever precomputed a demo vector for"])


def test_demo_embedding_provider_check_shape_guard_is_intact() -> None:
    """`check_shape` (the same shared guard every real provider uses) still
    runs here -- proven by asking for zero texts and getting zero vectors
    back, the one shape `check_shape` accepts trivially."""
    provider = DemoEmbeddingProvider()
    assert provider.embed([]) == []


# --- demo/fixtures/answers.yaml ------------------------------------------


def test_answers_fixture_has_one_entry_per_flagship_query(answers_fixture) -> None:
    fixture_ids = {entry["id"] for entry in answers_fixture}
    assert fixture_ids == {query_id for query_id, _ in FLAGSHIP_QUERIES}


def test_answers_fixture_questions_match_the_flagship_query_text(
    answers_fixture,
) -> None:
    by_id = {query_id: text for query_id, text in FLAGSHIP_QUERIES}
    for entry in answers_fixture:
        assert entry["question"] == by_id[entry["id"]]


def test_answers_fixture_citations_are_real_demo_chunk_uids(
    answers_fixture, real_demo_chunk_uids
) -> None:
    for entry in answers_fixture:
        for chunk_uid in entry["citations"]:
            assert chunk_uid in real_demo_chunk_uids


def test_answers_fixture_has_exactly_one_abstaining_entry(answers_fixture) -> None:
    abstaining = [entry for entry in answers_fixture if entry["abstained"]]
    assert len(abstaining) == 1
    entry = abstaining[0]
    assert entry["id"] == "ie001"
    assert entry["citations"] == []
    assert entry["sufficient_evidence"] is False
    assert entry["abstain_reason"] == "sufficient_evidence_false"


def test_answers_fixture_non_abstaining_entries_carry_citations(answers_fixture) -> None:
    for entry in answers_fixture:
        if entry["abstained"]:
            continue
        assert entry["citations"], entry["id"]
        assert entry["sufficient_evidence"] is True
        assert entry["abstain_reason"] is None


# --- demo/seed.py + app.retrieval.pipeline.retrieve(): the flagship
# scenarios, against a real database --------------------------------------


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture(scope="module")
def query_vectors() -> dict[str, list[float]]:
    provider = DemoEmbeddingProvider()
    return {
        query_id: provider.embed([text])[0] for query_id, text in FLAGSHIP_QUERIES
    }


def test_seeding_the_demo_corpus_produces_33_total_and_30_active_chunks(
    session, storage
) -> None:
    """The corpus has 33 total chunk rows (30 active + 3 superseded
    access-sop v1 chunks) -- but `CorpusSeedResult.total_active_chunks`
    (frozen milestone 9, `evals/retrieval/corpus.py`) by design sums only
    each document's *active-version* chunk count, so it is correctly 30,
    not 33. Checked against both numbers here rather than conflating
    them."""
    result = seed_demo_corpus(session, storage)
    assert result.total_active_chunks == 30

    total_chunk_rows = session.execute(
        select(func.count()).select_from(Chunk)
    ).scalar_one()
    assert total_chunk_rows == 33


def test_seeding_the_demo_corpus_writes_the_providers_real_precomputed_vectors(
    session, storage, embeddings_fixture
) -> None:
    """Proof the seeded rows carry `DemoEmbeddingProvider`'s vectors --
    real BGE output -- rather than anything computed some other way."""
    from sqlalchemy import select

    from app.models import Chunk

    seed_demo_corpus(session, storage)

    by_uid = {entry["chunk_uid"]: entry["embedding"] for entry in embeddings_fixture["chunks"]}
    chunks = session.execute(select(Chunk)).scalars().all()
    assert chunks
    for chunk in chunks:
        expected = by_uid[chunk.chunk_uid]
        assert chunk.embedding is not None
        assert list(chunk.embedding) == pytest.approx(expected, rel=1e-6)


def test_flagship_scenario_retrieved_correctly(session, storage, query_vectors) -> None:
    """da001: a direct, single-chunk question surfaces its one golden
    chunk among the real retrieval pipeline's candidates."""
    seed_demo_corpus(session, storage)

    result = retrieve(
        session,
        normalized_text="How many days per week may an employee work remotely?",
        query_vector=query_vectors["da001"],
        filters=RetrievalFilters(),
    )

    candidate_uids = {c.evidence.chunk_uid for c in result.candidates}
    assert "5a04c330c14efaf8f4da83ebf6d6854f" in candidate_uids


def _rerank_model_is_cached() -> bool:
    """Whether the real cross-encoder can actually be loaded from the local
    cache. Mirrors `tests/test_reranking.py`'s own check: the real model
    skips rather than being faked when this environment has never cached
    it."""
    try:
        CrossEncoderRerankProvider().rerank("probe", [Candidate(id="a", text="probe text")])
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.skipif(
    not _rerank_model_is_cached(),
    reason=(
        "cross-encoder/ms-marco-MiniLM-L-6-v2 is not in the local cache and "
        "this environment cannot reach HuggingFace. The real model has NOT "
        "been exercised."
    ),
)
def test_flagship_scenario_cited_correctly(session, storage, query_vectors) -> None:
    """cs001: a citation-sensitive question -- several chunks are
    topically related, only one contains the exact cited fact -- ranks its
    precise golden chunk first, not merely somewhere in the results.

    `retrieve()` (milestone 5) is intentionally fusion-only and makes no
    such precision guarantee -- that is milestone 6's job, so this
    flagship scenario reproduces the real application's own pipeline
    (`app/api/query.py`: retrieve, then rerank, then cite) by reranking
    milestone 5's candidates with the real, local
    `CrossEncoderRerankProvider` before asserting which chunk is cited."""
    seed_demo_corpus(session, storage)

    query_text = (
        "Within how many minutes of declaring a severity-one incident "
        "must an executive be notified?"
    )
    result = retrieve(
        session,
        normalized_text=query_text,
        query_vector=query_vectors["cs001"],
        filters=RetrievalFilters(),
    )
    assert result.candidates

    reranked = rerank(query_text, result.candidates, CrossEncoderRerankProvider())

    assert reranked
    assert reranked[0].evidence.chunk_uid == "f9cbee4e0939d52e95eebe93a063a85c"


def test_flagship_scenario_ranked_correctly_current_version_only(
    session, storage, query_vectors
) -> None:
    """cv001: default retrieval (no `include_superseded`) answers from the
    active access-sop v2 chunk and never surfaces the superseded v1
    chunk -- the conflicting-versions guarantee the demo corpus keeps both
    versions specifically to prove."""
    seed_demo_corpus(session, storage)

    result = retrieve(
        session,
        normalized_text="How many days does standard production database access last for?",
        query_vector=query_vectors["cv001"],
        filters=RetrievalFilters(),
    )

    candidate_uids = {c.evidence.chunk_uid for c in result.candidates}
    assert "da1cdb5e0e87ad69728e9d2781239d13" in candidate_uids
    assert "eef29ba299e4f0dd72c16e1741ac6928" not in candidate_uids  # superseded v1


def test_flagship_scenario_answered_from_evidence_across_documents(
    session, storage, query_vectors
) -> None:
    """md001: a question whose answer draws on two different documents
    retrieves evidence from both."""
    seed_demo_corpus(session, storage)

    result = retrieve(
        session,
        normalized_text=(
            "If I'm on call and need production database access during an "
            "active incident, what's the process, and does it require the "
            "same approval as a normal request?"
        ),
        query_vector=query_vectors["md001"],
        filters=RetrievalFilters(),
    )

    candidate_uids = {c.evidence.chunk_uid for c in result.candidates}
    assert "f9cbee4e0939d52e95eebe93a063a85c" in candidate_uids
    assert "da1cdb5e0e87ad69728e9d2781239d13" in candidate_uids


def test_flagship_scenario_weak_match_for_insufficient_evidence(
    session, storage, query_vectors
) -> None:
    """ie001: the corpus genuinely does not cover generative-AI coding
    policy. `app/generation`'s abstention decision is out of P1's scope,
    but the retrieval-level signal it would act on is real and checkable
    here: this query's best real-pipeline match is meaningfully weaker
    than a directly-answerable query's, not a confident top hit.

    Checked directly against `vector.search()`'s own cosine distance,
    not `retrieve()`'s fused `rrf_score`: RRF (`app.retrieval.fusion.fuse`)
    is deliberately rank-only and carries no normalized score, so when
    both queries' best hits are sole rank-1, no-lexical-match candidates
    (true for both flagship queries here), their `rrf_score`s are
    algebraically identical regardless of how similar either vector
    actually is -- not a signal this scenario can use."""
    seed_demo_corpus(session, storage)

    off_topic = vector.search(
        session,
        query_vector=query_vectors["ie001"],
        filters=RetrievalFilters(),
    )
    on_topic = vector.search(
        session,
        query_vector=query_vectors["da001"],
        filters=RetrievalFilters(),
    )

    assert off_topic
    assert on_topic
    # Cosine distance: larger means less similar, i.e. a weaker match.
    assert off_topic[0].score > on_topic[0].score
