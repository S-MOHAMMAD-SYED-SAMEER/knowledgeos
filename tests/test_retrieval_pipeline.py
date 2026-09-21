"""The retrieval pipeline end to end: two channels, fused, deduplicated,
truncated to 20 — against a real database."""

import ast
import pathlib

import pytest

from app.providers import FakeEmbeddingProvider
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import FINAL_CANDIDATE_LIMIT, retrieve
from app.storage import LocalStorage

from .retrieval_fixtures import seed_active_version


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def test_final_rank_is_contiguous_from_one(session, storage, embeddings) -> None:
    for i in range(5):
        seed_active_version(
            session, storage, text=f"widget policy number {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    result = retrieve(
        session,
        normalized_text="widget policy",
        query_vector=embeddings.embed(["widget policy"])[0],
        filters=RetrievalFilters(),
    )

    assert [c.final_rank for c in result.candidates] == list(
        range(1, len(result.candidates) + 1)
    )


def test_truncates_to_top_twenty(session, storage, embeddings) -> None:
    for i in range(30):
        seed_active_version(
            session, storage, text=f"widget candidate {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )

    assert len(result.candidates) == FINAL_CANDIDATE_LIMIT == 20
    assert result.fused_candidate_count >= FINAL_CANDIDATE_LIMIT
    assert result.vector_candidate_count == 30
    assert result.lexical_candidate_count == 30


def test_counts_reflect_each_channel_with_a_small_corpus(
    session, storage, embeddings
) -> None:
    seed_active_version(session, storage, text="widget alpha", embeddings=embeddings)

    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )

    assert result.vector_candidate_count == 1
    assert result.lexical_candidate_count == 1
    assert result.fused_candidate_count == 1
    assert len(result.candidates) == 1


def test_empty_corpus_returns_no_candidates(session, storage, embeddings) -> None:
    result = retrieve(
        session,
        normalized_text="anything",
        query_vector=embeddings.embed(["anything"])[0],
        filters=RetrievalFilters(),
    )

    assert result.candidates == []
    assert result.vector_candidate_count == 0
    assert result.lexical_candidate_count == 0


def test_repeated_queries_are_ordered_identically(session, storage, embeddings) -> None:
    for i in range(10):
        seed_active_version(
            session, storage, text=f"widget entry {i}", title=f"d{i}.txt", embeddings=embeddings
        )

    first = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )
    second = retrieve(
        session,
        normalized_text="widget",
        query_vector=embeddings.embed(["widget"])[0],
        filters=RetrievalFilters(),
    )

    assert [c.evidence.chunk_uid for c in first.candidates] == [
        c.evidence.chunk_uid for c in second.candidates
    ]


def test_a_chunk_found_by_both_channels_carries_both_ranks(
    session, storage, embeddings
) -> None:
    seed_active_version(
        session, storage, text="widget access procedure", embeddings=embeddings
    )

    result = retrieve(
        session,
        normalized_text="widget access procedure",
        query_vector=embeddings.embed(["widget access procedure"])[0],
        filters=RetrievalFilters(),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.vector_rank == 1
    assert candidate.lexical_rank == 1
    assert candidate.rrf_score > 0


def test_a_chunk_found_by_a_channel_carries_that_channels_real_rank(
    session, storage, embeddings
) -> None:
    """End to end: the per-channel rank on the final candidate is the real
    position from that channel's own query, not a value fusion invented.
    (The complementary case — a chunk absent from one channel getting a
    `None` rank rather than an imputed one — needs no database at all and
    is proven exhaustively in `tests/test_retrieval_fusion.py`.)
    """
    seed_active_version(
        session, storage, text="widget alpha", title="a.txt", embeddings=embeddings
    )
    seed_active_version(
        session, storage, text="widget beta", title="b.txt", embeddings=embeddings
    )

    query_vector = embeddings.embed(["widget alpha"])[0]
    result = retrieve(
        session,
        normalized_text="widget",
        query_vector=query_vector,
        filters=RetrievalFilters(),
    )

    by_text = {c.evidence.text: c for c in result.candidates}
    # The query vector is identical to "widget alpha"'s own embedding, so it
    # is the closer vector match; both contain "widget" lexically.
    assert by_text["widget alpha"].vector_rank == 1
    assert by_text["widget beta"].vector_rank == 2
    assert by_text["widget alpha"].lexical_rank is not None
    assert by_text["widget beta"].lexical_rank is not None


def test_filters_reach_both_channels_through_the_pipeline(
    session, storage, embeddings
) -> None:
    seed_active_version(
        session,
        storage,
        text="engineering widget policy",
        department="Engineering",
        title="a.txt",
        embeddings=embeddings,
    )
    seed_active_version(
        session,
        storage,
        text="sales widget policy",
        department="Sales",
        title="b.txt",
        embeddings=embeddings,
    )

    result = retrieve(
        session,
        normalized_text="widget policy",
        query_vector=embeddings.embed(["widget policy"])[0],
        filters=RetrievalFilters(department="Engineering"),
    )

    assert result.candidates
    assert all(
        c.evidence.department == "Engineering" for c in result.candidates
    )


def test_the_retrieval_package_imports_no_provider() -> None:
    """The layering rule, as an executable test: no module under
    `app/retrieval/` imports a provider, a model library, or does network
    I/O of any kind."""
    package = pathlib.Path(__file__).resolve().parent.parent / "app" / "retrieval"
    forbidden_prefixes = (
        "app.providers",
        "sentence_transformers",
        "torch",
        "httpx",
        "requests",
        "anthropic",
    )

    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module}
            else:
                continue
            for name in names:
                assert not name.startswith(forbidden_prefixes), f"{path.name} imports {name}"


def test_the_retrieval_package_does_not_reference_a_session_maker_globally() -> None:
    """No hidden global state: every function that touches the database
    takes its `Session` as an argument rather than reaching for one."""
    package = pathlib.Path(__file__).resolve().parent.parent / "app" / "retrieval"
    for path in package.glob("*.py"):
        source = path.read_text()
        assert "get_engine" not in source
        assert "get_sessionmaker" not in source
        assert "sessionmaker(" not in source
