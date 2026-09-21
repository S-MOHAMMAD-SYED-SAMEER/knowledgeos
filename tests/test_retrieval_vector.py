"""Vector retrieval against a real PostgreSQL + pgvector database."""

import pytest
import sqlalchemy as sa

from app.providers import FakeEmbeddingProvider
from app.retrieval import vector
from app.retrieval.filters import RetrievalFilters
from app.storage import LocalStorage

from .retrieval_fixtures import (
    add_draft_version,
    seed_active_version,
    seed_additional_active_version,
)


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def _captured_sql(engine, run) -> list[str]:
    captured: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        captured.append(statement)

    sa.event.listen(engine, "before_cursor_execute", _capture)
    try:
        run()
    finally:
        sa.event.remove(engine, "before_cursor_execute", _capture)
    return captured


# --- the operator ------------------------------------------------------


def test_the_operator_is_cosine_distance(session, storage, embeddings) -> None:
    seed_active_version(
        session, storage, text="production database access", embeddings=embeddings
    )
    query_vector = embeddings.embed(["production database access"])[0]

    statements = _captured_sql(
        session.get_bind(),
        lambda: vector.search(
            session, query_vector=query_vector, filters=RetrievalFilters()
        ),
    )

    assert any("<=>" in statement for statement in statements)
    assert not any("<->" in statement for statement in statements)
    assert not any("<#>" in statement for statement in statements)


# --- ordering and correctness --------------------------------------------


def test_the_identical_text_is_the_closest_match(session, storage, embeddings) -> None:
    seed_active_version(
        session,
        storage,
        text="production database access procedure",
        title="a.txt",
        embeddings=embeddings,
    )
    seed_active_version(
        session, storage, text="how to file expense reports", title="b.txt", embeddings=embeddings
    )
    query_vector = embeddings.embed(["production database access procedure"])[0]

    results = vector.search(
        session, query_vector=query_vector, filters=RetrievalFilters()
    )

    assert results[0].evidence.text == "production database access procedure"
    assert results[0].score == pytest.approx(0.0, abs=1e-6)


def test_ordering_is_ascending_by_distance(session, storage, embeddings) -> None:
    for i in range(5):
        seed_active_version(
            session,
            storage,
            text=f"unrelated content number {i}",
            title=f"d{i}.txt",
            embeddings=embeddings,
        )
    query_vector = embeddings.embed(["a query matching nothing exactly"])[0]

    results = vector.search(
        session, query_vector=query_vector, filters=RetrievalFilters()
    )

    distances = [candidate.score for candidate in results]
    assert distances == sorted(distances)


def test_rank_is_one_based_and_matches_position(session, storage, embeddings) -> None:
    for i in range(5):
        seed_active_version(
            session, storage, text=f"widget entry {i}", title=f"d{i}.txt", embeddings=embeddings
        )
    query_vector = embeddings.embed(["widget entry 0"])[0]

    results = vector.search(
        session, query_vector=query_vector, filters=RetrievalFilters()
    )

    assert [c.rank for c in results] == list(range(1, len(results) + 1))


# --- limits ----------------------------------------------------------------


def test_the_candidate_limit_is_fifty(session, storage, embeddings) -> None:
    for i in range(60):
        seed_active_version(
            session,
            storage,
            text=f"unique candidate number {i} widget",
            title=f"doc-{i}.txt",
            embeddings=embeddings,
        )
    query_vector = embeddings.embed(["widget"])[0]

    results = vector.search(
        session, query_vector=query_vector, filters=RetrievalFilters()
    )

    assert len(results) == vector.CANDIDATE_LIMIT == 50


# --- exclusions --------------------------------------------------------


def test_null_embeddings_are_excluded(session, storage, embeddings) -> None:
    """A chunk cut but never embedded must never be a candidate. The
    milestone 4 invariant means this cannot happen for an active version
    through the real pipeline — the predicate is still explicit."""
    from app.models import Chunk

    seeded = seed_active_version(session, storage, text="alpha content", embeddings=embeddings)
    chunk = (
        session.query(Chunk)
        .filter(Chunk.document_version_id == seeded.version.id)
        .one()
    )
    chunk.embedding = None
    session.commit()

    results = vector.search(
        session,
        query_vector=embeddings.embed(["alpha content"])[0],
        filters=RetrievalFilters(),
    )

    assert results == []


def test_draft_versions_are_excluded(session, storage, embeddings) -> None:
    seeded = seed_active_version(
        session, storage, text="alpha version one", embeddings=embeddings
    )
    add_draft_version(
        session, storage, document_id=seeded.document.id, text="alpha version two draft"
    )

    results = vector.search(
        session,
        query_vector=embeddings.embed(["alpha version two draft"])[0],
        filters=RetrievalFilters(),
    )

    assert all(c.evidence.version_status != "draft" for c in results)


def test_superseded_is_excluded_by_default(session, storage, embeddings) -> None:
    first = seed_active_version(
        session, storage, text="policy version one", embeddings=embeddings
    )
    seed_additional_active_version(
        session,
        storage,
        text="policy version two",
        document_id=first.document.id,
        embeddings=embeddings,
    )

    results = vector.search(
        session, query_vector=embeddings.embed(["policy"])[0], filters=RetrievalFilters()
    )

    assert results
    assert all(c.evidence.version_status == "active" for c in results)
    assert not any(c.evidence.text == "policy version one" for c in results)


def test_superseded_is_included_when_requested(session, storage, embeddings) -> None:
    first = seed_active_version(
        session, storage, text="policy version one", embeddings=embeddings
    )
    seed_additional_active_version(
        session,
        storage,
        text="policy version two",
        document_id=first.document.id,
        embeddings=embeddings,
    )

    results = vector.search(
        session,
        query_vector=embeddings.embed(["policy"])[0],
        filters=RetrievalFilters(include_superseded=True),
    )

    assert {c.evidence.version_status for c in results} == {"active", "superseded"}


# --- determinism -------------------------------------------------------


def test_tie_break_is_deterministic_by_chunk_uid(session, storage) -> None:
    """Two chunks at the identical distance always come back in the same
    order."""

    class SameVectorProvider:
        dimensions = 384
        model_name = "same-vector"

        def embed(self, texts):
            return [[0.1] * 384 for _ in texts]

    provider = SameVectorProvider()
    seed_active_version(session, storage, text="alpha content", title="a.txt", embeddings=provider)
    seed_active_version(session, storage, text="beta content", title="b.txt", embeddings=provider)

    query_vector = [0.1] * 384
    first_run = vector.search(session, query_vector=query_vector, filters=RetrievalFilters())
    second_run = vector.search(session, query_vector=query_vector, filters=RetrievalFilters())

    uids_first = [c.evidence.chunk_uid for c in first_run]
    uids_second = [c.evidence.chunk_uid for c in second_run]
    assert uids_first == uids_second
    assert uids_first == sorted(uids_first)


# --- filters -------------------------------------------------------------


def test_metadata_filters_restrict_the_channel(session, storage, embeddings) -> None:
    seed_active_version(
        session,
        storage,
        text="engineering access policy",
        department="Engineering",
        title="a.txt",
        embeddings=embeddings,
    )
    seed_active_version(
        session,
        storage,
        text="sales access policy",
        department="Sales",
        title="b.txt",
        embeddings=embeddings,
    )

    results = vector.search(
        session,
        query_vector=embeddings.embed(["access policy"])[0],
        filters=RetrievalFilters(department="Engineering"),
    )

    assert results
    assert all(c.evidence.department == "Engineering" for c in results)
