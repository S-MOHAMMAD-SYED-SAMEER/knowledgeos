"""Lexical retrieval against real PostgreSQL full-text search.

This exercises `chunks.tsv` — a **PostgreSQL full-text vector**, not BM25,
which the specification forbids ever calling it. Nothing here uses that
name, and it never will.
"""

import pytest
import sqlalchemy as sa

from app.retrieval import lexical
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


# --- the mechanism -----------------------------------------------------


def test_uses_websearch_to_tsquery_and_ts_rank_cd(session, storage) -> None:
    seed_active_version(session, storage, text="production database access procedure")

    statements = _captured_sql(
        session.get_bind(),
        lambda: lexical.search(
            session, normalized_text="production access", filters=RetrievalFilters()
        ),
    )

    assert any("websearch_to_tsquery(" in s for s in statements)
    assert any("ts_rank_cd(" in s for s in statements)
    for statement in statements:
        # The bare form never appears: "ts_rank_cd(" does not contain
        # "ts_rank(" as a substring (the next character is always "_").
        assert "ts_rank(" not in statement
        # Every "to_tsquery(" call is the "websearch_" form. A bare
        # `to_tsquery(...)` call would make this count one higher than the
        # "websearch_to_tsquery(" count, since the longer name contains the
        # shorter one as a substring.
        assert statement.count("to_tsquery(") == statement.count(
            "websearch_to_tsquery("
        )


def test_the_configuration_is_english() -> None:
    """The one setting the specification names. Passed as a bound parameter
    rather than inlined, so `TS_CONFIG` is the single source of truth for it
    rather than a string repeated at each call site."""
    assert lexical.TS_CONFIG == "english"


# --- ranking correctness -------------------------------------------------


def test_exact_term_is_found_where_a_paraphrase_is_not(session, storage) -> None:
    """What the lexical channel is for: an identifier a paraphrase-tolerant
    embedding would blur."""
    seed_active_version(
        session, storage, text="see policy KB-4471 for the exact procedure", title="a.txt"
    )
    seed_active_version(
        session, storage, text="see the relevant reference document instead", title="b.txt"
    )

    results = lexical.search(
        session, normalized_text="KB-4471", filters=RetrievalFilters()
    )

    assert results
    assert "KB-4471" in results[0].evidence.text


def test_ordering_is_descending_by_rank(session, storage) -> None:
    seed_active_version(session, storage, text="widget widget widget widget", title="a.txt")
    seed_active_version(
        session, storage, text="a document that mentions widget once", title="b.txt"
    )

    results = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_is_one_based_and_matches_position(session, storage) -> None:
    for i in range(4):
        seed_active_version(session, storage, text=f"widget entry {i}", title=f"d{i}.txt")

    results = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    assert [c.rank for c in results] == list(range(1, len(results) + 1))


# --- limits --------------------------------------------------------------


def test_the_candidate_limit_is_fifty(session, storage) -> None:
    for i in range(60):
        seed_active_version(
            session, storage, text=f"widget appears in document {i}", title=f"doc-{i}.txt"
        )

    results = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    assert len(results) == lexical.CANDIDATE_LIMIT == 50


# --- hostile and degenerate input -----------------------------------------


def test_hostile_input_does_not_raise(session, storage) -> None:
    seed_active_version(session, storage, text="ordinary policy text")

    hostile = 'foo & | ! ( " bar OR AND NOT )'
    results = lexical.search(
        session, normalized_text=hostile, filters=RetrievalFilters()
    )

    assert isinstance(results, list)


def test_stopword_only_query_returns_no_results_not_an_error(session, storage) -> None:
    seed_active_version(session, storage, text="the of and policy access")

    results = lexical.search(
        session, normalized_text="the of and", filters=RetrievalFilters()
    )

    assert results == []


# --- exclusions ------------------------------------------------------------


def test_draft_versions_are_excluded(session, storage) -> None:
    seeded = seed_active_version(session, storage, text="widget alpha")
    add_draft_version(
        session, storage, document_id=seeded.document.id, text="widget beta draft"
    )

    results = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    assert all(c.evidence.version_status != "draft" for c in results)


def test_superseded_excluded_by_default(session, storage) -> None:
    first = seed_active_version(session, storage, text="widget version one")
    seed_additional_active_version(
        session, storage, text="widget version two", document_id=first.document.id
    )

    results = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    assert all(c.evidence.version_status == "active" for c in results)


def test_superseded_included_when_requested(session, storage) -> None:
    first = seed_active_version(session, storage, text="widget version one")
    seed_additional_active_version(
        session, storage, text="widget version two", document_id=first.document.id
    )

    results = lexical.search(
        session,
        normalized_text="widget",
        filters=RetrievalFilters(include_superseded=True),
    )

    assert {c.evidence.version_status for c in results} == {"active", "superseded"}


# --- determinism -------------------------------------------------------


def test_tie_break_is_deterministic_by_chunk_uid(session, storage) -> None:
    seed_active_version(session, storage, text="widget widget", title="a.txt")
    seed_active_version(session, storage, text="widget widget", title="b.txt")

    first_run = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )
    second_run = lexical.search(
        session, normalized_text="widget", filters=RetrievalFilters()
    )

    uids_first = [c.evidence.chunk_uid for c in first_run]
    assert uids_first == [c.evidence.chunk_uid for c in second_run]
    assert uids_first == sorted(uids_first)


# --- filters -------------------------------------------------------------


def test_metadata_filters_restrict_the_channel(session, storage) -> None:
    seed_active_version(
        session, storage, text="widget access policy", department="Engineering", title="a.txt"
    )
    seed_active_version(
        session, storage, text="widget access policy", department="Sales", title="b.txt"
    )

    results = lexical.search(
        session,
        normalized_text="widget",
        filters=RetrievalFilters(department="Engineering"),
    )

    assert results
    assert all(c.evidence.department == "Engineering" for c in results)
