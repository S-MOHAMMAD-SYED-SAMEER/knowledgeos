"""Filter predicates: what they compile to. No database needed — these are
plain SQLAlchemy expressions, and compiling a statement is enough to prove
what SQL they produce.
"""

import uuid

import sqlalchemy as sa
from sqlalchemy import select

from app.models import Chunk, Document, DocumentVersion, VersionStatus
from app.retrieval.filters import RetrievalFilters, predicates, version_statuses


def _statement(filters: RetrievalFilters):
    return (
        select(Chunk.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(*predicates(filters))
    )


def _compiled(filters: RetrievalFilters) -> str:
    """The SQL text, with everything except JSONB values inlined.

    pgvector/JSONB literal rendering is not supported by SQLAlchemy's
    `literal_binds` compiler, so a tags filter is checked against the
    parameterized form instead — see `test_tags_filter_uses_jsonb_containment`.
    """
    return str(
        _statement(filters).compile(
            dialect=sa.dialects.postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


# --- version status -----------------------------------------------------


def test_default_filters_restrict_to_active_only() -> None:
    assert version_statuses(RetrievalFilters()) == (VersionStatus.ACTIVE,)


def test_include_superseded_adds_superseded() -> None:
    statuses = version_statuses(RetrievalFilters(include_superseded=True))
    assert set(statuses) == {VersionStatus.ACTIVE, VersionStatus.SUPERSEDED}


def test_draft_is_never_included_either_way() -> None:
    assert VersionStatus.DRAFT not in version_statuses(RetrievalFilters())
    assert VersionStatus.DRAFT not in version_statuses(
        RetrievalFilters(include_superseded=True)
    )


def test_no_filters_still_applies_the_version_predicate() -> None:
    sql = _compiled(RetrievalFilters())
    assert "document_versions.status" in sql
    assert "'active'" in sql
    assert "'draft'" not in sql


# --- metadata filters, each a WHERE clause -------------------------------


def test_document_id_filter_is_a_where_clause() -> None:
    doc_id = uuid.uuid4()
    sql = _compiled(RetrievalFilters(document_id=doc_id))
    assert "documents.id" in sql
    assert str(doc_id) in sql


def test_department_filter_is_a_where_clause() -> None:
    sql = _compiled(RetrievalFilters(department="Engineering"))
    assert "documents.department" in sql
    assert "Engineering" in sql


def test_category_filter_is_a_where_clause() -> None:
    sql = _compiled(RetrievalFilters(category="Policy"))
    assert "documents.category" in sql
    assert "Policy" in sql


def test_tags_filter_uses_jsonb_containment() -> None:
    """JSONB values can't be inlined by the literal-binds compiler, so this
    checks the parameterized SQL and its bound parameter separately."""
    sql = str(_statement(RetrievalFilters(tags=("access",))).compile(
        dialect=sa.dialects.postgresql.dialect()
    ))
    assert "documents.tags @>" in sql

    compiled = _statement(RetrievalFilters(tags=("access", "urgent"))).compile(
        dialect=sa.dialects.postgresql.dialect()
    )
    (bound_value,) = [v for v in compiled.params.values() if v == ["access", "urgent"]]
    assert bound_value == ["access", "urgent"]


def test_absent_filters_produce_no_extra_clause() -> None:
    sql = _compiled(RetrievalFilters())
    assert "documents.department" not in sql
    assert "documents.category" not in sql
    assert "documents.tags" not in sql
    # `documents.id` legitimately appears in the join condition; what must
    # be absent is a `document_id` *filter* clause, which reads `= '<uuid>'`.
    assert "documents.id =" not in sql


def test_all_filters_combine_in_one_statement() -> None:
    doc_id = uuid.uuid4()
    filters = RetrievalFilters(
        document_id=doc_id,
        department="Engineering",
        category="Policy",
        tags=("access", "urgent"),
        include_superseded=True,
    )
    sql = str(_statement(filters).compile(dialect=sa.dialects.postgresql.dialect()))

    assert "documents.id =" in sql
    assert "documents.department" in sql
    assert "documents.category" in sql
    assert "documents.tags @>" in sql
    assert "document_versions.status IN" in sql
