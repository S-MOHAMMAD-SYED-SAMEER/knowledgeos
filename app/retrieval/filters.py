"""What a query may be narrowed by, and the one place that turns it into SQL.

Both retrieval channels call `predicates()` for the same `RetrievalFilters`,
so the SQL each one runs is the identical set of `WHERE` clauses rather than
two hand-written copies that could drift apart. The specification is explicit
that this must happen *inside* both retrieval queries — "never post-filter,
and never pass unfiltered results to the model with an instruction to ignore
them" — and one shared function is what makes that true by construction
instead of by discipline.

No database, no provider, no I/O: this module only builds SQLAlchemy
expressions. Whether they filter anything is decided by whoever executes the
statement they are attached to.
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import ColumnElement

from app.models import Document, DocumentVersion, VersionStatus


@dataclass(frozen=True)
class RetrievalFilters:
    """What a caller may narrow retrieval by.

    Every field maps to a column that already has an index: `document_id` to
    the primary key, `department`/`category` to their own btree, `tags` to
    the GIN index on `documents.tags`. Nothing here is a research feature —
    `page`, `section` and `effective_date` are citation metadata, not filters
    the specification names, and are deliberately absent.
    """

    document_id: uuid.UUID | None = None
    department: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    include_superseded: bool = False


def version_statuses(filters: RetrievalFilters) -> tuple[VersionStatus, ...]:
    """Which version statuses this query may see.

    `draft` is never included, in either mode. A draft may not even be
    indexed yet — the milestone 4 invariant is `active ⇒ indexed`, nothing
    weaker — and it is not "history" either, so `include_superseded` does not
    reach it.
    """
    if filters.include_superseded:
        return (VersionStatus.ACTIVE, VersionStatus.SUPERSEDED)
    return (VersionStatus.ACTIVE,)


def predicates(filters: RetrievalFilters) -> list[ColumnElement[bool]]:
    """The `WHERE` clauses both channels apply.

    Assumes the statement already joins `Chunk` to `DocumentVersion` to
    `Document` — this function only ever adds conditions, it never joins.
    """
    clauses: list[ColumnElement[bool]] = [
        DocumentVersion.status.in_(version_statuses(filters))
    ]

    if filters.document_id is not None:
        clauses.append(Document.id == filters.document_id)
    if filters.department is not None:
        clauses.append(Document.department == filters.department)
    if filters.category is not None:
        clauses.append(Document.category == filters.category)
    if filters.tags:
        # JSONB containment: every requested tag must be present. Compiles to
        # `documents.tags @> :tags`, the operator the specification names.
        clauses.append(Document.tags.contains(list(filters.tags)))

    return clauses


__all__ = ["RetrievalFilters", "predicates", "version_statuses"]
