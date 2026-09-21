"""What the API accepts and returns.

Kept apart from the models so the wire shape is a decision rather than an
accident of the schema. A column added to `documents` should not silently
become a public field.
"""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    source_type: str
    department: str | None
    category: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    version_number: int
    status: str
    original_filename: str
    effective_date: date | None
    # Both null until a later milestone parses the file. Present in the
    # response so the shape does not change when they start being filled.
    checksum: str | None
    page_count: int | None
    created_at: datetime


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_version_id: uuid.UUID
    status: str
    stage_error: str | None
    attempts: int
    started_at: datetime | None
    completed_at: datetime | None


class UploadAccepted(BaseModel):
    """The answer to an upload: what was created, in one object."""

    document: DocumentOut
    version: VersionOut
    job: JobOut


class DocumentDetail(BaseModel):
    """A document and its whole version history."""

    document: DocumentOut
    versions: list[VersionOut]


class DocumentPage(BaseModel):
    documents: list[DocumentOut]
    limit: int
    offset: int


# `storage_path` appears in none of these. It describes where this system
# keeps a file, which is nobody else's business and is exactly the sort of
# detail the specification says error responses must not carry either.

__all__ = [
    "DocumentDetail",
    "DocumentOut",
    "DocumentPage",
    "JobOut",
    "QueryFiltersIn",
    "QueryCandidateOut",
    "QueryCounts",
    "QueryDocumentOut",
    "QueryRequest",
    "QueryResponse",
    "QueryVersionOut",
    "UploadAccepted",
    "VersionOut",
]


# --- retrieval (milestone 5) -------------------------------------------------


class QueryFiltersIn(BaseModel):
    """The metadata a `/query` caller may narrow retrieval by.

    Every field here maps to one already-indexed column. `page`, `section`
    and `effective_date` are deliberately absent: the specification carries
    them as citation metadata, not as filters.
    """

    document_id: uuid.UUID | None = None
    department: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    include_superseded: bool = False


class QueryRequest(BaseModel):
    query: str
    filters: QueryFiltersIn = Field(default_factory=QueryFiltersIn)


class QueryDocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    department: str | None
    category: str | None
    tags: list[str]


class QueryVersionOut(BaseModel):
    id: uuid.UUID
    version_number: int
    status: str


class QueryCandidateOut(BaseModel):
    """One retrieved chunk. Evidence only — no answer, no citation, no
    rerank score: those belong to milestones 6 and 8."""

    chunk_uid: str
    text: str
    sequence: int
    page: int | None
    section: str | None
    char_start: int
    char_end: int
    document: QueryDocumentOut
    version: QueryVersionOut
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float
    final_rank: int


class QueryCounts(BaseModel):
    """How many candidates each stage produced, so the response can be
    understood without re-running the query."""

    vector: int
    lexical: int
    fused: int
    returned: int


class QueryResponse(BaseModel):
    query: str
    normalized_query: str
    filters: QueryFiltersIn
    candidates: list[QueryCandidateOut]
    counts: QueryCounts
