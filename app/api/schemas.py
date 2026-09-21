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
    "AnswerDetailOut",
    "DocumentDetail",
    "DocumentOut",
    "DocumentPage",
    "JobOut",
    "QueryFiltersIn",
    "QueryCandidateOut",
    "QueryCounts",
    "QueryDetailOut",
    "QueryDocumentOut",
    "QueryRequest",
    "QueryResponse",
    "QueryVersionOut",
    "RetrievedChunkOut",
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
    """One reranked chunk, and (as of milestone 8) whether it was selected
    to reach the generation model.

    `final_rank` is the chunk's position after reranking — the rank a caller
    actually sees. `fusion_rank` is milestone 5's own `final_rank`: the
    position after RRF fusion, before reranking, kept for transparency about
    what reranking changed rather than discarded once it runs. `selected` is
    milestone 8's own field: the top 8 candidates by `final_rank`
    (`app.generation.evidence.SELECTION_LIMIT`) reach the model; the rest
    are retrieved but not selected, and a citation to one of them is
    invalid (specification §9, citation rule 3).
    """

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
    fusion_rank: int
    rerank_score: float
    final_rank: int
    selected: bool


class QueryCounts(BaseModel):
    """How many candidates each stage produced, so the response can be
    understood without re-running the query."""

    vector: int
    lexical: int
    fused: int
    returned: int


class QueryResponse(BaseModel):
    """Evidence (milestones 5-6) plus, as of milestone 8, the generated
    answer and its validation record. No cost, confidence, or latency
    field — those belong to milestone 9/10 and are not added here.
    """

    query: str
    normalized_query: str
    filters: QueryFiltersIn
    candidates: list[QueryCandidateOut]
    counts: QueryCounts

    # --- generation (milestone 8) -----------------------------------
    query_id: uuid.UUID
    answer_id: uuid.UUID
    answer: str
    abstained: bool
    citations: list[str]
    citation_valid: bool
    grounded: bool
    grounding_detail: dict
    model: str | None
    prompt_version: str


# --- persisted query detail (milestone 8) -----------------------------


class RetrievedChunkOut(BaseModel):
    """One row of the persisted `retrieved_chunks` retrieval/reranking
    snapshot, as `GET /queries/{id}` reads it back."""

    model_config = ConfigDict(from_attributes=True)

    chunk_uid: str
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float
    rerank_score: float
    final_rank: int
    selected: bool


class AnswerDetailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    answer_text: str
    abstained: bool
    citations: list[str]
    citation_valid: bool
    grounded: bool
    grounding_detail: dict
    created_at: datetime


class QueryDetailOut(BaseModel):
    """`GET /queries/{id}` — the persisted query, its retrieval snapshot,
    and its answer, read back from `queries`/`retrieved_chunks`/`answers`
    rather than recomputed."""

    id: uuid.UUID
    query_text: str
    normalized_text: str
    filters: QueryFiltersIn
    model: str | None
    prompt_version: str | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime
    retrieved_chunks: list[RetrievedChunkOut]
    answer: AnswerDetailOut | None
