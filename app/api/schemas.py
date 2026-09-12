"""What the API accepts and returns.

Kept apart from the models so the wire shape is a decision rather than an
accident of the schema. A column added to `documents` should not silently
become a public field.
"""

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


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
    "UploadAccepted",
    "VersionOut",
]
