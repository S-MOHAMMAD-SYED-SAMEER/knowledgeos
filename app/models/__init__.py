"""Every ORM model, re-exported.

Importing this package registers all of them on `Base.metadata`, which is what
`alembic/env.py` imports and what autogenerate compares the live database
against. A model that is not reachable from here is invisible to migrations.
"""

from app.models.answer import Answer
from app.models.chunk import Chunk
from app.models.document import (
    SOURCE_TYPE_UPLOAD,
    SOURCE_TYPES,
    Document,
    DocumentVersion,
    VersionStatus,
)
from app.models.eval_run import RETRIEVAL_SUITE, EvalRun
from app.models.feedback import (
    MAX_REASON_LENGTH,
    RATINGS,
    RATING_HELPFUL,
    RATING_NOT_HELPFUL,
    Feedback,
)
from app.models.ingestion_job import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    IngestionJob,
    JobStatus,
)
from app.models.query import Query
from app.models.retrieved_chunk import RetrievedChunk

__all__ = [
    "ALLOWED_TRANSITIONS",
    "MAX_REASON_LENGTH",
    "RATINGS",
    "RATING_HELPFUL",
    "RATING_NOT_HELPFUL",
    "RETRIEVAL_SUITE",
    "SOURCE_TYPES",
    "SOURCE_TYPE_UPLOAD",
    "TERMINAL_STATUSES",
    "Answer",
    "Chunk",
    "Document",
    "DocumentVersion",
    "EvalRun",
    "Feedback",
    "IngestionJob",
    "JobStatus",
    "Query",
    "RetrievedChunk",
    "VersionStatus",
]
