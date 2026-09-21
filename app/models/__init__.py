"""Every ORM model, re-exported.

Importing this package registers all of them on `Base.metadata`, which is what
`alembic/env.py` imports and what autogenerate compares the live database
against. A model that is not reachable from here is invisible to migrations.
"""

from app.models.chunk import Chunk
from app.models.document import (
    SOURCE_TYPE_UPLOAD,
    SOURCE_TYPES,
    Document,
    DocumentVersion,
    VersionStatus,
)
from app.models.eval_run import RETRIEVAL_SUITE, EvalRun
from app.models.ingestion_job import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    IngestionJob,
    JobStatus,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "RETRIEVAL_SUITE",
    "SOURCE_TYPES",
    "SOURCE_TYPE_UPLOAD",
    "TERMINAL_STATUSES",
    "Chunk",
    "Document",
    "DocumentVersion",
    "EvalRun",
    "IngestionJob",
    "JobStatus",
    "VersionStatus",
]
