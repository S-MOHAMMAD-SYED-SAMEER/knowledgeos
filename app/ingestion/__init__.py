"""Getting documents into the system, and turning them into chunks.

Validation of what arrives, the rows that record it, the pipeline that parses
and chunks a stored file, and the runner that drives it. Making chunks
findable — embeddings and a full-text vector — is milestone 4.
"""

from app.ingestion.service import (
    DocumentNotFound,
    Ingested,
    VersionNumberContested,
    add_version,
    create_document_with_version,
    ingest_with_retry,
    next_version_number,
    promote_version,
    storage_key,
)
from app.ingestion.pipeline import (
    StageFailed,
    build_chunks,
    parse_and_normalize,
    replace_chunks,
    run_job,
)
from app.ingestion.runner import (
    IngestionRunner,
    Outcome,
    claim_next_job,
    process_one,
    run_pending,
)
from app.ingestion.validation import (
    EmptyFile,
    FileTooLarge,
    UnsupportedFormat,
    UploadRejected,
    ValidatedUpload,
    sanitize_filename,
    validate,
)

__all__ = [
    "DocumentNotFound",
    "IngestionRunner",
    "Outcome",
    "StageFailed",
    "EmptyFile",
    "FileTooLarge",
    "Ingested",
    "UnsupportedFormat",
    "UploadRejected",
    "ValidatedUpload",
    "VersionNumberContested",
    "add_version",
    "build_chunks",
    "claim_next_job",
    "create_document_with_version",
    "ingest_with_retry",
    "next_version_number",
    "parse_and_normalize",
    "process_one",
    "promote_version",
    "replace_chunks",
    "run_job",
    "run_pending",
    "sanitize_filename",
    "storage_key",
    "validate",
]
