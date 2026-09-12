"""The identifier that makes re-indexing idempotent.

From the specification:

    chunk_uid = sha256(document_version_id : sequence : normalized_text)[:32]

Re-running a job for a version deletes and rewrites its chunks. Because this
identifier is derived from the content rather than allocated, the rewritten
rows carry the same identifiers as the ones they replace — which is what makes
a re-index produce identical rows instead of duplicates, and what lets a
citation recorded against a chunk still mean something after one.

The canonical form is pinned exactly, because the value is immutable once any
document exists:

* `document_version_id` — the ordinary lower-case hyphenated UUID string.
* `sequence` — a plain decimal integer.
* separator — a literal `:`, as written in the specification.
* `normalized_text` — the chunk's text, exactly as normalization produced it.
* encoding — UTF-8.
* digest — SHA-256, lower-case hexadecimal, **first 32 characters** (128 bits).

Three things are deliberately *not* inputs: the page, the section, and the
document id. A chunk whose page metadata is corrected keeps its identifier,
because its text and position in the version have not changed.

`tests/test_chunk_uid.py` pins a golden value so this cannot drift unnoticed.
"""

import uuid
from hashlib import sha256

SEPARATOR = ":"
UID_LENGTH = 32


def chunk_uid(document_version_id: uuid.UUID | str, sequence: int, normalized_text: str) -> str:
    """The deterministic identifier for one chunk of one version."""
    payload = (
        f"{document_version_id}{SEPARATOR}{sequence}{SEPARATOR}{normalized_text}"
    )
    return sha256(payload.encode("utf-8")).hexdigest()[:UID_LENGTH]


__all__ = ["SEPARATOR", "UID_LENGTH", "chunk_uid"]
