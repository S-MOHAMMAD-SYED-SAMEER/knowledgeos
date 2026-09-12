"""Deterministic chunking and chunk identity.

Pure functions over normalized text. No provider calls and no I/O — the
specification requires this package to be callable over fixture data, because
it is one of the parts that gets evaluated.
"""

from app.chunking.chunker import (
    PARAGRAPH_TOLERANCE,
    Chunk,
    chunk_document,
    count_tokens,
)
from app.chunking.uid import chunk_uid

__all__ = [
    "PARAGRAPH_TOLERANCE",
    "Chunk",
    "chunk_document",
    "chunk_uid",
    "count_tokens",
]
