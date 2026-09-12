"""Making chunks findable.

Embedding writes and full-text vector maintenance — the specification's
*index writes*. This is the stage after chunking and the last one before a
version can answer a question.

Retrieval itself is milestone 5: nothing here searches.
"""

from app.indexing.writer import (
    IndexingFailed,
    chunks_to_embed,
    generate_embeddings,
    texts_of,
    write_index,
)

__all__ = [
    "IndexingFailed",
    "chunks_to_embed",
    "generate_embeddings",
    "texts_of",
    "write_index",
]
