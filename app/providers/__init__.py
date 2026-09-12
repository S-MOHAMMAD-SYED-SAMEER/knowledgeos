"""Interfaces to the things this system does not implement itself.

The specification's rule is that every provider gets an interface. Milestone 4
adds the first one: embeddings. The language model and the reranker arrive with
the milestones that call them.
"""

from app.providers.bge import BgeEmbeddingProvider
from app.providers.embeddings import (
    DIMENSIONS,
    MODEL_NAME,
    EmbeddingError,
    EmbeddingProvider,
    check_shape,
)
from app.providers.fake_embeddings import FakeEmbeddingProvider

__all__ = [
    "DIMENSIONS",
    "MODEL_NAME",
    "BgeEmbeddingProvider",
    "EmbeddingError",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "check_shape",
]
