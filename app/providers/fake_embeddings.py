"""Deterministic embeddings, for tests.

The specification permits fake embeddings for **unit tests only**, and is
blunt about why: retrieval metrics computed over fake vectors are meaningless,
and the evaluation milestone must use the real local model. So this exists to
let the indexing pipeline, its transactions, its retries and its idempotency be
tested offline — not to stand in for the model anywhere a number is produced.

Deterministic by construction: the vector for a piece of text is derived from
its SHA-256 digest, so the same text always yields the same vector, in any
process, on any machine. That is what makes an idempotency test meaningful.

Nothing in the application selects this provider. Only a test does.
"""

from hashlib import sha256

from app.providers.embeddings import DIMENSIONS, check_shape

FAKE_MODEL_NAME = "fake-deterministic"


class FakeEmbeddingProvider:
    """Vectors derived from a hash of the text."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return FAKE_MODEL_NAME

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._vector(text) for text in texts]
        check_shape(vectors, texts, self._dimensions)
        return vectors

    def _vector(self, text: str) -> list[float]:
        """Enough hash bytes to fill the width, scaled into [-1, 1)."""
        raw = b""
        counter = 0
        while len(raw) < self._dimensions:
            raw += sha256(f"{counter}:{text}".encode("utf-8")).digest()
            counter += 1
        return [(byte - 128) / 128.0 for byte in raw[: self._dimensions]]


__all__ = ["FAKE_MODEL_NAME", "FakeEmbeddingProvider"]
