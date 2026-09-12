"""The real embedding model, run locally.

`BAAI/bge-small-en-v1.5`, 384 dimensions, through sentence-transformers. The
specification fixes the model; nothing here chooses it.

Two deliberate properties.

**The import is lazy.** sentence-transformers pulls in torch, and importing
that at module load would make starting the application — or collecting the
test suite — pay for a machine-learning stack that most of the process never
touches. The import happens when a model is first actually needed.

**The model is loaded once and kept.** Loading is the expensive part, and a
provider that reloaded per call would make indexing unusable.

Embeddings are stored exactly as the model produces them. The specification
says nothing about normalising them to unit length, and §7 specifies cosine
distance for retrieval, which is invariant to magnitude — so an undocumented
transformation here would buy nothing and change every stored vector.

No network at run time. The model is read from the local sentence-transformers
cache; if it is not there, this raises rather than reaching for it.
"""

import logging

from app.providers.embeddings import (
    DIMENSIONS,
    MODEL_NAME,
    EmbeddingError,
    check_shape,
)

logger = logging.getLogger(__name__)


class BgeEmbeddingProvider:
    """The specification's local embedding model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load()
        try:
            vectors = model.encode(texts, convert_to_numpy=True)
        except Exception as exc:  # noqa: BLE001 - any inference failure is one case
            raise EmbeddingError(
                f"embedding failed ({type(exc).__name__})"
            ) from exc

        result = [[float(value) for value in vector] for vector in vectors]
        check_shape(result, texts, DIMENSIONS)
        return result

    def _load(self):
        """Load the model once, from the local cache.

        A missing model is an `EmbeddingError` rather than a download: the
        specification requires everything but the generation smoke test to run
        with no internet, and a provider that quietly fetched a few hundred
        megabytes mid-job would not be that.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise EmbeddingError(
                "sentence-transformers is not installed"
            ) from exc

        try:
            logger.info("Loading the embedding model %s.", self._model_name)
            self._model = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"the embedding model {self._model_name} could not be loaded "
                f"from the local cache ({type(exc).__name__})"
            ) from exc

        return self._model


__all__ = ["BgeEmbeddingProvider"]
