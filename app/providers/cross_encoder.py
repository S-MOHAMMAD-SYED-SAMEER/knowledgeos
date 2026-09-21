"""The real reranking model, run locally.

`cross-encoder/ms-marco-MiniLM-L-6-v2`, through
`sentence_transformers.CrossEncoder`. The specification fixes the model;
nothing here chooses it.

Same two deliberate properties as `app/providers/bge.py`, for the same
reasons.

**The import is lazy.** `CrossEncoder` pulls in the same torch-backed stack
as the embedding model; importing it at module load would make starting the
application, or collecting the test suite, pay for a machine-learning stack
most of the process never touches.

**The model is loaded once and kept.** Loading is the expensive part, and a
provider that reloaded per call would make reranking every query slow for no
reason.

No network at run time. The model is read from the local
sentence-transformers cache; if it is not there, this raises rather than
downloading it — reranking must never silently substitute another model,
fall back to the passthrough, or reach an external API.
"""

import logging

from app.providers.reranker import Candidate, RerankError, ScoredChunk, check_shape

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderRerankProvider:
    """The specification's local reranking model."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, candidates: list[Candidate]) -> list[ScoredChunk]:
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - any inference failure is one case
            raise RerankError(f"reranking failed ({type(exc).__name__})") from exc

        result = [
            ScoredChunk(id=candidate.id, score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        check_shape(result, candidates)
        return result

    def _load(self):
        """Load the model once, from the local cache.

        A missing model is a `RerankError` rather than a download: the
        specification requires everything but the generation smoke test to
        run with no internet, and a provider that quietly fetched a few
        hundred megabytes mid-query would not be that.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RerankError("sentence-transformers is not installed") from exc

        try:
            logger.info("Loading the reranking model %s.", self._model_name)
            self._model = CrossEncoder(self._model_name, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            raise RerankError(
                f"the reranking model {self._model_name} could not be loaded "
                f"from the local cache ({type(exc).__name__})"
            ) from exc

        return self._model


__all__ = ["MODEL_NAME", "CrossEncoderRerankProvider"]
