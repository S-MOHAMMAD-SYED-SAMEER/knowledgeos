"""The demo's embedding provider: real BAAI/bge-small-en-v1.5 vectors,
precomputed once and replayed from a fixture.

Not a fake. Every vector `DemoEmbeddingProvider` returns is the real
model's own output, generated once (`demo/generate_embeddings.py`) in an
environment with the model available, and replayed byte-for-byte offline
here. `FakeEmbeddingProvider`'s hash-derived vectors are unusable as a
retrieval foundation -- PROJECT_PLAN.md records a verified probe where they
failed 3 of 5 flagship scenarios -- and this provider exists so the demo
never needs them.

Deliberately narrow: it knows only the exact texts recorded in
`demo/fixtures/embeddings.json` (the pinned 33-chunk demo corpus, plus the
demo's flagship query texts) and raises `EmbeddingError` for anything
else. A provider that silently returned *something* for unknown text
could be mistaken for a general-purpose embedder; this one cannot be,
by construction.
"""

import hashlib
import json
from pathlib import Path

from app.providers.embeddings import DIMENSIONS, MODEL_NAME, EmbeddingError, check_shape

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EMBEDDINGS_FIXTURE_PATH = FIXTURES_DIR / "embeddings.json"

# Distinct from `MODEL_NAME`: the vectors are the real model's, but nothing
# here loads or calls it, and the name says so -- a log line or an audit
# reading this value should never conclude the live model ran.
DEMO_MODEL_NAME = f"{MODEL_NAME} (demo fixture, precomputed)"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DemoEmbeddingProvider:
    """Real BGE vectors, for exactly the demo corpus's pinned texts."""

    def __init__(self, fixture_path: Path = EMBEDDINGS_FIXTURE_PATH) -> None:
        self._fixture_path = fixture_path
        self._vectors: dict[str, list[float]] | None = None

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    @property
    def model_name(self) -> str:
        return DEMO_MODEL_NAME

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load()
        result: list[list[float]] = []
        for text in texts:
            key = _sha256(text)
            if key not in vectors:
                raise EmbeddingError(
                    "no precomputed demo embedding exists for this text -- "
                    "DemoEmbeddingProvider only serves the pinned demo "
                    "corpus and its flagship query texts, never arbitrary "
                    "input"
                )
            result.append(vectors[key])
        check_shape(result, texts, DIMENSIONS)
        return result

    def _load(self) -> dict[str, list[float]]:
        if self._vectors is not None:
            return self._vectors

        payload = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        if payload.get("dimensions") != DIMENSIONS:
            raise EmbeddingError(
                f"{self._fixture_path.name}: fixture dimensions "
                f"{payload.get('dimensions')!r} do not match the required "
                f"{DIMENSIONS}"
            )

        vectors: dict[str, list[float]] = {}
        for entry in payload.get("chunks", []):
            vectors[entry["text_sha256"]] = entry["embedding"]
        for entry in payload.get("queries", []):
            vectors[entry["text_sha256"]] = entry["embedding"]

        self._vectors = vectors
        return vectors


__all__ = ["DEMO_MODEL_NAME", "EMBEDDINGS_FIXTURE_PATH", "DemoEmbeddingProvider"]
