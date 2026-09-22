"""The demo's reranking provider: real cross-encoder/ms-marco-MiniLM-L-6-v2
scores, precomputed once and replayed from a fixture.

Not a fake and not the passthrough. PROJECT_PLAN.md's P3 architecture
decision is explicit: the cs001 flagship scenario ("cited correctly")
only ranks its golden chunk first after real cross-encoder reranking --
P1's own triage established that raw RRF fusion alone does not, and the
specification-shipped `PassthroughRerankProvider` (which never looks at
the query at all) cannot fix that either. Every score `DemoRerankProvider`
returns is the real model's own output, generated once
(`demo/generate_reranker_scores.py`) in an environment with the model
available, and replayed byte-for-byte offline here -- exactly the
discipline `demo.providers.DemoEmbeddingProvider` already applies to
embeddings.

Deliberately narrow: it knows only the exact (query, chunk) pairs recorded
in `demo/fixtures/reranker_scores.json` -- the five flagship queries' real
top-20 RRF-fused candidates -- and raises `RerankError` for anything else.
A provider that silently scored *something* for an unrecognized pair could
be mistaken for a general-purpose reranker; this one cannot be, by
construction.
"""

import hashlib
import json
from pathlib import Path

from app.providers.cross_encoder import MODEL_NAME
from app.providers.reranker import Candidate, RerankError, ScoredChunk, check_shape

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
RERANKER_SCORES_FIXTURE_PATH = FIXTURES_DIR / "reranker_scores.json"

# Distinct from `MODEL_NAME`: the scores are the real model's, but nothing
# here loads or calls it -- a log line or an audit reading this value
# should never conclude the live model ran.
DEMO_RERANK_MODEL_NAME = f"{MODEL_NAME} (demo fixture, precomputed)"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DemoRerankProvider:
    """Real cross-encoder scores, for exactly the demo's pinned flagship
    (query, chunk) pairs."""

    def __init__(self, fixture_path: Path = RERANKER_SCORES_FIXTURE_PATH) -> None:
        self._fixture_path = fixture_path
        self._scores: dict[tuple[str, str], float] | None = None

    @property
    def model_name(self) -> str:
        return DEMO_RERANK_MODEL_NAME

    def rerank(self, query: str, candidates: list[Candidate]) -> list[ScoredChunk]:
        scores = self._load()
        query_key = _sha256(query)

        result: list[ScoredChunk] = []
        for candidate in candidates:
            key = (query_key, _sha256(candidate.text))
            if key not in scores:
                raise RerankError(
                    "no precomputed demo reranker score exists for this "
                    "query/chunk pair -- DemoRerankProvider only serves "
                    "the pinned flagship queries' real top-20 candidates, "
                    "never arbitrary input"
                )
            result.append(ScoredChunk(id=candidate.id, score=scores[key]))

        check_shape(result, candidates)
        return result

    def _load(self) -> dict[tuple[str, str], float]:
        if self._scores is not None:
            return self._scores

        payload = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        if payload.get("model_name") != MODEL_NAME:
            raise RerankError(
                f"{self._fixture_path.name}: fixture model_name "
                f"{payload.get('model_name')!r} does not match the "
                f"required {MODEL_NAME!r}"
            )

        scores: dict[tuple[str, str], float] = {}
        for query_entry in payload.get("queries", []):
            query_key = query_entry["text_sha256"]
            for candidate_entry in query_entry.get("candidates", []):
                chunk_key = candidate_entry["chunk_text_sha256"]
                scores[(query_key, chunk_key)] = candidate_entry["rerank_score"]

        self._scores = scores
        return scores


__all__ = [
    "DEMO_RERANK_MODEL_NAME",
    "RERANKER_SCORES_FIXTURE_PATH",
    "DemoRerankProvider",
]
