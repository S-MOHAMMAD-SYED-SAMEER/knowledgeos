"""A deterministic reranker that actually reorders, for tests only.

`PassthroughRerankProvider` provably preserves order by construction, which
makes it useless for proving that the reranking stage *can* change an
ordering. This exists for exactly that: a score derived from a hash of the
`(query, text)` pair, independent of the candidate's position, so the same
pair always scores the same way — in any process, on any machine — and a
test can assert the reranked result is genuinely reordered rather than
merely re-labelled.

Nothing in the application selects this provider. Only a test does — the
same rule `FakeEmbeddingProvider` follows, for the same reason: a score with
no relation to actual relevance is meaningless anywhere a real ranking
number is expected.
"""

from hashlib import sha256

from app.providers.reranker import Candidate, ScoredChunk, check_shape

FAKE_RERANKER_MODEL_NAME = "fake-deterministic-reranker"


class FakeRerankProvider:
    """Scores derived from a hash of `(query, text)`.

    Deterministic and independent of input order, so it can — and, given
    distinct candidate texts, does — reorder candidates.
    """

    @property
    def model_name(self) -> str:
        return FAKE_RERANKER_MODEL_NAME

    def rerank(self, query: str, candidates: list[Candidate]) -> list[ScoredChunk]:
        result = [
            ScoredChunk(id=candidate.id, score=self._score(query, candidate.text))
            for candidate in candidates
        ]
        check_shape(result, candidates)
        return result

    def _score(self, query: str, text: str) -> float:
        """A float in `[0, 1)`, derived from the first 4 hash bytes."""
        digest = sha256(f"{query}:{text}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") / 2**32


__all__ = ["FAKE_RERANKER_MODEL_NAME", "FakeRerankProvider"]
