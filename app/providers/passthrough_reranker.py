"""The specification-mandated passthrough reranker.

Unlike `FakeEmbeddingProvider`, this is not a test-only stand-in. The
specification requires it as a shipped, selectable configuration: "so the
pipeline is testable without the model and so evals can measure what
reranking actually adds" (§8). It preserves whatever order it is given —
which, in the query pipeline, is the RRF/fusion order milestone 5 already
produced — and is what "reranking disabled" is expected to mean for a future
evaluation run comparing retrieval with and without reranking.

Scores here are synthetic and exist only to satisfy `RerankProvider`'s
contract of one score per candidate; they carry no signal beyond input
position and must never be compared across calls or read as a real
relevance measure. Nothing in this milestone selects this provider by
default — the production default is the real cross-encoder — but it is not
forbidden from being selected explicitly, the way a test-only fake is.
"""

from app.providers.reranker import Candidate, ScoredChunk, check_shape


class PassthroughRerankProvider:
    """Preserves input order. No model, no inference, no load failure."""

    @property
    def model_name(self) -> str:
        return "passthrough"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[ScoredChunk]:
        del query  # Order-preserving only; the query never affects the score.

        count = len(candidates)
        # Descending by position, so sorting the result by score DESC
        # reproduces exactly the order `candidates` was given in.
        result = [
            ScoredChunk(id=candidate.id, score=float(count - index))
            for index, candidate in enumerate(candidates)
        ]
        check_shape(result, candidates)
        return result


__all__ = ["PassthroughRerankProvider"]
