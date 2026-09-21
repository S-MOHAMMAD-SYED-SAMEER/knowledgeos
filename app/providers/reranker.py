"""The reranking boundary.

`RerankProvider.rerank(query, chunks) -> list[ScoredChunk]` is the
specification's own signature (§8). Both `Candidate` and `ScoredChunk` are
plain, provider-owned types — a chunk identity plus text going in, an
identity plus a score coming back — deliberately not ORM rows and not
`app.retrieval`'s `ChunkEvidence`. `EmbeddingProvider.embed` takes plain
strings rather than `Chunk` rows for the same reason: a provider interface
must not import the domain of whoever happens to call it, so it can be
tested and swapped independently of everything above it.

Two implementations ship under `app/providers/`, mirroring
`embeddings.py` + `bge.py` + `fake_embeddings.py`:

* `CrossEncoderRerankProvider` (`app/providers/cross_encoder.py`) —
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, run locally through
  sentence-transformers. The specification's default.
* `PassthroughRerankProvider` (`app/providers/passthrough_reranker.py`) —
  preserves input order. The specification names this explicitly as a
  shipped, selectable configuration — "so the pipeline is testable without
  the model and so evals can measure what reranking actually adds" — not as
  a test-only convenience the way a fake embedding provider is.

A third, deterministic-but-order-changing double
(`app/providers/fake_reranker.py`) exists for tests only, the same way
`FakeEmbeddingProvider` does: passthrough provably never reorders anything,
so it cannot by itself prove that the reranking stage *can*.

Higher score means more relevant. The orchestration in `app/reranking/`
sorts by score descending; nothing here normalizes or rescales a score —
each provider's numbers stand on their own.
"""

from typing import NamedTuple, Protocol, runtime_checkable


class RerankError(RuntimeError):
    """Reranking could not be completed.

    The message names what failed structurally — a model that would not
    load, an inference error, a shape mismatch — and never the text that was
    being scored.
    """


class Candidate(NamedTuple):
    """One item to be scored: an opaque identity plus the text to rank.

    `id` is never interpreted by a provider — it exists only to be echoed
    back on the corresponding `ScoredChunk`, which is what lets a caller map
    scores back to whatever it actually cares about (a `chunk_uid`, in this
    application) without the provider needing to know what a chunk is.
    """

    id: str
    text: str


class ScoredChunk(NamedTuple):
    """One candidate's score. `id` matches the `Candidate.id` it came from."""

    id: str
    score: float


@runtime_checkable
class RerankProvider(Protocol):
    """Scores each candidate against a query. Higher is more relevant."""

    @property
    def model_name(self) -> str:
        """Which model produced the scores, for the record."""
        ...

    def rerank(self, query: str, candidates: list[Candidate]) -> list[ScoredChunk]:
        """One score per candidate.

        Raises `RerankError` rather than returning something the caller
        would have to guess about. The order of the result need not match
        the order of `candidates` — callers match by `id`, never by
        position — but every input `id` must appear exactly once.
        """
        ...


def check_shape(scored: list[ScoredChunk], candidates: list[Candidate]) -> None:
    """Refuse a result that does not match what was asked for.

    Checked here rather than left to whoever calls a provider: a shape or
    identity mismatch would otherwise surface as a confusing `KeyError`
    several layers away from the provider that caused it.
    """
    if len(scored) != len(candidates):
        raise RerankError(
            f"the provider returned {len(scored)} scores for "
            f"{len(candidates)} candidates"
        )
    expected_ids = {candidate.id for candidate in candidates}
    returned_ids = {item.id for item in scored}
    if expected_ids != returned_ids:
        raise RerankError(
            "the provider returned scores for different candidates than it "
            "was given"
        )


__all__ = ["Candidate", "RerankError", "RerankProvider", "ScoredChunk", "check_shape"]
