"""Reciprocal Rank Fusion — the specification's correction, not a normalized
score.

Min-max normalizing a `ts_rank_cd` value against a cosine distance is
unstable across queries: the two numbers have no shared scale and no shared
meaning. RRF sidesteps the problem entirely by never looking at either score
— only at where a chunk placed in each channel's ordering.

Pure function: two ordered lists of `chunk_uid`, in, one fused ranking out.
No database, no provider, no filesystem — which is what makes it directly
testable against hand-worked numbers, independent of anything either
retrieval channel does.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# The specification's default. Configurable — nothing else about RRF is.
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedRank:
    """One chunk's place in the fused ranking.

    `lexical_rank` and `vector_rank` are 1-based positions in their channel's
    input list, or `None` when the chunk was absent from that channel
    entirely — never an imputed worst-case rank, which the specification
    does not ask for and which would understate how little is known about a
    chunk only one channel found.
    """

    chunk_uid: str
    lexical_rank: int | None
    vector_rank: int | None
    rrf_score: float


def fuse(
    *,
    vector_ranking: Sequence[str],
    lexical_ranking: Sequence[str],
    k: int = DEFAULT_RRF_K,
) -> list[FusedRank]:
    """RRF over two rank-ordered lists of `chunk_uid`.

    `score(d) = Σ 1/(k + rank_i(d))`, summed over whichever channels a chunk
    appears in. A chunk in both lists sums two terms; a chunk in one list
    contributes only that term — there is no penalty term and no invented
    rank for the channel that never saw it.

    The result is one entry per distinct `chunk_uid` across both inputs —
    fusion cannot produce duplicates, because a `chunk_uid` is a dictionary
    key here, not a row. Ordered by score descending, `chunk_uid` ascending,
    so identical inputs always fuse to the identical output.
    """
    vector_ranks = {uid: position + 1 for position, uid in enumerate(vector_ranking)}
    lexical_ranks = {uid: position + 1 for position, uid in enumerate(lexical_ranking)}

    fused: dict[str, FusedRank] = {}
    for chunk_uid in vector_ranks.keys() | lexical_ranks.keys():
        vector_rank = vector_ranks.get(chunk_uid)
        lexical_rank = lexical_ranks.get(chunk_uid)

        score = 0.0
        if vector_rank is not None:
            score += 1.0 / (k + vector_rank)
        if lexical_rank is not None:
            score += 1.0 / (k + lexical_rank)

        fused[chunk_uid] = FusedRank(
            chunk_uid=chunk_uid,
            lexical_rank=lexical_rank,
            vector_rank=vector_rank,
            rrf_score=score,
        )

    return sorted(fused.values(), key=lambda entry: (-entry.rrf_score, entry.chunk_uid))


__all__ = ["DEFAULT_RRF_K", "FusedRank", "fuse"]
