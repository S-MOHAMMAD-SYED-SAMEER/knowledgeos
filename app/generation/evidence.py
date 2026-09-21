"""Selecting which of milestone 6's reranked candidates reach the model.

The specification names no selection rule for this — no top-N, no score
threshold, no token budget (see the README's milestone 8 section for the
full discussion). This project's own, documented, locked choice: the top 8
candidates by `final_rank`, fixed and deterministic, no score threshold.
Fixed rather than configurable, for the same reason the specification's own
per-channel candidate limits (top 50) and fusion cutoff (top 20) are fixed
facts rather than settings — a caller-tunable "top 8" would make that
number untrue on request.

Does not modify anything upstream. `app.retrieval` and `app.reranking` are
untouched; this module only decides, of what they already produced, which
chunks are `selected` — the boolean `retrieved_chunks.selected` records.
"""

from dataclasses import dataclass

from app.reranking.pipeline import RerankedChunk

# Locked project decision (D17): fixed at 8, not configurable.
SELECTION_LIMIT = 8


@dataclass(frozen=True)
class SelectedEvidence:
    """One reranked chunk, plus whether it was selected to reach the
    model. One entry per input candidate — including the ones that were
    *not* selected — so a `selected=false` chunk is still available for
    citation rule 3 (a citation to a retrieved-but-not-selected chunk is
    invalid) and for the `retrieved_chunks.selected` persistence record.
    """

    candidate: RerankedChunk
    selected: bool


def select_evidence(
    candidates: list[RerankedChunk], *, limit: int = SELECTION_LIMIT
) -> list[SelectedEvidence]:
    """Flag the top `limit` candidates by `final_rank` as selected.

    `final_rank` is already contiguous from 1 (see
    `app/reranking/pipeline.py`), so `final_rank <= limit` is exactly "the
    top `limit`" with no separate sort needed here.
    """
    return [
        SelectedEvidence(candidate=candidate, selected=candidate.final_rank <= limit)
        for candidate in candidates
    ]


__all__ = ["SELECTION_LIMIT", "SelectedEvidence", "select_evidence"]
