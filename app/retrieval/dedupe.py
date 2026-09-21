"""Collapsing the same chunk found by both channels down to one evidence
record.

`vector.search()` and `lexical.search()` each run their own query, so a chunk
both channels find comes back as two separate `Candidate` objects — two
copies of the same row, fetched twice. `fusion.fuse()` already merges their
*ranks* by `chunk_uid`; this module merges the *evidence* the same way, so
the final response attaches exactly one text, one page, one section to each
fused entry rather than picking arbitrarily between two identical copies.

Deliberately keyed on `chunk_uid` alone, never on `.text`. Two versions of a
document can carry the same wording — the whole point of a version history —
and their chunks still have different `chunk_uid`s, because it is derived
from `document_version_id` as well as the text. Comparing text would collapse
them into one and quietly discard a distinct piece of evidence.

Pure: takes and returns plain values, no database, no provider.
"""

from collections.abc import Sequence

from app.retrieval.vector import Candidate, ChunkEvidence


def dedupe_evidence(*candidate_lists: Sequence[Candidate]) -> dict[str, ChunkEvidence]:
    """One `ChunkEvidence` per distinct `chunk_uid`, across any number of
    candidate lists.

    Where the same `chunk_uid` appears in more than one list, the first
    occurrence wins — later ones describe the identical row, since
    `chunk_uid` is unique in the database, so which copy survives changes
    nothing about the result.
    """
    evidence: dict[str, ChunkEvidence] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            evidence.setdefault(candidate.evidence.chunk_uid, candidate.evidence)
    return evidence


__all__ = ["dedupe_evidence"]
