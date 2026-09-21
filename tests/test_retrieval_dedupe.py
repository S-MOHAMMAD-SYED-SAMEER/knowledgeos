"""Deduplicating evidence by `chunk_uid`. Pure — no database, no provider;
every candidate here is built by hand.
"""

import uuid

from app.retrieval.dedupe import dedupe_evidence
from app.retrieval.vector import Candidate, ChunkEvidence


def _evidence(chunk_uid: str, text: str = "text") -> ChunkEvidence:
    return ChunkEvidence(
        chunk_uid=chunk_uid,
        text=text,
        sequence=0,
        page=None,
        section=None,
        char_start=0,
        char_end=len(text),
        token_count=1,
        document_id=uuid.uuid4(),
        document_title="doc",
        department=None,
        category=None,
        tags=(),
        version_id=uuid.uuid4(),
        version_number=1,
        version_status="active",
    )


def _candidate(chunk_uid: str, *, rank: int = 1, score: float = 0.0, text: str = "text") -> Candidate:
    return Candidate(evidence=_evidence(chunk_uid, text=text), rank=rank, score=score)


def test_a_chunk_in_only_one_list_is_kept() -> None:
    result = dedupe_evidence([_candidate("a")], [])
    assert set(result) == {"a"}


def test_a_chunk_in_both_lists_collapses_to_one_entry() -> None:
    result = dedupe_evidence([_candidate("a")], [_candidate("a")])
    assert list(result) == ["a"]


def test_two_disjoint_lists_are_merged() -> None:
    result = dedupe_evidence([_candidate("a")], [_candidate("b")])
    assert set(result) == {"a", "b"}


def test_identical_text_under_different_chunk_uids_stays_distinct() -> None:
    """The whole reason dedup keys on `chunk_uid` and never on `.text`: two
    versions of a document can share wording, and their chunks must not
    collapse into one entry because of it."""
    version_one = _candidate("uid-v1", text="the access policy")
    version_two = _candidate("uid-v2", text="the access policy")

    result = dedupe_evidence([version_one, version_two], [])

    assert set(result) == {"uid-v1", "uid-v2"}
    assert result["uid-v1"].text == result["uid-v2"].text == "the access policy"


def test_no_lists_produce_an_empty_result() -> None:
    assert dedupe_evidence() == {}


def test_the_surviving_evidence_object_is_preserved_exactly() -> None:
    candidate = _candidate("a", text="the exact wording, unmodified")
    result = dedupe_evidence([candidate], [])
    assert result["a"] is candidate.evidence


def test_three_lists_can_be_merged_at_once() -> None:
    result = dedupe_evidence([_candidate("a")], [_candidate("b")], [_candidate("a")])
    assert set(result) == {"a", "b"}
