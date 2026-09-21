"""`evals/answers/semantic.py`: per-sentence entailment scoring against a
local cross-encoder-shaped scorer, and its explicit unavailable state
(D2). `FakeRerankProvider` stands in for the real cross-encoder here —
its scores are meaningless as grounding evidence, exactly like every
other pytest-only use of a fake reranker in this codebase, and nothing
here is presented as an official semantic-grounding result.
"""

import pytest

from app.providers.fake_reranker import FakeRerankProvider
from app.providers.reranker import Candidate, RerankError, ScoredChunk
from evals.answers.semantic import (
    SemanticGroundingResult,
    check_semantic_scorer_available,
    score_answer_semantic,
)
from tests.generation_fixtures import cited_answer_text


class _AlwaysFailsScorer:
    model_name = "always-fails-scorer"

    def rerank(self, query, candidates):
        raise RerankError("the scorer is unavailable")


class _FirstCallOkThenFailsScorer:
    """Passes the canary, then fails on the first real scoring call --
    proving `score_answer_semantic` treats a mid-run failure the same as
    an up-front one."""

    model_name = "flaky-scorer"

    def __init__(self) -> None:
        self._calls = 0

    def rerank(self, query, candidates):
        self._calls += 1
        if self._calls == 1:
            return [ScoredChunk(id=c.id, score=1.0) for c in candidates]
        raise RerankError("failed on a later call")


def test_check_semantic_scorer_available_returns_none_when_it_responds() -> None:
    assert check_semantic_scorer_available(FakeRerankProvider()) is None


def test_check_semantic_scorer_available_reports_the_model_name_on_failure() -> None:
    reason = check_semantic_scorer_available(_AlwaysFailsScorer())
    assert reason is not None
    assert "always-fails-scorer" in reason


def test_score_answer_semantic_skips_sentences_without_a_citation() -> None:
    result = score_answer_semantic(
        answer_text="This sentence has no citation marker at all.",
        chunk_text_by_uid={},
        scorer=FakeRerankProvider(),
    )
    assert result.status == "scored"
    assert result.sentence_scores == []


def test_score_answer_semantic_scores_every_cited_sentence() -> None:
    uid1, uid2 = "a" * 32, "b" * 32
    answer = cited_answer_text(uid1, uid2)
    chunk_text_by_uid = {uid1: "evidence text one", uid2: "evidence text two"}

    result = score_answer_semantic(
        answer_text=answer, chunk_text_by_uid=chunk_text_by_uid, scorer=FakeRerankProvider()
    )

    assert result.status == "scored"
    assert len(result.sentence_scores) == 2
    for entry in result.sentence_scores:
        assert isinstance(entry.max_score, float)
        # No threshold configured by default (D2): raw scores only.
        assert entry.supported is None
    assert result.unsupported_claim_rate is None


def test_score_answer_semantic_with_no_threshold_never_computes_unsupported_claim_rate() -> None:
    uid = "c" * 32
    result = score_answer_semantic(
        answer_text=cited_answer_text(uid),
        chunk_text_by_uid={uid: "some evidence"},
        scorer=FakeRerankProvider(),
        unsupported_claim_threshold=None,
    )
    assert result.unsupported_claim_rate is None
    assert all(entry.supported is None for entry in result.sentence_scores)


def test_score_answer_semantic_with_a_threshold_classifies_each_sentence() -> None:
    uid = "d" * 32
    scorer = FakeRerankProvider()
    # A threshold no real score could clear (the fake's scores are bounded
    # by its own hash-derived range) forces every sentence to "unsupported"
    # -- proving the classification, not a claim about the fake's numbers.
    result = score_answer_semantic(
        answer_text=cited_answer_text(uid),
        chunk_text_by_uid={uid: "some evidence"},
        scorer=scorer,
        unsupported_claim_threshold=float("inf"),
    )
    assert result.unsupported_claim_rate == 1.0
    assert all(entry.supported is False for entry in result.sentence_scores)


def test_score_answer_semantic_returns_unavailable_on_scorer_failure() -> None:
    uid = "e" * 32
    result = score_answer_semantic(
        answer_text=cited_answer_text(uid),
        chunk_text_by_uid={uid: "some evidence"},
        scorer=_AlwaysFailsScorer(),
    )
    assert isinstance(result, SemanticGroundingResult)
    assert result.status == "unavailable"
    assert result.reason is not None
    assert result.sentence_scores == []
    assert result.unsupported_claim_rate is None


def test_score_answer_semantic_unavailable_never_carries_partial_scores() -> None:
    uid1, uid2 = "f" * 32, "1" * 32
    result = score_answer_semantic(
        answer_text=cited_answer_text(uid1, uid2),
        chunk_text_by_uid={uid1: "evidence one", uid2: "evidence two"},
        scorer=_FirstCallOkThenFailsScorer(),
    )
    assert result.status == "unavailable"
    assert result.sentence_scores == []
