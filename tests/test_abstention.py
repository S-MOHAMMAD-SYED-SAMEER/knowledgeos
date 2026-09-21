"""Abstention decisions: pure functions, no provider, no I/O.

The specification (§9): triggered when top rerank score falls below a
threshold, or when the model sets `sufficient_evidence=false`. This
project adds a third, pre-LLM case: zero retrieved candidates must never
reach the model at all.
"""

from app.generation.abstention import (
    DEFAULT_ABSTENTION_RERANK_THRESHOLD,
    REASON_NO_CANDIDATES,
    REASON_RERANK_SCORE_BELOW_THRESHOLD,
    REASON_SUFFICIENT_EVIDENCE_FALSE,
    post_llm_abstention,
    pre_llm_abstention,
)


def test_the_default_threshold_is_none_never_invented() -> None:
    """Locked project decision D6: never a chosen number."""
    assert DEFAULT_ABSTENTION_RERANK_THRESHOLD is None


# --- pre-LLM: zero candidates --------------------------------------------


def test_zero_candidates_always_abstains_regardless_of_threshold() -> None:
    decision = pre_llm_abstention(
        candidate_count=0, top_rerank_score=None, threshold=None
    )
    assert decision.abstain
    assert decision.reason == REASON_NO_CANDIDATES


def test_zero_candidates_abstains_even_with_a_threshold_configured() -> None:
    decision = pre_llm_abstention(
        candidate_count=0, top_rerank_score=None, threshold=0.5
    )
    assert decision.abstain
    assert decision.reason == REASON_NO_CANDIDATES


# --- pre-LLM: rerank score threshold ---------------------------------------


def test_a_none_threshold_never_triggers_the_score_gate() -> None:
    """The default, uncalibrated state -- only the zero-candidate case can
    abstain."""
    decision = pre_llm_abstention(
        candidate_count=5, top_rerank_score=-10.0, threshold=None
    )
    assert not decision.abstain
    assert decision.reason is None


def test_a_score_below_a_configured_threshold_abstains() -> None:
    decision = pre_llm_abstention(
        candidate_count=5, top_rerank_score=0.1, threshold=0.5
    )
    assert decision.abstain
    assert decision.reason == REASON_RERANK_SCORE_BELOW_THRESHOLD


def test_a_score_at_or_above_a_configured_threshold_does_not_abstain() -> None:
    decision = pre_llm_abstention(
        candidate_count=5, top_rerank_score=0.5, threshold=0.5
    )
    assert not decision.abstain


def test_a_score_above_a_configured_threshold_does_not_abstain() -> None:
    decision = pre_llm_abstention(
        candidate_count=5, top_rerank_score=0.9, threshold=0.5
    )
    assert not decision.abstain


# --- post-LLM: the model's own sufficient_evidence -------------------------


def test_sufficient_evidence_false_abstains() -> None:
    decision = post_llm_abstention(sufficient_evidence=False)
    assert decision.abstain
    assert decision.reason == REASON_SUFFICIENT_EVIDENCE_FALSE


def test_sufficient_evidence_true_does_not_abstain() -> None:
    decision = post_llm_abstention(sufficient_evidence=True)
    assert not decision.abstain
    assert decision.reason is None
