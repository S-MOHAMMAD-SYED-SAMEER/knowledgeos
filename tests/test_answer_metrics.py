"""`evals/answers/metrics.py`: pure aggregation over `AnswerAttempt`
records -- citation validity/coverage, grounded-answer rate,
unsupported-claim rate, abstention precision/recall, latency, tokens, and
cost. No provider, no database: every attempt here is built by hand.
"""

import pytest

from app.observability.pricing import PricingError
from evals.answers.metrics import (
    AnswerAttempt,
    AttemptOutcome,
    abstention_precision_recall,
    citation_coverage_mean,
    cost_summary,
    grounded_answer_rate,
    invalid_citation_rate,
    latency_summary,
    token_summary,
    unsupported_claim_rate,
)
from evals.answers.semantic import SemanticGroundingResult


def _attempt(
    *,
    question_id: str = "q1",
    expected_abstain: bool = False,
    outcome: AttemptOutcome = AttemptOutcome.SUCCESS,
    abstained: bool | None = False,
    citation_valid: bool | None = True,
    citation_coverage: float | None = 1.0,
    grounded: bool | None = True,
    semantic: SemanticGroundingResult | None = None,
    model_name: str | None = "test-model",
    input_tokens: int | None = 100,
    output_tokens: int | None = 50,
    retrieval_ms: float | None = 10.0,
    rerank_ms: float | None = 5.0,
    llm_ms: float | None = 20.0,
    total_ms: float | None = 35.0,
) -> AnswerAttempt:
    return AnswerAttempt(
        question_id=question_id,
        expected_abstain=expected_abstain,
        outcome=outcome,
        abstained=abstained,
        citation_valid=citation_valid,
        citation_coverage=citation_coverage,
        grounded=grounded,
        semantic=semantic,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
    )


# --- invalid citation rate -------------------------------------------------


def test_invalid_citation_rate_counts_citation_invalid_over_validated() -> None:
    attempts = [
        _attempt(outcome=AttemptOutcome.SUCCESS),
        _attempt(outcome=AttemptOutcome.CITATION_INVALID, abstained=None, citation_valid=False),
        _attempt(outcome=AttemptOutcome.MALFORMED_OUTPUT, abstained=None),
    ]
    assert invalid_citation_rate(attempts) == pytest.approx(1 / 3)


def test_invalid_citation_rate_excludes_pre_llm_abstentions() -> None:
    attempts = [
        _attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None),
        _attempt(outcome=AttemptOutcome.SUCCESS),
    ]
    assert invalid_citation_rate(attempts) == 0.0


def test_invalid_citation_rate_undefined_with_only_pre_llm_abstentions() -> None:
    attempts = [_attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None)]
    with pytest.raises(ValueError):
        invalid_citation_rate(attempts)


# --- citation coverage ------------------------------------------------------


def test_citation_coverage_mean_over_successful_non_abstained_attempts() -> None:
    attempts = [
        _attempt(citation_coverage=1.0),
        _attempt(citation_coverage=0.5),
        _attempt(outcome=AttemptOutcome.CITATION_INVALID, abstained=None, citation_coverage=None),
    ]
    assert citation_coverage_mean(attempts) == pytest.approx(0.75)


def test_citation_coverage_mean_undefined_with_no_successful_attempts() -> None:
    attempts = [_attempt(outcome=AttemptOutcome.MALFORMED_OUTPUT, abstained=None, citation_coverage=None)]
    with pytest.raises(ValueError):
        citation_coverage_mean(attempts)


# --- grounded answer rate ---------------------------------------------------


def test_grounded_answer_rate_excludes_pre_llm_abstentions() -> None:
    attempts = [
        _attempt(grounded=True),
        _attempt(grounded=False),
        _attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None, grounded=None),
    ]
    assert grounded_answer_rate(attempts) == pytest.approx(0.5)


def test_grounded_answer_rate_counts_rejected_attempts_as_not_grounded() -> None:
    attempts = [
        _attempt(grounded=True),
        _attempt(outcome=AttemptOutcome.CITATION_INVALID, abstained=None, grounded=None),
    ]
    assert grounded_answer_rate(attempts) == pytest.approx(0.5)


# --- unsupported claim rate --------------------------------------------------


def test_unsupported_claim_rate_is_none_when_never_computed() -> None:
    attempts = [_attempt(semantic=None)]
    assert unsupported_claim_rate(attempts) is None


def test_unsupported_claim_rate_is_none_when_semantic_never_produced_a_rate() -> None:
    semantic = SemanticGroundingResult(
        status="scored", reason=None, sentence_scores=[], unsupported_claim_rate=None
    )
    attempts = [_attempt(semantic=semantic)]
    assert unsupported_claim_rate(attempts) is None


def test_unsupported_claim_rate_averages_over_attempts_with_a_computed_rate() -> None:
    s1 = SemanticGroundingResult(status="scored", reason=None, sentence_scores=[], unsupported_claim_rate=0.0)
    s2 = SemanticGroundingResult(status="scored", reason=None, sentence_scores=[], unsupported_claim_rate=1.0)
    attempts = [_attempt(semantic=s1), _attempt(semantic=s2)]
    assert unsupported_claim_rate(attempts) == pytest.approx(0.5)


# --- abstention precision/recall --------------------------------------------


def test_abstention_recall_and_precision_on_a_mixed_set() -> None:
    attempts = [
        _attempt(question_id="a", expected_abstain=True, abstained=True),
        _attempt(question_id="b", expected_abstain=True, abstained=False),
        _attempt(question_id="c", expected_abstain=False, abstained=True),
        _attempt(question_id="d", expected_abstain=False, abstained=False),
    ]
    score = abstention_precision_recall(attempts)
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.positive_count == 2
    assert score.recall == pytest.approx(0.5)
    assert score.precision == pytest.approx(0.5)


def test_abstention_precision_is_none_when_nothing_predicted_abstain() -> None:
    attempts = [_attempt(question_id="a", expected_abstain=True, abstained=False)]
    score = abstention_precision_recall(attempts)
    assert score.precision is None
    assert score.recall == 0.0


def test_rejected_attempts_score_as_predicted_non_abstention() -> None:
    attempts = [
        _attempt(
            question_id="a",
            expected_abstain=True,
            outcome=AttemptOutcome.CITATION_INVALID,
            abstained=None,
        )
    ]
    score = abstention_precision_recall(attempts)
    assert score.true_positives == 0
    assert score.false_negatives == 1
    assert score.recall == 0.0


def test_abstention_recall_undefined_with_zero_expected_abstain_questions() -> None:
    attempts = [_attempt(expected_abstain=False)]
    with pytest.raises(ValueError):
        abstention_precision_recall(attempts)


# --- latency / tokens / cost -------------------------------------------------


def test_latency_summary_reports_p50_p95_and_count() -> None:
    summary = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary.count == 4
    assert summary.p50_ms == pytest.approx(25.0)


def test_latency_summary_undefined_for_empty_values() -> None:
    with pytest.raises(ValueError):
        latency_summary([])


def test_token_summary_totals_and_means() -> None:
    attempts = [
        _attempt(input_tokens=100, output_tokens=50),
        _attempt(input_tokens=200, output_tokens=100),
        _attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None, input_tokens=None, output_tokens=None),
    ]
    summary = token_summary(attempts)
    assert summary.count == 2
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 150
    assert summary.mean_input_tokens == pytest.approx(150.0)


def test_token_summary_undefined_with_no_measured_attempts() -> None:
    attempts = [_attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None, input_tokens=None, output_tokens=None)]
    with pytest.raises(ValueError):
        token_summary(attempts)


def test_cost_summary_computes_from_measured_tokens_and_pricing() -> None:
    table = {"test-model": {"input": 1.0, "output": 2.0}}
    attempts = [_attempt(input_tokens=1_000_000, output_tokens=1_000_000)]
    summary = cost_summary(attempts, pricing_table=table)
    assert summary.mean_cost_usd == pytest.approx(3.0)
    assert summary.count == 1


def test_cost_summary_raises_pricing_error_for_unconfigured_model() -> None:
    attempts = [_attempt(model_name="unpriced-model")]
    with pytest.raises(PricingError):
        cost_summary(attempts, pricing_table={})


def test_cost_summary_undefined_with_no_measured_attempts() -> None:
    attempts = [_attempt(outcome=AttemptOutcome.PRE_LLM_ABSTAIN, abstained=True, model_name=None, input_tokens=None, output_tokens=None)]
    with pytest.raises(ValueError):
        cost_summary(attempts, pricing_table={})
