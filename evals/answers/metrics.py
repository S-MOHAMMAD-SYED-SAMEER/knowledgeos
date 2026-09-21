"""Answer-quality metrics (§13): grounded-answer rate, unsupported-claim
rate, citation validity, citation coverage, abstention precision and
recall, plus the latency/token/cost aggregation §14 asks for.

Pure functions over `AnswerAttempt` records — no provider call, no
database. `evals/answers/suite.py` is what produces one `AnswerAttempt`
per question by actually calling `app.generation.generator.generate_answer`
(reused unmodified from milestone 8, per locked decision D6); this module
only aggregates what that produced.

**Rejected answers count against the metrics they would have satisfied.**
An attempt whose citations were rejected (`CitationError`) or whose model
output was malformed (`GenerationError`) is still one row in every
denominator below — it is not quietly excluded because it failed. A suite
that only ever averaged over the answers that happened to succeed could
report a perfect grounded-answer rate by silently dropping every failure,
which is exactly the kind of number the specification's "no fabricated
metrics" rule (§18) exists to prevent.

**Invalid-citation rate is `CitationError` specifically** (locked
decision D6, reusing `app.generation.citations` unmodified) — not folded
together with a malformed-JSON parse failure (`GenerationError`), which
the specification never calls a citation problem. Both are still counted
and reported (`AttemptOutcome`), just under separate names.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.observability.pricing import PricingError, compute_cost_usd
from app.observability.timing import p50, p95
from evals.answers.semantic import SemanticGroundingResult


class AttemptOutcome(StrEnum):
    """What happened when this question was run through generation."""

    PRE_LLM_ABSTAIN = "pre_llm_abstain"
    SUCCESS = "success"
    CITATION_INVALID = "citation_invalid"
    MALFORMED_OUTPUT = "malformed_output"


@dataclass(frozen=True)
class AnswerAttempt:
    """One question's outcome from `evals/answers/suite.py`.

    Every field past `outcome` may be `None` when it does not apply —
    citation coverage means nothing for a rejected attempt, and stage
    timings mean nothing when the model was never called.
    """

    question_id: str
    expected_abstain: bool
    outcome: AttemptOutcome
    abstained: bool | None
    citation_valid: bool | None
    citation_coverage: float | None
    grounded: bool | None
    semantic: SemanticGroundingResult | None
    model_name: str | None
    input_tokens: int | None
    output_tokens: int | None
    retrieval_ms: float | None
    rerank_ms: float | None
    llm_ms: float | None
    total_ms: float | None


def invalid_citation_rate(attempts: list[AnswerAttempt]) -> float:
    """`CitationError` attempts over every attempt that reached citation
    validation at all — a pre-LLM abstention never had citations to
    validate and is excluded from this denominator, the same way
    milestone 7's retrieval metrics exclude a question with no expected
    evidence rather than counting it as a miss.

    Raises `ValueError` if no attempt reached validation (denominator 0),
    the same "undefined, not zero" discipline
    `evals/retrieval/metrics.py::recall_at_k` already applies.
    """
    validated = [a for a in attempts if a.outcome != AttemptOutcome.PRE_LLM_ABSTAIN]
    if not validated:
        raise ValueError("invalid-citation rate is undefined with no validated attempts")
    invalid = sum(1 for a in validated if a.outcome == AttemptOutcome.CITATION_INVALID)
    return invalid / len(validated)


def citation_coverage_mean(attempts: list[AnswerAttempt]) -> float:
    """Mean citation coverage over every successful, non-abstained
    attempt. Structurally always `1.0` in this system: `generate_answer`
    raises `CitationError` for any answer below full coverage, so a
    `SUCCESS` outcome already guarantees it — reported anyway, as
    real, measured data rather than assumed, the same defense-in-depth
    `app/generation/grounding.py` already applies to its own
    already-guaranteed checks.
    """
    covered = [
        a.citation_coverage
        for a in attempts
        if a.outcome == AttemptOutcome.SUCCESS and a.abstained is False
    ]
    if not covered:
        raise ValueError("citation coverage is undefined with no successful non-abstained attempts")
    return sum(covered) / len(covered)


def grounded_answer_rate(attempts: list[AnswerAttempt]) -> float:
    """Fraction of all attempts that reached a grounded, valid answer.

    The denominator is every attempt whose question expected an actual
    answer to be possible to ground at all — i.e. every attempt, since
    even a rejected or malformed output was still a real attempt to
    answer. A pre-LLM abstention is excluded: there was no answer for
    "grounded" to describe.
    """
    countable = [a for a in attempts if a.outcome != AttemptOutcome.PRE_LLM_ABSTAIN]
    if not countable:
        raise ValueError("grounded-answer rate is undefined with no countable attempts")
    grounded = sum(1 for a in countable if a.grounded is True)
    return grounded / len(countable)


def unsupported_claim_rate(attempts: list[AnswerAttempt]) -> float | None:
    """Mean per-attempt unsupported-claim rate, or `None` if it was never
    computed for any attempt — which is always true while
    `evals.answers.semantic.DEFAULT_UNSUPPORTED_CLAIM_THRESHOLD` is `None`
    (see that module's docstring: no threshold is invented). Never `0.0`
    standing in for "not measured."
    """
    rates = [
        a.semantic.unsupported_claim_rate
        for a in attempts
        if a.semantic is not None and a.semantic.unsupported_claim_rate is not None
    ]
    if not rates:
        return None
    return sum(rates) / len(rates)


@dataclass(frozen=True)
class AbstentionScore:
    precision: float | None
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    positive_count: int


def abstention_precision_recall(attempts: list[AnswerAttempt]) -> AbstentionScore:
    """Precision and recall of the abstention decision against
    `expected_abstain`. A rejected or malformed attempt never
    successfully communicated an abstention and is scored as a predicted
    non-abstention — the model tried to answer and failed, which is not
    the same thing as correctly recognizing insufficient evidence.

    `precision` is `None` when nothing was ever predicted to abstain
    (0/0) — recall is still defined and reported, since the positive
    class (`expected_abstain`) is known regardless of what was
    predicted.
    """
    # Every PRE_LLM_ABSTAIN and SUCCESS attempt has a real `abstained`
    # value; every CITATION_INVALID/MALFORMED_OUTPUT attempt has
    # `abstained=None` and is scored as a predicted non-abstention (see
    # the docstring).
    predicted_abstain = [a.abstained is True for a in attempts]

    true_positives = sum(
        1 for a, pred in zip(attempts, predicted_abstain, strict=True) if pred and a.expected_abstain
    )
    false_positives = sum(
        1 for a, pred in zip(attempts, predicted_abstain, strict=True) if pred and not a.expected_abstain
    )
    false_negatives = sum(
        1 for a, pred in zip(attempts, predicted_abstain, strict=True) if not pred and a.expected_abstain
    )
    positive_count = sum(1 for a in attempts if a.expected_abstain)

    predicted_positive_count = true_positives + false_positives
    precision = (
        true_positives / predicted_positive_count if predicted_positive_count > 0 else None
    )
    if positive_count == 0:
        raise ValueError("abstention recall is undefined with zero expected-abstain questions")
    recall = true_positives / positive_count

    return AbstentionScore(
        precision=precision,
        recall=recall,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        positive_count=positive_count,
    )


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float
    p95_ms: float
    count: int


def latency_summary(values: list[float]) -> LatencySummary:
    """p50/p95 over one stage's measured milliseconds. Raises `ValueError`
    for an empty list — undefined, not zero."""
    if not values:
        raise ValueError("latency summary is undefined with no measured values")
    return LatencySummary(p50_ms=p50(values), p95_ms=p95(values), count=len(values))


@dataclass(frozen=True)
class TokenSummary:
    mean_input_tokens: float
    mean_output_tokens: float
    total_input_tokens: int
    total_output_tokens: int
    count: int


def token_summary(attempts: list[AnswerAttempt]) -> TokenSummary:
    """Mean and total tokens over attempts that actually called the
    model — a pre-LLM abstention used none."""
    measured = [a for a in attempts if a.input_tokens is not None and a.output_tokens is not None]
    if not measured:
        raise ValueError("token summary is undefined with no measured attempts")
    total_in = sum(a.input_tokens for a in measured)
    total_out = sum(a.output_tokens for a in measured)
    return TokenSummary(
        mean_input_tokens=total_in / len(measured),
        mean_output_tokens=total_out / len(measured),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        count=len(measured),
    )


@dataclass(frozen=True)
class CostSummary:
    mean_cost_usd: float
    total_cost_usd: float
    count: int


def cost_summary(
    attempts: list[AnswerAttempt], *, pricing_table: dict[str, dict[str, float]]
) -> CostSummary:
    """Cost per query, from measured tokens and configured pricing only.

    Raises `PricingError` (re-raised, not caught) the moment any priced
    attempt's model has no configured entry — official cost reporting
    must be all-or-nothing, never an average that silently skipped the
    requests it could not price.
    """
    measured = [a for a in attempts if a.input_tokens is not None and a.output_tokens is not None]
    if not measured:
        raise ValueError("cost summary is undefined with no measured attempts")

    costs = [
        compute_cost_usd(
            model=a.model_name,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            table=pricing_table,
        )
        for a in measured
    ]
    return CostSummary(
        mean_cost_usd=sum(costs) / len(costs),
        total_cost_usd=sum(costs),
        count=len(costs),
    )


__all__ = [
    "AbstentionScore",
    "AnswerAttempt",
    "AttemptOutcome",
    "CostSummary",
    "LatencySummary",
    "PricingError",
    "TokenSummary",
    "abstention_precision_recall",
    "citation_coverage_mean",
    "cost_summary",
    "grounded_answer_rate",
    "invalid_citation_rate",
    "latency_summary",
    "token_summary",
    "unsupported_claim_rate",
]
