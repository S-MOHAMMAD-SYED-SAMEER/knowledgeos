"""Whether to abstain — before the model is ever called, or after it
responds.

The specification (§9): abstention is "triggered when top rerank score
falls below a threshold, or when the model sets `sufficient_evidence=false`."
Zero retrieved candidates is not named separately, but is the same
"nothing relevant was found" condition the threshold trigger exists to
catch, at its most extreme, and is handled here as a third, pre-LLM case —
locked project decision: zero candidates must never reach the model at
all.

Pure: no provider call, no I/O, no database. Both functions take
already-computed values and return a decision.
"""

from dataclasses import dataclass

# Locked project decision (D6): never invented. `None` means the
# rerank-score trigger is inactive — no threshold has been genuinely
# calibrated against a dev split in this build, and the specification is
# explicit that the number must come from calibration, "not chosen by
# taste." Only a documented calibration run may change this default.
DEFAULT_ABSTENTION_RERANK_THRESHOLD: float | None = None

# The deterministic text used when a query abstains without ever reaching
# the model — there is no model output to abstain *with*. Fixed and
# documented as a constant, not written to imply the model produced it.
DEFAULT_ABSTENTION_TEXT = (
    "There is insufficient evidence in the retrieved documents to answer "
    "this question."
)

REASON_NO_CANDIDATES = "no_candidates"
REASON_RERANK_SCORE_BELOW_THRESHOLD = "rerank_score_below_threshold"
REASON_SUFFICIENT_EVIDENCE_FALSE = "sufficient_evidence_false"


@dataclass(frozen=True)
class AbstentionDecision:
    abstain: bool
    reason: str | None


def pre_llm_abstention(
    *,
    candidate_count: int,
    top_rerank_score: float | None,
    threshold: float | None,
) -> AbstentionDecision:
    """Whether to abstain WITHOUT ever calling the model.

    `threshold=None` disables the score-based path entirely — only the
    zero-candidate case can trigger here, which is the one abstention
    condition this project has always been able to determine without any
    provider at all.
    """
    if candidate_count == 0:
        return AbstentionDecision(abstain=True, reason=REASON_NO_CANDIDATES)
    if (
        threshold is not None
        and top_rerank_score is not None
        and top_rerank_score < threshold
    ):
        return AbstentionDecision(
            abstain=True, reason=REASON_RERANK_SCORE_BELOW_THRESHOLD
        )
    return AbstentionDecision(abstain=False, reason=None)


def post_llm_abstention(*, sufficient_evidence: bool) -> AbstentionDecision:
    """Whether to abstain based on what the model itself reported."""
    if not sufficient_evidence:
        return AbstentionDecision(abstain=True, reason=REASON_SUFFICIENT_EVIDENCE_FALSE)
    return AbstentionDecision(abstain=False, reason=None)


__all__ = [
    "DEFAULT_ABSTENTION_RERANK_THRESHOLD",
    "DEFAULT_ABSTENTION_TEXT",
    "REASON_NO_CANDIDATES",
    "REASON_RERANK_SCORE_BELOW_THRESHOLD",
    "REASON_SUFFICIENT_EVIDENCE_FALSE",
    "AbstentionDecision",
    "post_llm_abstention",
    "pre_llm_abstention",
]
