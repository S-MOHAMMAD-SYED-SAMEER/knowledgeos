"""Abstention-threshold calibration, against the frozen dev split (§9,
locked decisions D3/D4).

Two separable steps, kept in separate functions so the sweep itself is
testable with synthetic numbers and needs no database or provider at all:

1. `collect_dev_observations` — for every dev-split question, run the
   real retrieval + reranking pipeline (`app.retrieval`, `app.reranking`,
   both reused unmodified) and record the top rerank score, exactly the
   number `app.generation.abstention.pre_llm_abstention` compares against
   a threshold. **Needs real models** — a fake reranker's scores are as
   meaningless for calibration as milestone 7's own correction 2 already
   says fake embedding vectors are for retrieval metrics.
2. `calibrate_threshold` — a pure sweep over already-collected
   observations, choosing the threshold that maximizes recall on the dev
   split (the specification's own gate is a recall target), tie-broken by
   precision, tie-broken by the lowest surviving threshold for
   determinism. **This objective is a project-level decision** — the
   specification states only "calibrated on a dev split… not chosen by
   taste," and gives no formula. Maximizing recall directly targets what
   §13 measures; nothing here was tuned by eye against a desired outcome.

**Never invented.** If real models are unavailable, nothing calls
`collect_dev_observations` at all (see `evals/answers/suite.py`), and the
calibrated threshold stays `app.generation.abstention.DEFAULT_ABSTENTION_RERANK_THRESHOLD`
(`None`) — documented as uncalibrated, never filled with a guess to make
a gate read green.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.parsing import normalize
from app.providers.embeddings import EmbeddingProvider
from app.providers.reranker import RerankProvider
from app.reranking.pipeline import rerank
from app.retrieval.fusion import DEFAULT_RRF_K
from app.retrieval.pipeline import retrieve
from evals.retrieval.questions import Question

# The statistical caveat this project will not paper over — see
# `evals/fixtures/splits.yaml`'s own header for the exact counts.
SAMPLE_SIZE_LIMITATION = (
    "the dev split has too few expected_abstain=true questions (see "
    "evals/fixtures/splits.yaml) for the resulting recall figure to carry "
    "real statistical precision; treat any calibrated threshold and its "
    "reported dev recall as evidence from a small sample, not proof of "
    "generalization"
)


@dataclass(frozen=True)
class DevObservation:
    """One dev-split question's real top rerank score, or `None` if
    nothing was retrieved at all (which always abstains regardless of
    threshold — see `app.generation.abstention.pre_llm_abstention`)."""

    question_id: str
    expected_abstain: bool
    top_rerank_score: float | None


@dataclass(frozen=True)
class CalibrationResult:
    """The outcome of one calibration sweep.

    `threshold` is `None` whenever calibration could not or did not run
    with real models — never a fallback number.
    """

    threshold: float | None
    dev_recall: float | None
    dev_precision: float | None
    dev_positive_count: int
    swept_threshold_count: int
    limitation: str


def collect_dev_observations(
    session: Session,
    embeddings: EmbeddingProvider,
    reranker_provider: RerankProvider,
    dev_questions: list[Question],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[DevObservation]:
    """Run the real retrieve → rerank pipeline for every dev-split
    question and record its top rerank score. Reuses
    `app.retrieval.pipeline.retrieve` and `app.reranking.pipeline.rerank`
    unmodified — the same functions `evals/retrieval/suite.py` already
    calls for the retrieval suite, never a second implementation of
    either stage.
    """
    observations: list[DevObservation] = []
    for question in dev_questions:
        normalized_text = normalize(question.text)
        query_vector = embeddings.embed([normalized_text])[0]
        result = retrieve(
            session,
            normalized_text=normalized_text,
            query_vector=query_vector,
            filters=question.filters,
            rrf_k=rrf_k,
        )
        reranked = rerank(normalized_text, result.candidates, reranker_provider)
        top_score = reranked[0].rerank_score if reranked else None
        observations.append(
            DevObservation(
                question_id=question.id,
                expected_abstain=question.expected_abstain,
                top_rerank_score=top_score,
            )
        )
    return observations


def calibrate_threshold(observations: list[DevObservation]) -> CalibrationResult:
    """Sweep every observed top score as a candidate threshold and pick
    the one maximizing dev-split abstention recall (ties broken by
    precision, then by the lowest surviving threshold).

    A question with `top_rerank_score is None` (zero candidates) always
    abstains, at every threshold — it never enters the sweep as a
    candidate cutoff, but it does count in every threshold's recall and
    precision.

    Raises `ValueError` if the dev split has zero `expected_abstain`
    questions — recall against zero positives is undefined, not zero.
    """
    positive_count = sum(1 for obs in observations if obs.expected_abstain)
    if positive_count == 0:
        raise ValueError(
            "calibration is undefined: the dev split has zero "
            "expected_abstain=true questions"
        )

    candidate_thresholds = sorted(
        {obs.top_rerank_score for obs in observations if obs.top_rerank_score is not None}
    )
    if not candidate_thresholds:
        # Every dev question had zero candidates -- every threshold behaves
        # identically (everything abstains). Report the trivial case
        # explicitly rather than sweeping over nothing.
        candidate_thresholds = [0.0]

    best: tuple[float, float, float] | None = None  # (recall, precision, -threshold)
    best_threshold = candidate_thresholds[0]
    best_recall = -1.0
    best_precision = -1.0

    for threshold in candidate_thresholds:
        true_positives = false_positives = 0
        for obs in observations:
            predicted_abstain = (
                obs.top_rerank_score is None or obs.top_rerank_score < threshold
            )
            if predicted_abstain and obs.expected_abstain:
                true_positives += 1
            elif predicted_abstain and not obs.expected_abstain:
                false_positives += 1

        recall = true_positives / positive_count
        predicted_positive = true_positives + false_positives
        precision = true_positives / predicted_positive if predicted_positive > 0 else 0.0

        is_better = (
            recall > best_recall
            or (recall == best_recall and precision > best_precision)
        )
        if is_better:
            best_recall, best_precision, best_threshold = recall, precision, threshold

    return CalibrationResult(
        threshold=best_threshold,
        dev_recall=best_recall,
        dev_precision=best_precision,
        dev_positive_count=positive_count,
        swept_threshold_count=len(candidate_thresholds),
        limitation=SAMPLE_SIZE_LIMITATION,
    )


__all__ = [
    "SAMPLE_SIZE_LIMITATION",
    "CalibrationResult",
    "DevObservation",
    "calibrate_threshold",
    "collect_dev_observations",
]
