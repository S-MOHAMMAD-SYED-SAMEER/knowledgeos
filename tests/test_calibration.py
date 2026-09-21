"""`evals/answers/calibration.py`: the pure threshold sweep is
deterministic and reproducible (D3), and `collect_dev_observations`'
wiring against the real retrieve/rerank pipeline is exercised end to end
with fake providers (D2's pytest-only allowance) to prove the plumbing,
not to produce an official calibrated threshold.
"""

import pytest

from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider
from app.storage import LocalStorage
from evals.answers.calibration import (
    CalibrationResult,
    DevObservation,
    calibrate_threshold,
    collect_dev_observations,
)
from evals.retrieval.corpus import ensure_corpus_seeded
from evals.retrieval.questions import load_question_set


def test_calibrate_threshold_is_deterministic_across_repeated_calls() -> None:
    observations = [
        DevObservation(question_id="a", expected_abstain=True, top_rerank_score=1.0),
        DevObservation(question_id="b", expected_abstain=False, top_rerank_score=5.0),
        DevObservation(question_id="c", expected_abstain=True, top_rerank_score=3.0),
        DevObservation(question_id="d", expected_abstain=False, top_rerank_score=8.0),
    ]
    first = calibrate_threshold(observations)
    second = calibrate_threshold(list(reversed(observations)))
    assert first == second


def test_calibrate_threshold_maximizes_dev_recall() -> None:
    # A threshold at or below 1.0 abstains on nothing; a threshold above
    # both positive scores (1.0, 3.0) but at or below the lowest negative
    # score (5.0) abstains on exactly the two positives and nothing else --
    # perfect recall and precision.
    observations = [
        DevObservation(question_id="a", expected_abstain=True, top_rerank_score=1.0),
        DevObservation(question_id="b", expected_abstain=True, top_rerank_score=3.0),
        DevObservation(question_id="c", expected_abstain=False, top_rerank_score=5.0),
        DevObservation(question_id="d", expected_abstain=False, top_rerank_score=8.0),
    ]
    result = calibrate_threshold(observations)
    assert isinstance(result, CalibrationResult)
    assert result.dev_recall == 1.0
    assert result.dev_precision == 1.0
    assert result.dev_positive_count == 2


def test_calibrate_threshold_treats_missing_score_as_always_abstaining() -> None:
    observations = [
        DevObservation(question_id="a", expected_abstain=True, top_rerank_score=None),
        DevObservation(question_id="b", expected_abstain=False, top_rerank_score=5.0),
    ]
    result = calibrate_threshold(observations)
    # The None-scored positive abstains at every threshold, so recall is
    # always at least 0.5 regardless of the swept threshold.
    assert result.dev_recall >= 0.5


def test_calibrate_threshold_raises_with_zero_expected_abstain_questions() -> None:
    observations = [DevObservation(question_id="a", expected_abstain=False, top_rerank_score=5.0)]
    with pytest.raises(ValueError):
        calibrate_threshold(observations)


def test_calibrate_threshold_ties_broken_by_lowest_threshold() -> None:
    # Two candidate thresholds (2.0 and 4.0) both abstain on the single
    # positive (score 1.0) and neither on the single negative (score 5.0)
    # -- identical recall and precision, so the lower threshold must win.
    observations = [
        DevObservation(question_id="a", expected_abstain=True, top_rerank_score=1.0),
        DevObservation(question_id="b", expected_abstain=False, top_rerank_score=5.0),
        DevObservation(question_id="c", expected_abstain=False, top_rerank_score=2.0),
        DevObservation(question_id="d", expected_abstain=False, top_rerank_score=4.0),
    ]
    result = calibrate_threshold(observations)
    assert result.threshold == 2.0


# --- collect_dev_observations wiring, against fake providers ----------------


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


def test_collect_dev_observations_returns_one_observation_per_dev_question(session, storage) -> None:
    embeddings = FakeEmbeddingProvider()
    ensure_corpus_seeded(session, storage, embeddings)
    questions = load_question_set()
    dev_questions = questions[:3]

    observations = collect_dev_observations(session, embeddings, FakeRerankProvider(), dev_questions)

    assert len(observations) == 3
    assert [o.question_id for o in observations] == [q.id for q in dev_questions]
    assert [o.expected_abstain for o in observations] == [q.expected_abstain for q in dev_questions]


def test_collect_dev_observations_feeds_calibrate_threshold_without_error(session, storage) -> None:
    embeddings = FakeEmbeddingProvider()
    ensure_corpus_seeded(session, storage, embeddings)
    questions = load_question_set()
    dev_questions = [q for q in questions if q.expected_abstain][:1] + [
        q for q in questions if not q.expected_abstain
    ][:3]

    observations = collect_dev_observations(session, embeddings, FakeRerankProvider(), dev_questions)
    result = calibrate_threshold(observations)

    assert isinstance(result, CalibrationResult)
    assert 0.0 <= result.dev_recall <= 1.0
