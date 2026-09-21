"""`evals/answers/suite.py` end to end, against a real database and the
real fixture corpus, but through fake providers (D1/D2's pytest-only
allowance): what these tests prove is the suite's wiring -- every
question produces exactly one attempt, real generation/citation/
abstention/persistence happens through milestone 8's own unmodified
functions, timing and semantic scoring are threaded through correctly.
None of the numbers these tests produce are an official M9 evaluation
result; only `evals/run.py`, run against the real local models and
Gemini, could produce that.
"""

import pytest
from sqlalchemy import select

from app.models import Query
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider
from app.storage import LocalStorage
from evals.answers.metrics import AnswerAttempt, AttemptOutcome
from evals.answers.suite import AnswerSuiteResult, run_answers_suite
from evals.retrieval.questions import load_question_set
from tests.generation_fixtures import AutoCitingLLMProvider


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


def test_the_suite_runs_every_frozen_question_exactly_once(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )

    assert isinstance(result, AnswerSuiteResult)
    questions = load_question_set()
    assert result.total_question_count == len(questions)
    assert len(result.attempts) == len(questions)
    assert {a.question_id for a in result.attempts} == {q.id for q in questions}


def test_dev_and_test_counts_match_the_frozen_split(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )
    assert result.dev_question_count + result.test_question_count == result.total_question_count
    assert result.dev_question_count > 0
    assert result.test_question_count > 0


def test_every_attempt_is_a_recognized_outcome(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )
    for attempt in result.attempts:
        assert isinstance(attempt, AnswerAttempt)
        assert attempt.outcome in AttemptOutcome


def test_every_attempt_has_measured_retrieval_and_rerank_timing(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )
    for attempt in result.attempts:
        assert attempt.retrieval_ms is not None
        assert attempt.retrieval_ms >= 0
        assert attempt.rerank_ms is not None
        assert attempt.total_ms is not None
        assert attempt.total_ms >= attempt.retrieval_ms + attempt.rerank_ms


def test_a_successful_attempt_is_persisted_and_readable_back(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )
    successful = [a for a in result.attempts if a.outcome == AttemptOutcome.SUCCESS]
    assert successful, "expected at least one successful attempt against the fixture corpus"
    assert len(result.persisted_query_ids) >= 1

    rows = session.execute(select(Query).where(Query.id.in_(result.persisted_query_ids))).scalars().all()
    assert len(rows) == len(result.persisted_query_ids)


def test_rejected_attempts_are_not_persisted(session, storage, embeddings) -> None:
    from app.providers.fake_llm import FakeLLMProvider

    # Every call returns malformed JSON, forcing GenerationError for every
    # question that reaches the model at all.
    questions = load_question_set()
    llm = FakeLLMProvider(["not valid json"] * len(questions))

    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        llm,
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )

    malformed = [a for a in result.attempts if a.outcome == AttemptOutcome.MALFORMED_OUTPUT]
    assert malformed
    for attempt in malformed:
        assert attempt.retrieval_ms is not None
        assert attempt.rerank_ms is not None
        assert attempt.llm_ms is None
    rows = session.execute(select(Query)).scalars().all()
    assert len(rows) == len(result.persisted_query_ids)
    assert len(rows) < len(questions)


def test_semantic_scorer_none_means_no_attempt_is_scored(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
    )
    assert all(a.semantic is None for a in result.attempts)


def test_semantic_scorer_configured_scores_non_abstained_successes(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=FakeRerankProvider(),
        abstention_threshold=None,
        max_tokens=512,
    )
    non_abstained_success = [
        a for a in result.attempts if a.outcome == AttemptOutcome.SUCCESS and a.abstained is False
    ]
    assert non_abstained_success
    for attempt in non_abstained_success:
        assert attempt.semantic is not None
        assert attempt.semantic.status in ("scored", "unavailable")


def test_config_records_the_model_names_and_abstention_threshold(session, storage, embeddings) -> None:
    result = run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        AutoCitingLLMProvider(),
        semantic_scorer=None,
        abstention_threshold=7.5,
        max_tokens=512,
    )
    assert result.config["embedding_model"] == embeddings.model_name
    assert result.config["rerank_model"] == FakeRerankProvider().model_name
    assert result.config["llm_model"] == AutoCitingLLMProvider().model_name
    assert result.config["abstention_threshold"] == 7.5
    assert result.config["semantic_scorer_model"] is None
