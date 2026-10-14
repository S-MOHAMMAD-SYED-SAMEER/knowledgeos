"""Pacing, call budget, checkpoints, and stopping and resuming an answers run.

No network and no waiting: pacing is tested with a fake clock, and the provider failures are
scripted. The suite tests use the real database and the real fixture corpus through the same fake
providers `tests/test_answers_suite.py` uses.
"""

import json
from uuid import uuid4

import pytest
from sqlalchemy import select

import evals.run as run_module
from app.models import EvalRun, Query
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider
from app.providers.llm import LLMError, LLMRateLimited
from app.storage import LocalStorage
from evals.answers.checkpoint import (
    CheckpointError,
    fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from evals.answers.metrics import AnswerAttempt, AttemptOutcome
from evals.answers.pacing import PacedLLMProvider, describe_call_budget
from evals.answers.semantic import SemanticGroundingResult, SentenceEntailment
from evals.answers.suite import run_answers_suite
from evals.retrieval.questions import load_question_set
from tests.generation_fixtures import AutoCitingLLMProvider


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# --- pacing ------------------------------------------------------------------------------


class FakeTime:
    """A clock that only moves when somebody sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _paced(rpm: float, time: FakeTime) -> PacedLLMProvider:
    return PacedLLMProvider(AutoCitingLLMProvider(), requests_per_minute=rpm, clock=time.clock, sleep=time.sleep)


def _call(provider) -> None:
    provider.complete(system="s", user='<evidence chunk_uid="' + "a" * 32 + '">x</evidence>', max_tokens=10)


def test_the_first_call_is_not_delayed_and_later_ones_are_spaced() -> None:
    time = FakeTime()
    paced = _paced(10, time)  # one every 6 seconds

    for _ in range(3):
        _call(paced)

    assert time.slept == [6.0, 6.0]
    assert paced.calls == 3


def test_time_a_call_already_took_counts_towards_the_gap() -> None:
    time = FakeTime()
    paced = _paced(10, time)

    _call(paced)
    time.now += 4.0  # the model took four seconds, or the run did other work
    _call(paced)

    assert time.slept == [pytest.approx(2.0)]


def test_a_call_after_a_long_pause_is_not_delayed_at_all() -> None:
    time = FakeTime()
    paced = _paced(10, time)

    _call(paced)
    time.now += 60.0
    _call(paced)

    assert time.slept == []


def test_pacing_passes_the_request_through_unchanged() -> None:
    seen: dict = {}

    class Recorder:
        model_name = "recorder"

        def complete(self, *, system, user, max_tokens, json_schema=None):
            seen.update(system=system, user=user, max_tokens=max_tokens, json_schema=json_schema)
            return AutoCitingLLMProvider().complete(system=system, user=user, max_tokens=max_tokens)

    PacedLLMProvider(Recorder(), requests_per_minute=600).complete(
        system="sys", user="usr", max_tokens=7, json_schema={"type": "object"}
    )

    assert seen == {"system": "sys", "user": "usr", "max_tokens": 7, "json_schema": {"type": "object"}}
    assert PacedLLMProvider(Recorder(), requests_per_minute=600).model_name == "recorder"


@pytest.mark.parametrize("bad", [0, -1])
def test_a_rate_that_is_not_positive_is_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        PacedLLMProvider(AutoCitingLLMProvider(), requests_per_minute=bad)


def test_the_call_budget_says_calls_worst_case_and_minimum_time() -> None:
    text = describe_call_budget(question_count=52, already_done=0, requests_per_minute=10)

    assert "about 53 model calls" in text
    assert "1 availability check + 52 question(s)" in text
    assert "up to 105" in text
    assert "10 a minute" in text
    assert "at least 5.3 minutes" in text


def test_the_call_budget_for_a_resumed_run_counts_only_what_is_left() -> None:
    text = describe_call_budget(question_count=52, already_done=40, requests_per_minute=10)

    assert "about 13 model calls" in text
    assert "40 already done" in text


# --- checkpoints ---------------------------------------------------------------------------


def _attempt(question_id: str, *, semantic=None, outcome=AttemptOutcome.SUCCESS) -> AnswerAttempt:
    return AnswerAttempt(
        question_id=question_id,
        expected_abstain=False,
        outcome=outcome,
        abstained=False,
        citation_valid=True,
        citation_coverage=1.0,
        grounded=True,
        semantic=semantic,
        model_name="m",
        input_tokens=100,
        output_tokens=20,
        retrieval_ms=1.0,
        rerank_ms=2.0,
        llm_ms=3.0,
        total_ms=6.0,
        attempts_used=2,
    )


FINGERPRINT = fingerprint(
    llm_model="m", embedding_model="e", rerank_model="r", abstention_threshold=0.25, question_ids=["q1", "q2"]
)


def test_a_checkpoint_round_trips_every_field(tmp_path) -> None:
    semantic = SemanticGroundingResult(
        status="scored",
        reason=None,
        sentence_scores=[
            SentenceEntailment(sentence="A fact.", cited_chunk_uids=frozenset({"b", "a"}), max_score=0.5, supported=None)
        ],
        unsupported_claim_rate=None,
    )
    attempts = [_attempt("q1", semantic=semantic), _attempt("q2", outcome=AttemptOutcome.CITATION_INVALID)]
    ids = [uuid4(), uuid4()]
    path = tmp_path / "answers.partial.json"

    save_checkpoint(path, fingerprint=FINGERPRINT, attempts=attempts, persisted_query_ids=ids, stopped_early="why")
    loaded = load_checkpoint(path, expected_fingerprint=FINGERPRINT)

    assert loaded.attempts == attempts
    assert loaded.persisted_query_ids == ids
    assert loaded.stopped_early == "why"


def test_saving_leaves_no_temporary_file_behind(tmp_path) -> None:
    path = tmp_path / "sub" / "answers.partial.json"

    save_checkpoint(path, fingerprint=FINGERPRINT, attempts=[], persisted_query_ids=[])

    assert [p.name for p in path.parent.iterdir()] == ["answers.partial.json"]


def test_a_checkpoint_from_a_different_configuration_is_refused(tmp_path) -> None:
    path = tmp_path / "c.json"
    save_checkpoint(path, fingerprint=FINGERPRINT, attempts=[_attempt("q1")], persisted_query_ids=[uuid4()])
    other = fingerprint(
        llm_model="another-model", embedding_model="e", rerank_model="r", abstention_threshold=0.25, question_ids=["q1", "q2"]
    )

    with pytest.raises(CheckpointError, match="different configuration.*llm_model"):
        load_checkpoint(path, expected_fingerprint=other)


def test_a_checkpoint_for_a_different_question_set_is_refused(tmp_path) -> None:
    path = tmp_path / "c.json"
    save_checkpoint(path, fingerprint=FINGERPRINT, attempts=[], persisted_query_ids=[])
    other = fingerprint(
        llm_model="m", embedding_model="e", rerank_model="r", abstention_threshold=0.25, question_ids=["q1", "q2", "q3"]
    )

    with pytest.raises(CheckpointError, match="question_ids"):
        load_checkpoint(path, expected_fingerprint=other)


@pytest.mark.parametrize("content", ["not json", "{}", '{"schema_version": 1, "fingerprint": {}}'])
def test_a_damaged_checkpoint_is_refused_without_a_traceback_leaking_content(tmp_path, content) -> None:
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CheckpointError):
        load_checkpoint(path, expected_fingerprint={})


def test_a_missing_checkpoint_is_refused(tmp_path) -> None:
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_checkpoint(tmp_path / "nope.json", expected_fingerprint={})


# --- the suite: stopping and resuming --------------------------------------------------------


class FlakyLLM:
    """The auto-citing double, except that chosen calls (counted from 1) raise instead."""

    model_name = "flaky-test-double"

    def __init__(self, failing_calls: set[int], error: type[LLMError] = LLMRateLimited) -> None:
        self._inner = AutoCitingLLMProvider()
        self._failing = failing_calls
        self._error = error
        self.calls = 0

    def complete(self, *, system, user, max_tokens, json_schema=None):
        self.calls += 1
        if self.calls in self._failing:
            raise self._error("simulated")
        return self._inner.complete(system=system, user=user, max_tokens=max_tokens)


def _run(session, storage, embeddings, llm, **kwargs):
    return run_answers_suite(
        session,
        storage,
        embeddings,
        FakeRerankProvider(),
        llm,
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
        **kwargs,
    )


def test_a_healthy_run_is_complete_with_nothing_pending(session, storage, embeddings) -> None:
    result = _run(session, storage, embeddings, AutoCitingLLMProvider())

    assert result.complete
    assert result.pending_question_ids == []
    assert result.stopped_early is None
    assert result.provider_errors == []


def test_it_stops_after_the_configured_number_of_failures_in_a_row(session, storage, embeddings) -> None:
    llm = FlakyLLM(failing_calls=set(range(4, 100)))  # three questions answered, then nothing works

    result = _run(session, storage, embeddings, llm, max_consecutive_failures=2)

    assert not result.complete
    assert len(result.attempts) == 3
    assert llm.calls == 5  # three answered, then two failures, then it stopped asking
    assert "2 provider failures in a row" in result.stopped_early
    assert "LLMRateLimited" in result.stopped_early
    assert [e["error"] for e in result.provider_errors] == ["LLMRateLimited", "LLMRateLimited"]
    assert len(result.pending_question_ids) == result.total_question_count - 3


def test_a_single_failure_is_survived_and_leaves_exactly_that_question_pending(session, storage, embeddings) -> None:
    llm = FlakyLLM(failing_calls={5})

    result = _run(session, storage, embeddings, llm, max_consecutive_failures=2)

    assert result.total_question_count - len(result.attempts) == 1  # every other question was answered
    assert len(result.pending_question_ids) == 1
    assert len(result.provider_errors) == 1
    assert not result.complete  # one hole makes the run incomplete, so it must not be reported
    assert "the provider failed on them" in result.stopped_early


def test_a_success_between_failures_resets_the_count(session, storage, embeddings) -> None:
    llm = FlakyLLM(failing_calls={3, 5, 7})  # never two in a row

    result = _run(session, storage, embeddings, llm, max_consecutive_failures=2)

    assert len(result.provider_errors) == 3
    assert len(result.attempts) == result.total_question_count - 3  # it kept going the whole way


def test_progress_is_reported_after_every_answered_question(session, storage, embeddings) -> None:
    seen: list[int] = []

    result = _run(
        session,
        storage,
        embeddings,
        AutoCitingLLMProvider(),
        on_progress=lambda attempts, ids: seen.append(len(attempts)),
    )

    assert seen == list(range(1, result.total_question_count + 1))


def test_a_stopped_run_resumes_and_matches_an_uninterrupted_one(session, storage, embeddings) -> None:
    first = _run(session, storage, embeddings, FlakyLLM(failing_calls=set(range(4, 100))), max_consecutive_failures=2)
    assert not first.complete
    queries_after_first = len(session.execute(select(Query)).scalars().all())

    resumed = _run(
        session,
        storage,
        embeddings,
        AutoCitingLLMProvider(),
        resume_from=(first.attempts, first.persisted_query_ids),
    )

    assert resumed.complete
    assert {a.question_id for a in resumed.attempts} == {q.id for q in load_question_set()}
    # The questions answered before the stop were not asked again, and nothing was persisted twice.
    assert resumed.attempts[:3] == first.attempts
    assert len(session.execute(select(Query)).scalars().all()) == queries_after_first + (len(resumed.attempts) - 3)
    assert len(set(resumed.persisted_query_ids)) == len(resumed.persisted_query_ids)

    # And the numbers are the numbers an uninterrupted run gives.
    uninterrupted = _run(session, storage, embeddings, AutoCitingLLMProvider())
    key = lambda a: (a.question_id, a.outcome, a.abstained, a.citation_valid, a.grounded)  # noqa: E731
    assert sorted(map(key, resumed.attempts)) == sorted(map(key, uninterrupted.attempts))


def test_resuming_asks_the_model_only_for_what_is_left(session, storage, embeddings) -> None:
    first = _run(session, storage, embeddings, FlakyLLM(failing_calls=set(range(4, 100))), max_consecutive_failures=2)
    llm = FlakyLLM(failing_calls=set())

    _run(session, storage, embeddings, llm, resume_from=(first.attempts, first.persisted_query_ids))

    assert llm.calls == len(first.pending_question_ids)


# --- the CLI ------------------------------------------------------------------------------------


class _StubSettings:
    llm_model = "stub-gemini-model"
    llm_max_output_tokens = 512
    llm_timeout_seconds = 30.0
    llm_max_retries = 2
    llm_retry_max_wait_seconds = 15.0
    llm_temperature = None
    llm_pricing_usd_per_million_tokens: dict = {}


def _patch(monkeypatch, tmp_path, llm_factory):
    from tests.test_answers_cli import _StubCrossEncoderProvider, _StubEmbeddingProvider

    class _Gemini:
        def __new__(cls, model_name, **kwargs):
            return llm_factory()

    monkeypatch.setattr(run_module, "BgeEmbeddingProvider", _StubEmbeddingProvider)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _StubCrossEncoderProvider)
    monkeypatch.setattr(run_module, "GeminiLLMProvider", _Gemini)
    monkeypatch.setattr(run_module, "get_settings", lambda: _StubSettings())
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")


def _unpaced(session, **kwargs):
    return run_module.run_answers(session, requests_per_minute=0, **kwargs)


def test_a_run_that_stops_early_writes_a_checkpoint_and_nothing_official(session, tmp_path, monkeypatch, capsys) -> None:
    # call 1 is the availability check; a few questions work; then the provider is gone
    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set(range(5, 500))))

    exit_code = _unpaced(session)

    assert exit_code == 2
    assert session.execute(select(EvalRun)).scalars().all() == []  # a partial run is not a result
    assert list((tmp_path / "eval_runs").glob("answers-*.json")) == [] or all(
        p.name.endswith(".partial.json") for p in (tmp_path / "eval_runs").glob("answers-*.json")
    )
    checkpoints = list((tmp_path / "eval_runs").glob("answers-*.partial.json"))
    assert len(checkpoints) == 1
    saved = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert 0 < len(saved["attempts"]) < 52  # what was done is kept; the rest is pending
    assert "provider failures in a row" in saved["stopped_early"]
    err = capsys.readouterr().err
    assert "stopped early" in err
    assert "No eval_runs row was written" in err
    assert f"--resume {checkpoints[0]}" in err


def test_resuming_finishes_the_run_reports_it_and_removes_the_checkpoint(session, tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set(range(5, 500))))
    assert _unpaced(session) == 2
    checkpoint = next((tmp_path / "eval_runs").glob("answers-*.partial.json"))

    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set()))
    exit_code = _unpaced(session, resume=checkpoint)

    assert exit_code == 0
    rows = session.execute(select(EvalRun)).scalars().all()
    assert len(rows) == 1
    assert rows[0].metrics["outcome_counts"]["success"] + rows[0].metrics["outcome_counts"]["pre_llm_abstain"] == 52
    assert not checkpoint.exists()
    assert len(list((tmp_path / "eval_runs").glob("answers-*.json"))) == 1  # one report, for the whole run


def test_resuming_with_a_different_model_is_refused(session, tmp_path, monkeypatch, capsys) -> None:
    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set(range(5, 500))))
    assert _unpaced(session) == 2
    checkpoint = next((tmp_path / "eval_runs").glob("answers-*.partial.json"))

    class Other(FlakyLLM):
        model_name = "a-different-model"

    _patch(monkeypatch, tmp_path, lambda: Other(failing_calls=set()))
    exit_code = _unpaced(session, resume=checkpoint)

    assert exit_code == 1
    assert "different configuration" in capsys.readouterr().err
    assert session.execute(select(EvalRun)).scalars().all() == []


def test_the_call_budget_is_printed_before_the_run_starts(session, tmp_path, monkeypatch, capsys) -> None:
    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set()))

    run_module.run_answers(session, requests_per_minute=600)

    out = capsys.readouterr().out
    assert "Call budget: about 53 model calls" in out
    assert out.index("Call budget") < out.index("Answer evaluation (suite=answers)")


def test_the_real_provider_is_wrapped_in_the_pacer(session, tmp_path, monkeypatch) -> None:
    built: list = []

    class Spy(PacedLLMProvider):
        def __init__(self, inner, **kwargs) -> None:
            built.append(kwargs["requests_per_minute"])
            super().__init__(inner, **{**kwargs, "requests_per_minute": 100_000})

    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set()))
    monkeypatch.setattr(run_module, "PacedLLMProvider", Spy)

    assert run_module.run_answers(session, requests_per_minute=7) == 0

    assert built == [7]


def test_zero_turns_pacing_off(session, tmp_path, monkeypatch, capsys) -> None:
    _patch(monkeypatch, tmp_path, lambda: FlakyLLM(failing_calls=set()))
    monkeypatch.setattr(run_module, "PacedLLMProvider", lambda *a, **k: (_ for _ in ()).throw(AssertionError("paced")))

    assert _unpaced(session) == 0
    assert "unpaced" in capsys.readouterr().out


def test_the_command_line_flags_are_parsed_and_checked(session, monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(run_module, "run_answers", lambda s, **kwargs: seen.update(kwargs) or 0)

    run_module.main(["--suite", "answers", "--requests-per-minute", "4", "--max-consecutive-failures", "3", "--resume", "x.json"])

    assert seen["requests_per_minute"] == 4
    assert seen["max_consecutive_failures"] == 3
    assert str(seen["resume"]) == "x.json"
    with pytest.raises(SystemExit):
        run_module.main(["--suite", "answers", "--max-consecutive-failures", "0"])
    with pytest.raises(SystemExit):
        run_module.main(["--suite", "answers", "--requests-per-minute", "-1"])


def test_a_question_that_never_reached_the_model_does_not_end_a_run_of_failures(session, storage, embeddings) -> None:
    """Pre-LLM abstentions sit between questions; the provider failing on both sides of one is still
    two failures in a row, because the provider was not asked in between."""
    llm = FlakyLLM(failing_calls=set(range(5, 500)))

    result = _run(session, storage, embeddings, llm, max_consecutive_failures=2)

    assert result.stopped_early is not None
    assert "2 provider failures in a row" in result.stopped_early
