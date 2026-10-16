"""What an answers run keeps: every raw model reply, the answer shown, the temperature and the gold label,
so the run can be scored again offline and nobody mistakes what the numbers are scored against."""

import json

import pytest

from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider
from app.providers.llm import LLMError, LLMResult
from app.storage import LocalStorage
from evals.answers.checkpoint import _attempt_from_dict, _attempt_to_dict
from evals.answers.metrics import AttemptOutcome
from evals.answers.recording import RecordingLLMProvider
from evals.answers.report import GOLD_LABEL, build_json_report, render_console_summary
from evals.answers.suite import run_answers_suite
from tests.generation_fixtures import AutoCitingLLMProvider


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


def _run(session, storage, llm, **kwargs):
    return run_answers_suite(
        session,
        storage,
        FakeEmbeddingProvider(),
        FakeRerankProvider(),
        llm,
        semantic_scorer=None,
        abstention_threshold=None,
        max_tokens=512,
        **kwargs,
    )


def test_the_recorder_keeps_the_raw_text_and_hands_each_call_out_once() -> None:
    class Inner:
        model_name = "m"

        def complete(self, *, system, user, max_tokens, json_schema=None):
            return LLMResult(text="raw {text}", input_tokens=7, output_tokens=3, finish_reason="STOP")

    recorder = RecordingLLMProvider(Inner())
    recorder.complete(system="s", user="u", max_tokens=8)
    recorder.complete(system="s", user="u", max_tokens=8)

    calls = recorder.take_calls()
    assert [c["text"] for c in calls] == ["raw {text}", "raw {text}"]
    assert calls[0]["input_tokens"] == 7 and calls[0]["finish_reason"] == "STOP"
    assert recorder.take_calls() == []  # handed out once
    assert recorder.total_calls == 2 and recorder.model_name == "m"


def test_every_successful_attempt_keeps_the_answer_citations_evidence_and_raw_reply(
    session, storage
) -> None:
    result = _run(session, storage, RecordingLLMProvider(AutoCitingLLMProvider()))

    answered = [a for a in result.attempts if a.outcome == AttemptOutcome.SUCCESS]
    assert answered
    for attempt in answered:
        assert attempt.answer_text
        assert attempt.citations is not None and attempt.selected_chunk_uids
        assert len(attempt.raw_replies) == attempt.attempts_used
        assert all(reply["text"] for reply in attempt.raw_replies)


def test_a_provider_failure_keeps_the_adapters_message_in_the_stop_reason(session, storage) -> None:
    class Failing:
        model_name = "m"

        def complete(self, *, system, user, max_tokens, json_schema=None):
            raise LLMError("generation failed (HTTP 503)")

    result = _run(session, storage, RecordingLLMProvider(Failing()), max_consecutive_failures=1)

    assert not result.complete
    assert "generation failed (HTTP 503)" in result.stopped_early
    assert result.provider_errors[0]["message"] == "generation failed (HTTP 503)"


def test_the_report_carries_the_gold_label_every_attempt_and_survives_a_checkpoint(
    session, storage
) -> None:
    result = _run(session, storage, RecordingLLMProvider(AutoCitingLLMProvider()))
    result.config["llm_temperature"] = 1.0

    report = build_json_report(result, eval_run_id=None, pricing_table={})
    json.dumps(report)  # serialisable as it stands

    assert report["gold_label"] == GOLD_LABEL
    assert "AI-written" in GOLD_LABEL and "not yet human-reviewed" in GOLD_LABEL
    assert len(report["attempts"]) == result.total_question_count
    first = next(a for a in report["attempts"] if a["outcome"] == "success")
    assert first["raw_replies"] and first["answer_text"]

    summary = render_console_summary(result, eval_run_id=None, pricing_table={})
    assert GOLD_LABEL in summary and "llm temperature:  1.0" in summary

    # A checkpoint written mid-run keeps the same fields, so a resumed run loses no raw reply.
    restored = _attempt_from_dict(json.loads(json.dumps(_attempt_to_dict(result.attempts[0]))))
    assert restored == result.attempts[0]


def test_a_question_that_never_reached_the_model_is_left_out_of_tokens_and_cost() -> None:
    """A pre-LLM abstention has token counts of 0 but no model; counting it priced a model called None
    and pulled the mean tokens down. Only questions the model answered are measured."""
    from app.observability.pricing import PricingError
    from evals.answers.metrics import AnswerAttempt, cost_summary, token_summary

    def attempt(qid, outcome, model, tokens_in, tokens_out):
        return AnswerAttempt(
            question_id=qid, expected_abstain=False, outcome=outcome, abstained=False, citation_valid=True,
            citation_coverage=1.0, grounded=True, semantic=None, model_name=model, input_tokens=tokens_in,
            output_tokens=tokens_out, retrieval_ms=1.0, rerank_ms=1.0, llm_ms=None, total_ms=2.0,
        )

    attempts = [
        attempt("a", AttemptOutcome.SUCCESS, "gemini-x", 1000, 100),
        attempt("b", AttemptOutcome.PRE_LLM_ABSTAIN, None, 0, 0),
    ]
    summary = token_summary(attempts)
    assert (summary.count, summary.mean_input_tokens, summary.total_output_tokens) == (1, 1000, 100)

    table = {"gemini-x": {"input": 1.0, "output": 2.0}}
    assert cost_summary(attempts, pricing_table=table).total_cost_usd == pytest.approx(0.0012)
    with pytest.raises(PricingError, match="gemini-x"):
        cost_summary(attempts, pricing_table={})
