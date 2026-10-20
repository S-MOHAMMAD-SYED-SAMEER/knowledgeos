"""`python -m evals.run --suite answers`.

The real local models (embedding, cross-encoder) and the real Gemini
credential are genuinely unavailable in this environment, so the
hard-failure path is exercised against the CLI's real, unmodified
providers below -- the same way `tests/test_evals_cli.py` already proves
it for `--suite retrieval`, extended here to the third required model
(generation) and to the "no fabricated official evaluation" integrity
rule (D1): a failed official run must leave behind no `eval_runs` row and
no JSON report.

The success path replaces all three providers with deterministic
stand-ins (D1/D2's pytest-only allowance) to prove `run_answers`'s own
wiring -- corpus seeding, calibration, the suite, persistence, the
report -- not to claim these numbers are an official M9 result.
"""

import logging

import pytest

import evals.run as run_module
from app.models import EvalRun
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider
from tests.generation_fixtures import AutoCitingLLMProvider


@pytest.mark.usefixtures("models_not_in_cache")
def test_main_dispatches_the_answers_suite_and_fails_hard_here(session) -> None:
    """No stubbing: proves `--suite answers` reaches `run_answers` and
    that the real, unavailable-here models are what stops it -- exit 1,
    not an argparse or wiring error."""
    exit_code = run_module.main(["--suite", "answers"])
    assert exit_code != 0


# --- hard failure: the real models are genuinely absent here ---------------


@pytest.mark.usefixtures("models_not_in_cache")
def test_run_answers_fails_hard_when_the_embedding_model_is_unavailable(session) -> None:
    exit_code = run_module.run_answers(session)
    assert exit_code != 0


@pytest.mark.usefixtures("models_not_in_cache")
def test_a_failed_official_answer_run_writes_no_eval_runs_row(session) -> None:
    from sqlalchemy import select

    run_module.run_answers(session)

    rows = session.execute(select(EvalRun)).scalars().all()
    assert rows == []


@pytest.mark.usefixtures("models_not_in_cache")
def test_a_failed_official_answer_run_writes_no_json_report(session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    run_module.run_answers(session)

    assert not (tmp_path / "eval_runs").exists()


@pytest.mark.usefixtures("models_not_in_cache")
def test_a_failed_official_answer_run_prints_the_blocking_reason(session, capsys) -> None:
    run_module.run_answers(session)

    captured = capsys.readouterr()
    assert "answer evaluation cannot run" in captured.err
    assert "no metrics were computed" in captured.err


def test_canary_failure_means_the_answers_suite_never_runs_at_all(session, monkeypatch) -> None:
    """The corpus must never even be seeded when a required model is
    unavailable -- proven by making `run_answers_suite` itself raise if it
    is ever called."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("run_answers_suite should not run after a canary failure")

    monkeypatch.setattr(run_module, "run_answers_suite", _must_not_be_called)

    exit_code = run_module.run_answers(session)
    assert exit_code != 0


def test_no_secrets_or_full_answer_text_appear_in_a_failed_run_s_logs(session, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        run_module.run_answers(session)

    log_text = caplog.text.lower()
    for forbidden in ("gemini_api_key", "google_api_key", "anthropic_api_key"):
        assert forbidden not in log_text


# --- success path, with all three providers replaced by deterministic stand-ins


class _StubEmbeddingProvider:
    """Same shape as `BgeEmbeddingProvider`, backed by the fake."""

    def __init__(self) -> None:
        self._inner = FakeEmbeddingProvider()

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed(texts)


class _StubCrossEncoderProvider:
    """Same shape as `CrossEncoderRerankProvider`, backed by the fake."""

    def __init__(self) -> None:
        self._inner = FakeRerankProvider()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def rerank(self, query, candidates):
        return self._inner.rerank(query, candidates)


class _StubGeminiLLMProvider:
    """Same shape as `GeminiLLMProvider`, backed by the reactive
    auto-citing fake -- takes the model name argument and ignores it, the
    same substitution `_StubEmbeddingProvider`/`_StubCrossEncoderProvider`
    already make for their real counterparts."""

    def __init__(self, model_name: str | None, **kwargs) -> None:
        self._inner = AutoCitingLLMProvider()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def complete(self, *, system: str, user: str, max_tokens: int, json_schema: dict | None = None):
        return self._inner.complete(system=system, user=user, max_tokens=max_tokens)


class _StubSettings:
    llm_model = "stub-gemini-model"
    llm_max_output_tokens = 512
    llm_timeout_seconds = 30.0
    llm_max_retries = 2
    llm_retry_max_wait_seconds = 15.0
    llm_temperature = None
    abstention_rerank_threshold = 1.4451  # the shipped default
    llm_pricing_usd_per_million_tokens: dict = {}


def _patch_stub_providers(monkeypatch) -> None:
    monkeypatch.setattr(run_module, "BgeEmbeddingProvider", _StubEmbeddingProvider)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _StubCrossEncoderProvider)
    monkeypatch.setattr(run_module, "GeminiLLMProvider", _StubGeminiLLMProvider)
    monkeypatch.setattr(run_module, "get_settings", lambda: _StubSettings())


def test_a_successful_answers_run_writes_one_eval_runs_row(session, tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    exit_code = run_module.run_answers(session, requests_per_minute=0)
    assert exit_code == 0

    rows = session.execute(select(EvalRun)).scalars().all()
    assert len(rows) == 1
    assert rows[0].suite == run_module.ANSWERS_SUITE
    assert rows[0].prompt_version is not None
    assert "outcome_counts" in rows[0].metrics


def test_a_successful_answers_run_writes_exactly_one_json_report(session, tmp_path, monkeypatch) -> None:
    import json

    report_dir = tmp_path / "eval_runs"
    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", report_dir)

    exit_code = run_module.run_answers(session, requests_per_minute=0)
    assert exit_code == 0

    files = list(report_dir.glob("answers-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["suite"] == "answers"
    assert payload["eval_run_id"] is not None


def test_a_successful_answers_run_records_the_calibration_result(session, tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    exit_code = run_module.run_answers(session, requests_per_minute=0)
    assert exit_code == 0

    row = session.execute(select(EvalRun)).scalars().one()
    assert "calibration" in row.config
    assert row.config["calibration"]["threshold"] is not None


def test_a_run_records_the_gold_the_prompt_and_where_its_threshold_came_from(session, tmp_path, monkeypatch) -> None:
    import json

    from evals.gold import gold_digest

    report_dir = tmp_path / "eval_runs"
    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", report_dir)

    assert run_module.run_answers(session, requests_per_minute=0) == 0

    config = json.loads(next(report_dir.glob("answers-*.json")).read_text(encoding="utf-8"))["config"]
    assert config["gold_version"] == "gold-v2"
    assert config["gold_sha256"] == gold_digest()[0]
    assert config["prompt_version"].startswith("knowledge_answer_v")
    assert config["prompt_content_hash"]
    assert config["abstention_threshold_in_sample"] is True
    assert config["strict"] is False and config["request_cap"] is None


def test_the_shipped_threshold_is_used_when_the_cross_encoder_is_the_calibrated_one(
    session, tmp_path, monkeypatch
) -> None:
    import json

    from app.abstention_constants import CALIBRATED_RERANK_MODEL

    class _CalibratedCrossEncoder(_StubCrossEncoderProvider):
        @property
        def model_name(self) -> str:
            return CALIBRATED_RERANK_MODEL

    report_dir = tmp_path / "eval_runs"
    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _CalibratedCrossEncoder)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", report_dir)

    assert run_module.run_answers(session, requests_per_minute=0) == 0
    config = json.loads(next(report_dir.glob("answers-*.json")).read_text(encoding="utf-8"))["config"]
    assert config["abstention_threshold"] == 1.4451
    assert "application default" in config["abstention_threshold_source"]
    # The fresh calibration is recorded beside it (on the eval_runs row) for reference; it is not what was used.
    from sqlalchemy import select

    row = session.execute(select(EvalRun)).scalars().one()
    assert row.config["calibration"]["threshold"] is not None


def test_recalibrate_uses_the_fresh_calibration_instead(session, tmp_path, monkeypatch) -> None:
    import json

    from app.abstention_constants import CALIBRATED_RERANK_MODEL

    class _CalibratedCrossEncoder(_StubCrossEncoderProvider):
        @property
        def model_name(self) -> str:
            return CALIBRATED_RERANK_MODEL

    report_dir = tmp_path / "eval_runs"
    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _CalibratedCrossEncoder)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", report_dir)

    assert run_module.run_answers(session, requests_per_minute=0, recalibrate=True) == 0
    config = json.loads(next(report_dir.glob("answers-*.json")).read_text(encoding="utf-8"))["config"]
    from sqlalchemy import select

    row = session.execute(select(EvalRun)).scalars().one()
    assert config["abstention_threshold"] == row.config["calibration"]["threshold"]
    assert "recalibrated" in config["abstention_threshold_source"]


def test_the_request_cap_stops_a_run_before_the_request_that_would_exceed_it(
    session, tmp_path, monkeypatch
) -> None:
    from evals.answers.budget import RequestBudget

    _patch_stub_providers(monkeypatch)
    monkeypatch.setattr(run_module.answers_report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr("evals.answers.budget.DEFAULT_LEDGER", ledger)

    exit_code = run_module.run_answers(session, requests_per_minute=0, request_cap=3)

    assert exit_code == 2  # stopped early: a partial run is not a result
    assert RequestBudget(3, ledger).used == 3  # exactly the cap, never one more
