"""`python -m evals.run --suite retrieval`.

The real local models are genuinely unavailable in this environment (no
sentence-transformers cache), so the hard-failure path is exercised against
the CLI's real, unmodified providers below -- no monkeypatching needed to
prove that case. The success path is exercised with the CLI's two provider
classes replaced by deterministic stand-ins (D2's pytest-only allowance):
what is under test there is `run.py`'s own wiring -- does it persist what
`run_retrieval_suite` reports, write the JSON artifact, print a summary --
not a claim that these numbers are an official evaluation result.
"""

import evals.run as run_module
from app.models import RETRIEVAL_SUITE, EvalRun
from app.providers.fake_embeddings import FakeEmbeddingProvider
from app.providers.fake_reranker import FakeRerankProvider


def test_main_rejects_an_unsupported_suite_name() -> None:
    try:
        run_module.main(["--suite", "generation"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected argparse to reject an unknown suite")


def test_main_requires_the_suite_argument() -> None:
    try:
        run_module.main([])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected argparse to require --suite")


# --- hard failure: the real models are genuinely absent here -------------


def test_run_retrieval_fails_hard_when_the_embedding_model_is_unavailable(
    session,
) -> None:
    exit_code = run_module.run_retrieval(session)
    assert exit_code != 0


def test_a_failed_run_writes_no_eval_runs_row(session) -> None:
    from sqlalchemy import select

    run_module.run_retrieval(session)

    rows = session.execute(select(EvalRun)).scalars().all()
    assert rows == []


def test_a_failed_run_writes_no_json_report(session, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_module.report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    run_module.run_retrieval(session)

    assert not (tmp_path / "eval_runs").exists()


def test_canary_failure_means_the_suite_never_runs_at_all(session, monkeypatch) -> None:
    """The corpus must never even be touched when the embedding model is
    unavailable -- proven by making `run_retrieval_suite` itself raise if
    it is ever called."""

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("run_retrieval_suite should not run after a canary failure")

    monkeypatch.setattr(run_module, "run_retrieval_suite", _must_not_be_called)

    exit_code = run_module.run_retrieval(session)
    assert exit_code != 0


# --- success path, with the two providers replaced by deterministic stand-ins --


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


def test_a_successful_run_writes_one_eval_runs_row(session, tmp_path, monkeypatch) -> None:
    from sqlalchemy import select

    monkeypatch.setattr(run_module, "BgeEmbeddingProvider", _StubEmbeddingProvider)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _StubCrossEncoderProvider)
    monkeypatch.setattr(run_module.report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    exit_code = run_module.run_retrieval(session)
    assert exit_code == 0

    rows = session.execute(select(EvalRun)).scalars().all()
    assert len(rows) == 1
    assert rows[0].suite == RETRIEVAL_SUITE
    assert rows[0].prompt_version is None
    assert rows[0].config["chunk_size_tokens"] == 512
    assert "passthrough" in rows[0].metrics
    assert "cross_encoder" in rows[0].metrics


def test_a_successful_run_writes_exactly_one_json_report(
    session, tmp_path, monkeypatch
) -> None:
    import json

    report_dir = tmp_path / "eval_runs"
    monkeypatch.setattr(run_module, "BgeEmbeddingProvider", _StubEmbeddingProvider)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _StubCrossEncoderProvider)
    monkeypatch.setattr(run_module.report, "DEFAULT_REPORT_DIR", report_dir)

    exit_code = run_module.run_retrieval(session)
    assert exit_code == 0

    files = list(report_dir.glob("retrieval-*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["suite"] == "retrieval"
    assert payload["eval_run_id"] is not None


def test_a_successful_run_never_calls_the_real_local_model_classes(
    session, tmp_path, monkeypatch
) -> None:
    """Confirms the stub substitution actually took effect -- if it hadn't,
    this run would hit the same hard failure the tests above prove."""
    monkeypatch.setattr(run_module, "BgeEmbeddingProvider", _StubEmbeddingProvider)
    monkeypatch.setattr(run_module, "CrossEncoderRerankProvider", _StubCrossEncoderProvider)
    monkeypatch.setattr(run_module.report, "DEFAULT_REPORT_DIR", tmp_path / "eval_runs")

    exit_code = run_module.run_retrieval(session)
    assert exit_code == 0
