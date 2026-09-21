"""JSON and console reporting from a `SuiteResult` -- no database, no
provider: these tests build a `SuiteResult` by hand so they can run without
PostgreSQL or any model."""

import json
import uuid
from datetime import UTC, datetime

from evals.retrieval.corpus import CorpusSeedResult
from evals.retrieval.report import (
    build_json_report,
    metrics_payload,
    render_console_summary,
    report_filename,
    write_json_report,
)
from evals.retrieval.suite import ConditionMetrics, SuiteResult


def _result() -> SuiteResult:
    return SuiteResult(
        passthrough=ConditionMetrics(
            recall_at_5=0.5, recall_at_10=0.6, precision_at_5=0.4, mrr=0.45, ndcg_at_10=0.55
        ),
        cross_encoder=ConditionMetrics(
            recall_at_5=0.7, recall_at_10=0.8, precision_at_5=0.5, mrr=0.65, ndcg_at_10=0.75
        ),
        metadata_filter_correctness=1.0,
        evaluated_question_count=46,
        excluded_question_count=6,
        filtered_question_count=7,
        total_question_count=52,
        corpus=CorpusSeedResult(
            documents=[], total_active_chunks=30, skipped_existing=0, newly_seeded=11
        ),
        config={
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "passthrough_rerank_model": "passthrough",
            "cross_encoder_rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 64,
            "rrf_k": 60,
            "precision_depth": 5,
            "ndcg_depth": 10,
            "question_count": 52,
            "corpus_active_chunks": 30,
        },
    )


def test_metrics_payload_has_both_conditions_and_question_accounting() -> None:
    payload = metrics_payload(_result())
    assert payload["passthrough"]["recall_at_5"] == 0.5
    assert payload["cross_encoder"]["recall_at_5"] == 0.7
    assert payload["metadata_filter_correctness"] == 1.0
    assert payload["evaluated_question_count"] == 46
    assert payload["excluded_question_count"] == 6
    assert payload["filtered_question_count"] == 7
    assert payload["total_question_count"] == 52


def test_report_filename_is_timestamped_and_never_collides() -> None:
    a = report_filename(when=datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC))
    b = report_filename(when=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC))
    assert a != b
    assert a.startswith("retrieval-")
    assert a.endswith(".json")


def test_build_json_report_is_json_serializable_and_carries_the_run_id() -> None:
    run_id = uuid.uuid4()
    report = build_json_report(_result(), eval_run_id=run_id)
    dumped = json.dumps(report)
    reloaded = json.loads(dumped)
    assert reloaded["eval_run_id"] == str(run_id)
    assert reloaded["suite"] == "retrieval"
    assert reloaded["config"]["chunk_size_tokens"] == 512
    assert reloaded["metrics"]["passthrough"]["mrr"] == 0.45


def test_build_json_report_allows_a_null_run_id() -> None:
    report = build_json_report(_result(), eval_run_id=None)
    assert report["eval_run_id"] is None


def test_write_json_report_writes_a_readable_file(tmp_path) -> None:
    run_id = uuid.uuid4()
    path = write_json_report(_result(), eval_run_id=run_id, directory=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path

    on_disk = json.loads(path.read_text())
    assert on_disk["eval_run_id"] == str(run_id)
    assert on_disk["metrics"]["cross_encoder"]["recall_at_10"] == 0.8


def test_write_json_report_creates_the_directory_if_missing(tmp_path) -> None:
    target = tmp_path / "nested" / "eval_runs"
    path = write_json_report(_result(), eval_run_id=None, directory=target)
    assert path.exists()
    assert path.parent == target


def test_console_summary_contains_no_fabricated_section_and_both_conditions() -> None:
    summary = render_console_summary(_result())
    assert "passthrough" in summary
    assert "cross_encoder" in summary
    assert "0.5000" in summary  # passthrough recall@5
    assert "0.7000" in summary  # cross_encoder recall@5
    assert "1.0000" in summary  # metadata-filter correctness
    assert "52" in summary  # total question count


def test_console_summary_includes_the_run_id_when_given() -> None:
    run_id = uuid.uuid4()
    summary = render_console_summary(_result(), eval_run_id=run_id)
    assert str(run_id) in summary


def test_console_summary_omits_run_id_line_when_none() -> None:
    summary = render_console_summary(_result(), eval_run_id=None)
    assert "eval_run_id" not in summary
