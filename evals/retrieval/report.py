"""Turning a `SuiteResult` into the two outputs the specification asks for:
a machine-readable JSON artifact and a human-readable console summary
(D7). Neither persists anything -- `evals/run.py` is what writes the
`eval_runs` row, using the same `metrics_payload()`/`config` this module
also uses for the JSON file, so the database row and the file on disk never
disagree about what a run measured.

No fabricated numbers ever pass through here: every value below is read
from a `SuiteResult` that `run_retrieval_suite` actually produced. This
module has no path where it invents, estimates, or defaults a metric.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from evals.retrieval.suite import SuiteResult

DEFAULT_REPORT_DIR = Path("var") / "eval_runs"


def metrics_payload(result: SuiteResult) -> dict:
    """The `metrics` JSONB payload: both conditions, the metadata-filter
    number, and the question-count accounting (D4) that explains how the
    rank-sensitive means were computed."""
    return {
        "passthrough": asdict(result.passthrough),
        "cross_encoder": asdict(result.cross_encoder),
        "metadata_filter_correctness": result.metadata_filter_correctness,
        "evaluated_question_count": result.evaluated_question_count,
        "excluded_question_count": result.excluded_question_count,
        "filtered_question_count": result.filtered_question_count,
        "total_question_count": result.total_question_count,
    }


def report_filename(*, when: datetime | None = None) -> str:
    """`retrieval-<UTC timestamp>.json` -- one file per run, never
    overwritten by the next one."""
    moment = when or datetime.now(UTC)
    return f"retrieval-{moment.strftime('%Y%m%dT%H%M%SZ')}.json"


def build_json_report(result: SuiteResult, *, eval_run_id: UUID | None) -> dict:
    return {
        "suite": "retrieval",
        "eval_run_id": str(eval_run_id) if eval_run_id is not None else None,
        "config": result.config,
        "metrics": metrics_payload(result),
    }


def write_json_report(
    result: SuiteResult,
    *,
    eval_run_id: UUID | None,
    directory: Path = DEFAULT_REPORT_DIR,
    when: datetime | None = None,
) -> Path:
    """Write the JSON artifact and return its path. Creates `directory` if
    it does not already exist -- `var/` is gitignored, so this is the first
    thing to create it on a fresh checkout."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / report_filename(when=when)
    path.write_text(json.dumps(build_json_report(result, eval_run_id=eval_run_id), indent=2, sort_keys=True))
    return path


def render_console_summary(result: SuiteResult, *, eval_run_id: UUID | None = None) -> str:
    """A short, human-readable summary: both conditions side by side, the
    metadata-filter number, and the question-count accounting."""
    lines = ["Retrieval evaluation (suite=retrieval)"]
    if eval_run_id is not None:
        lines.append(f"  eval_run_id: {eval_run_id}")

    lines.append(f"  embedding model:            {result.config['embedding_model']}")
    lines.append(
        f"  reranking models:           passthrough={result.config['passthrough_rerank_model']}, "
        f"cross_encoder={result.config['cross_encoder_rerank_model']}"
    )
    lines.append(
        f"  chunk config:                size={result.config['chunk_size_tokens']} "
        f"overlap={result.config['chunk_overlap_tokens']} tokens"
    )
    lines.append(f"  corpus active chunks:       {result.corpus.total_active_chunks}")
    lines.append(
        f"  questions:                  total={result.total_question_count} "
        f"evaluated={result.evaluated_question_count} "
        f"excluded(no expected chunks)={result.excluded_question_count} "
        f"filtered={result.filtered_question_count}"
    )
    lines.append("")
    lines.append(f"  {'metric':<16}{'passthrough':>14}{'cross_encoder':>16}")
    for label, field in (
        ("Recall@5", "recall_at_5"),
        ("Recall@10", "recall_at_10"),
        ("Precision@5", "precision_at_5"),
        ("MRR", "mrr"),
        ("nDCG@10", "ndcg_at_10"),
    ):
        a = getattr(result.passthrough, field)
        b = getattr(result.cross_encoder, field)
        lines.append(f"  {label:<16}{a:>14.4f}{b:>16.4f}")
    lines.append("")
    lines.append(
        f"  metadata-filter correctness: {result.metadata_filter_correctness:.4f} "
        f"(condition-invariant, over {result.filtered_question_count} filtered question(s))"
    )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPORT_DIR",
    "build_json_report",
    "metrics_payload",
    "render_console_summary",
    "report_filename",
    "write_json_report",
]
