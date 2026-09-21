"""Turning an `AnswerSuiteResult` into the JSON artifact and console
summary §13/§14 ask for — mirrors `evals/retrieval/report.py` exactly:
neither persists anything (`evals/run.py` writes the `eval_runs` row),
and every number here is read from a result the suite actually produced.

**Every unavailable computation stays visibly unavailable.** A metric
whose denominator was empty, or whose pricing was unconfigured, is `None`
in the payload with a reason alongside it — never silently dropped from
the JSON, and never replaced with `0` or an average that quietly skipped
it.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.observability.pricing import PricingError, projected_monthly_cost_usd
from evals.answers.metrics import (
    AttemptOutcome,
    abstention_precision_recall,
    citation_coverage_mean,
    cost_summary,
    grounded_answer_rate,
    invalid_citation_rate,
    latency_summary,
    token_summary,
    unsupported_claim_rate,
)
from evals.answers.suite import AnswerSuiteResult

DEFAULT_REPORT_DIR = Path("var") / "eval_runs"

MONTHLY_PROJECTION_QUERIES_PER_DAY = (100, 1_000)


def _latency_section(attempts) -> dict:
    section = {}
    for stage in ("retrieval_ms", "rerank_ms", "llm_ms", "total_ms"):
        values = [getattr(a, stage) for a in attempts if getattr(a, stage) is not None]
        if not values:
            section[stage] = None
            continue
        summary = latency_summary(values)
        section[stage] = {
            "p50_ms": summary.p50_ms,
            "p95_ms": summary.p95_ms,
            "count": summary.count,
        }
    return section


def _semantic_section(attempts) -> dict:
    scored_attempts = [a for a in attempts if a.semantic is not None]
    return {
        "configured": bool(scored_attempts),
        "scored_count": sum(1 for a in scored_attempts if a.semantic.status == "scored"),
        "unavailable_count": sum(1 for a in scored_attempts if a.semantic.status == "unavailable"),
        "unsupported_claim_threshold_configured": unsupported_claim_rate(attempts) is not None,
    }


def metrics_payload(result: AnswerSuiteResult, *, pricing_table: dict) -> dict:
    attempts = result.attempts
    payload: dict = {
        "total_question_count": result.total_question_count,
        "dev_question_count": result.dev_question_count,
        "test_question_count": result.test_question_count,
        "outcome_counts": {
            outcome.value: sum(1 for a in attempts if a.outcome == outcome)
            for outcome in AttemptOutcome
        },
    }

    try:
        payload["invalid_citation_rate"] = invalid_citation_rate(attempts)
    except ValueError:
        payload["invalid_citation_rate"] = None

    try:
        payload["citation_coverage_mean"] = citation_coverage_mean(attempts)
    except ValueError:
        payload["citation_coverage_mean"] = None

    try:
        payload["grounded_answer_rate"] = grounded_answer_rate(attempts)
    except ValueError:
        payload["grounded_answer_rate"] = None

    payload["unsupported_claim_rate"] = unsupported_claim_rate(attempts)
    payload["semantic_grounding"] = _semantic_section(attempts)

    try:
        abstention = abstention_precision_recall(attempts)
        payload["abstention"] = asdict(abstention)
    except ValueError:
        payload["abstention"] = None

    payload["latency"] = _latency_section(attempts)

    try:
        tokens = token_summary(attempts)
        payload["tokens"] = asdict(tokens)
    except ValueError:
        payload["tokens"] = None

    try:
        cost = cost_summary(attempts, pricing_table=pricing_table)
        payload["cost"] = {
            **asdict(cost),
            "projected_monthly_usd": {
                str(n): projected_monthly_cost_usd(
                    mean_cost_per_query_usd=cost.mean_cost_usd, queries_per_day=n
                )
                for n in MONTHLY_PROJECTION_QUERIES_PER_DAY
            },
        }
        payload["cost_unavailable_reason"] = None
    except PricingError as exc:
        payload["cost"] = None
        payload["cost_unavailable_reason"] = str(exc)
    except ValueError:
        payload["cost"] = None
        payload["cost_unavailable_reason"] = "no attempt reached the model"

    return payload


def report_filename(*, when: datetime | None = None) -> str:
    moment = when or datetime.now(UTC)
    return f"answers-{moment.strftime('%Y%m%dT%H%M%SZ')}.json"


def build_json_report(
    result: AnswerSuiteResult, *, eval_run_id: UUID | None, pricing_table: dict
) -> dict:
    return {
        "suite": "answers",
        "eval_run_id": str(eval_run_id) if eval_run_id is not None else None,
        "config": result.config,
        "metrics": metrics_payload(result, pricing_table=pricing_table),
        "persisted_query_ids": [str(qid) for qid in result.persisted_query_ids],
    }


def write_json_report(
    result: AnswerSuiteResult,
    *,
    eval_run_id: UUID | None,
    pricing_table: dict,
    directory: Path = DEFAULT_REPORT_DIR,
    when: datetime | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / report_filename(when=when)
    report = build_json_report(result, eval_run_id=eval_run_id, pricing_table=pricing_table)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path


def render_console_summary(
    result: AnswerSuiteResult, *, eval_run_id: UUID | None, pricing_table: dict
) -> str:
    metrics = metrics_payload(result, pricing_table=pricing_table)
    lines = ["Answer evaluation (suite=answers)"]
    if eval_run_id is not None:
        lines.append(f"  eval_run_id: {eval_run_id}")

    lines.append(f"  embedding model:  {result.config['embedding_model']}")
    lines.append(f"  rerank model:     {result.config['rerank_model']}")
    lines.append(f"  llm model:        {result.config['llm_model']}")
    lines.append(f"  semantic scorer:  {result.config['semantic_scorer_model']}")
    lines.append(
        f"  abstention threshold: {result.config['abstention_threshold']}"
        + ("" if result.config["abstention_threshold"] is not None else "  (uncalibrated / inactive)")
    )
    lines.append(
        f"  questions:        total={result.total_question_count} "
        f"dev={result.dev_question_count} test={result.test_question_count}"
    )
    lines.append(f"  outcomes:         {metrics['outcome_counts']}")
    lines.append("")

    def _fmt(value, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.4f}{suffix}"

    lines.append(f"  invalid citation rate:     {_fmt(metrics['invalid_citation_rate'])}  (gate: 0%)")
    lines.append(f"  citation coverage (mean):  {_fmt(metrics['citation_coverage_mean'])}  (gate: 100% on non-abstained)")
    lines.append(f"  grounded-answer rate:      {_fmt(metrics['grounded_answer_rate'])}")
    lines.append(
        f"  unsupported-claim rate:    {_fmt(metrics['unsupported_claim_rate'])}"
        + ("" if metrics["unsupported_claim_rate"] is not None else "  (no threshold calibrated -- see evals/answers/semantic.py)")
    )

    abstention = metrics["abstention"]
    if abstention is None:
        lines.append("  abstention precision/recall: n/a (no expected-abstain questions)")
    else:
        lines.append(
            f"  abstention recall:  {abstention['recall']:.4f}  (gate: >= 90%; "
            f"n={abstention['positive_count']} -- see the split file's own sample-size caveat)"
        )
        precision = abstention["precision"]
        lines.append(
            f"  abstention precision: {'n/a' if precision is None else f'{precision:.4f}'}"
        )

    lines.append("")
    lines.append("  latency (ms), p50 / p95 / n:")
    for stage, label in (
        ("retrieval_ms", "retrieval"),
        ("rerank_ms", "rerank"),
        ("llm_ms", "generation"),
        ("total_ms", "total"),
    ):
        entry = metrics["latency"][stage]
        if entry is None:
            lines.append(f"    {label:<10} n/a")
        else:
            lines.append(
                f"    {label:<10} {entry['p50_ms']:>8.1f} / {entry['p95_ms']:>8.1f} / {entry['count']}"
            )

    lines.append("")
    if metrics["cost"] is None:
        lines.append(f"  cost: unavailable ({metrics['cost_unavailable_reason']})")
    else:
        cost = metrics["cost"]
        lines.append(f"  mean cost per query: ${cost['mean_cost_usd']:.6f}")
        for n, projected in cost["projected_monthly_usd"].items():
            lines.append(f"  projected monthly cost at {n}/day: ${projected:,.2f}")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPORT_DIR",
    "MONTHLY_PROJECTION_QUERIES_PER_DAY",
    "build_json_report",
    "metrics_payload",
    "render_console_summary",
    "report_filename",
    "write_json_report",
]
