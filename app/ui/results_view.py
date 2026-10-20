"""The headline numbers on the Evaluation page, read from the committed results files.

`docs/results/` holds the evaluation runs as JSON, written by `python -m evals.run`. The page shows the
numbers in them, each with the sample size behind it, instead of an empty state that waits for a run to
be written into the database it happens to be pointed at. This module only reads JSON: it does not
import `evals/` (the guard that no `app` module does still holds) and it does not recompute anything
the file already states.

Every figure is a count over a named denominator ("27 of 34"), never a bare percentage, and every
answers run carries the gold label and the threshold's provenance next to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: The repository's committed results. In the Docker image the same folder is copied to the same place.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "results"

#: What a run scored against the first gold (everything before Gold Review 2) is called.
GOLD_V1 = "gold-v1"


@dataclass(frozen=True)
class Figure:
    """One number, with what it is a count of."""

    label: str
    value: str
    detail: str = ""


@dataclass(frozen=True)
class AnswersRunView:
    file: str
    date: str
    model: str
    prompt_version: str
    gold_version: str
    gold_label: str
    threshold_note: str
    requests: int | None
    figures: list[Figure] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalRunView:
    file: str
    date: str
    gold_version: str
    figures: list[Figure] = field(default_factory=list)
    caveat: str = ""


def _load(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _of(count: int | None, total: int | None) -> str:
    return "—" if count is None or total is None else f"{count} of {total}"


def _pct(count: int | None, total: int | None) -> str:
    if count is None or not total:
        return ""
    return f"{100 * count / total:.1f}%"


def _answers_view(path: Path, data: dict) -> AnswersRunView | None:
    metrics = data.get("metrics")
    config = data.get("config") or {}
    if not isinstance(metrics, dict):
        return None

    verification = metrics.get("verification") or {}
    abstention = metrics.get("abstention") or {}
    outcomes = metrics.get("outcome_counts") or {}
    asked = verification.get("asked_count")
    unverified = verification.get("unverified")
    verified = asked - unverified if asked is not None and unverified is not None else None
    total_questions = metrics.get("total_question_count") or config.get("question_count")
    tp = abstention.get("true_positives")
    fp = abstention.get("false_positives")
    expected_abstain = abstention.get("positive_count")
    answerable = (
        total_questions - expected_abstain if total_questions and expected_abstain is not None else None
    )
    flip = f"{100 / expected_abstain:.0f} points" if expected_abstain else "many points"

    figures = [
        Figure(
            "Questions",
            str(total_questions),
            f"{config.get('dev_question_count', '?')} dev, {config.get('test_question_count', '?')} test",
        ),
        Figure(
            "Reached the model",
            _of(asked, total_questions),
            f"the other {outcomes.get('pre_llm_abstain', 0)} were stopped by the abstention threshold "
            "before the model was asked",
        ),
        Figure(
            "Grounded answers",
            _of(verified, asked),
            f"{_pct(verified, asked)} of the questions that reached the model; an answer that could "
            "not be verified counts as not grounded",
        ),
        Figure(
            "Rejected by the citation check",
            _of(unverified, asked),
            'the person saw "I couldn\'t verify this answer"; '
            f"{verification.get('verified_after_retry', 0)} others passed only on the retry",
        ),
        Figure(
            "Abstention recall",
            _of(tp, expected_abstain),
            f"questions the gold says have no answer in the documents; a small sample: one flip moves it by {flip}",
        ),
        Figure(
            "Abstention precision",
            _of(tp, tp + fp if tp is not None and fp is not None else None),
            "of all abstentions, how many were right",
        ),
        Figure(
            "Answerable questions abstained on",
            _of(fp, answerable),
            "by the threshold or by the model; these are answers a person would have wanted",
        ),
    ]

    threshold = config.get("abstention_threshold")
    source = config.get("abstention_threshold_source")
    threshold_note = (
        "Abstention threshold "
        + (f"{threshold:.4f}" if isinstance(threshold, (int, float)) else "not recorded")
        + (f": {source}" if source else ", calibrated on the dev split of the same 52 questions it is scored on")
        + ("" if source and "in-sample" in source else ". It is in-sample.")
        + (" The 52 questions it was calibrated on are the ones it is scored on." if source and "in-sample" in source else "")
    )

    return AnswersRunView(
        file=path.name,
        date=path.name[:10],
        model=str(config.get("llm_model", "?")),
        prompt_version=str(config.get("prompt_version") or "knowledge_answer_v1"),
        gold_version=str(config.get("gold_version") or GOLD_V1),
        gold_label=str(data.get("gold_label") or config.get("gold_label") or ""),
        threshold_note=threshold_note,
        requests=(config.get("model_requests") or {}).get("total"),
        figures=figures,
    )


def _retrieval_view(path: Path, data: dict) -> RetrievalRunView | None:
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None
    cross = metrics.get("cross_encoder") or {}
    plain = metrics.get("passthrough") or {}

    def pair(key: str, label: str) -> Figure:
        with_rerank = cross.get(key)
        without = plain.get(key)
        return Figure(
            label,
            f"{with_rerank:.4f}" if isinstance(with_rerank, (int, float)) else "—",
            f"with the reranker; {without:.4f} without it"
            if isinstance(without, (int, float))
            else "with the reranker",
        )

    return RetrievalRunView(
        file=path.name,
        date=path.name[:10],
        gold_version=str((data.get("config") or {}).get("gold_version") or GOLD_V1),
        figures=[
            Figure(
                "Questions scored",
                str(metrics.get("evaluated_question_count")),
                f"of {metrics.get('total_question_count')}; the rest expect no answer, so there is no chunk to find",
            ),
            pair("recall_at_5", "Recall@5"),
            pair("recall_at_10", "Recall@10"),
            pair("mrr", "MRR"),
            pair("ndcg_at_10", "nDCG@10"),
        ],
        caveat=(
            "30 chunks from 10 documents, so Recall@10 covers a third of the corpus and is easy. "
            "Questions and expected chunks were written by the same author against the same fixtures."
        ),
    )


def load_results(directory: Path | None = None) -> tuple[list[AnswersRunView], list[RetrievalRunView]]:
    """The answers runs and the retrieval runs in the committed results folder, newest first."""
    folder = directory or RESULTS_DIR
    answers: list[AnswersRunView] = []
    retrieval: list[RetrievalRunView] = []
    if not folder.is_dir():
        return answers, retrieval

    for path in sorted(folder.glob("*.json"), reverse=True):
        data = _load(path)
        if data is None:
            continue
        if data.get("suite") == "answers":
            answer_view = _answers_view(path, data)
            if answer_view is not None:
                answers.append(answer_view)
        elif data.get("suite") == "retrieval":
            retrieval_view = _retrieval_view(path, data)
            if retrieval_view is not None:
                retrieval.append(retrieval_view)
    return answers, retrieval
