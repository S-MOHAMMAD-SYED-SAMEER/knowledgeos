"""The numbers the README and the results page quote are the numbers in the committed run.

A figure typed into prose can drift from the run it came from. These tests read the committed
JSON and check that every figure the documents print is that run's, to the same four decimals,
so a re-run that changes the result cannot leave an old number on the README.
"""

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = sorted((ROOT / "docs" / "results").glob("*-retrieval-eval.json"))


def _text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_there_is_a_committed_retrieval_run() -> None:
    assert RESULTS, "docs/results/ has no *-retrieval-eval.json"


@pytest.mark.parametrize("run_path", RESULTS, ids=lambda p: p.name)
def test_the_committed_run_is_a_complete_retrieval_run(run_path: pathlib.Path) -> None:
    run = json.loads(_text(run_path))

    assert run["suite"] == "retrieval"
    assert run["config"]["question_count"] == 52
    assert run["config"]["corpus_active_chunks"] == 30
    assert run["config"]["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert run["config"]["cross_encoder_rerank_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    metrics = run["metrics"]
    assert metrics["evaluated_question_count"] + metrics["excluded_question_count"] == 52
    for condition in ("passthrough", "cross_encoder"):
        assert set(metrics[condition]) == {"recall_at_5", "recall_at_10", "precision_at_5", "mrr", "ndcg_at_10"}


@pytest.mark.parametrize("run_path", RESULTS, ids=lambda p: p.name)
def test_the_readme_and_the_results_page_quote_the_committed_figures(run_path: pathlib.Path) -> None:
    metrics = json.loads(_text(run_path))["metrics"]
    page = _text(run_path.with_suffix(".md"))
    readme = _text(ROOT / "README.md")

    for condition in ("passthrough", "cross_encoder"):
        for name in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10", "precision_at_5"):
            figure = f"{metrics[condition][name]:.4f}"
            assert figure in page, f"{run_path.stem}: {condition} {name} {figure} is not on the results page"
            if name != "precision_at_5" or condition == "cross_encoder":
                assert figure in readme, f"{condition} {name} {figure} is not in the README"
    assert f"{metrics['metadata_filter_correctness']:.4f}" in readme


@pytest.mark.parametrize("run_path", RESULTS, ids=lambda p: p.name)
def test_the_readme_links_to_the_results_page_and_the_page_exists(run_path: pathlib.Path) -> None:
    page = run_path.with_suffix(".md")

    assert page.exists()
    assert f"docs/results/{page.name}" in _text(ROOT / "README.md")


ANSWER_RUNS = {
    "v1": ROOT / "docs" / "results" / "2026-10-20-answers-eval-v1-gemini-3.5-flash-lite.json",
    "v2": ROOT / "docs" / "results" / "2026-10-20-answers-eval-v2-gemini-3.5-flash-lite.json",
}


def _figures(run: dict) -> dict[str, str]:
    """The counts the README and EVAL.md print for one answers run, as printed ("29 of 33")."""
    metrics = run["metrics"]
    verification = metrics["verification"]
    abstention = metrics["abstention"]
    asked, unverified = verification["asked_count"], verification["unverified"]
    tp, fp = abstention["true_positives"], abstention["false_positives"]
    return {
        "grounded": f"{asked - unverified} of {asked}",
        "rejected": f"{unverified} of {asked}",
        "recall": f"{tp} of {abstention['positive_count']}",
        "precision": f"{tp} of {tp + fp}",
        "abstained_on_answerable": f"{fp} of {metrics['total_question_count'] - abstention['positive_count']}",
    }


@pytest.mark.parametrize("name", sorted(ANSWER_RUNS))
def test_the_committed_answers_runs_are_on_the_frozen_gold_and_within_the_cap(name: str) -> None:
    run = json.loads(_text(ANSWER_RUNS[name]))
    config = run["config"]

    assert run["suite"] == "answers"
    assert config["gold_version"] == "gold-v2"
    assert config["gold_sha256"] in _text(ROOT / "docs" / "EVAL.md")
    assert config["llm_model"] == "gemini-3.5-flash-lite"
    assert config["llm_temperature"] == 1.0 and config["llm_max_retries"] == 0
    assert config["abstention_threshold"] == 1.4451 and config["abstention_threshold_in_sample"] is True
    assert config["prompt_version"] == f"knowledge_answer_{name}"
    assert run["metrics"]["total_question_count"] == 52
    assert run["metrics"]["provider_failures"] in (0, None) or not run["provider_errors"]


@pytest.mark.parametrize("name", sorted(ANSWER_RUNS))
def test_the_readme_and_eval_md_quote_the_committed_answers_figures(name: str) -> None:
    figures = _figures(json.loads(_text(ANSWER_RUNS[name])))
    readme = _text(ROOT / "README.md")
    eval_md = _text(ROOT / "docs" / "EVAL.md")
    report = _text(ROOT / "docs" / "results" / "2026-10-20-answers-eval.md")

    for key, figure in figures.items():
        for where, text in (("README", readme), ("EVAL.md", eval_md), ("results page", report)):
            assert figure in text, f"{name}: {key} '{figure}' is not in the {where}"


def test_the_two_answers_runs_used_no_more_than_the_request_cap() -> None:
    used = sum(json.loads(_text(path))["config"]["model_requests"]["total"] for path in ANSWER_RUNS.values())
    stopped_attempt = 2  # the first v2 attempt: the availability check and the question that returned a 503
    assert used + stopped_attempt == 78 <= 100
    for text in (_text(ROOT / "README.md"), _text(ROOT / "docs" / "EVAL.md")):
        assert "78 of the 100" in " ".join(text.split())


def test_the_readme_states_the_limits_beside_the_answer_figures() -> None:
    readme = _text(ROOT / "README.md")

    for phrase in ("in-sample", "Claude-drafted", "approved by Sameer", "one run per prompt", "no unsupported-claim rate"):
        assert phrase in readme, phrase
    assert "No answer-quality number exists" not in readme
    assert "no Gemini call has succeeded" not in readme
