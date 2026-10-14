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


def test_the_readme_does_not_claim_answer_quality_was_measured() -> None:
    readme = _text(ROOT / "README.md")

    assert "No answer-quality number exists" in readme
