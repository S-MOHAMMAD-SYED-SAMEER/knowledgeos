"""The hard cap on model requests: counted before sending, kept on disk, shared across runs."""

import pytest

from app.providers.llm import LLMResult
from evals.answers.budget import BudgetedLLMProvider, RequestBudget, RequestCapReached
from evals.answers.suite import is_rate_limit


class Counting:
    model_name = "counting"

    def __init__(self) -> None:
        self.sent = 0

    def complete(self, *, system, user, max_tokens, json_schema=None):
        self.sent += 1
        return LLMResult(text="{}", input_tokens=1, output_tokens=1, finish_reason="STOP")


def test_the_request_after_the_cap_is_never_sent(tmp_path) -> None:
    inner = Counting()
    llm = BudgetedLLMProvider(inner, RequestBudget(3, tmp_path / "ledger.json"))
    for _ in range(3):
        llm.complete(system="s", user="u", max_tokens=1)
    with pytest.raises(RequestCapReached):
        llm.complete(system="s", user="u", max_tokens=1)
    assert inner.sent == 3


def test_the_count_is_kept_on_disk_so_a_second_run_cannot_start_over(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    first = BudgetedLLMProvider(Counting(), RequestBudget(4, ledger))
    first.complete(system="s", user="u", max_tokens=1)
    first.complete(system="s", user="u", max_tokens=1)

    second_budget = RequestBudget(4, ledger)  # a new process, the same ledger
    assert (second_budget.used, second_budget.remaining) == (2, 2)
    second = BudgetedLLMProvider(Counting(), second_budget)
    second.complete(system="s", user="u", max_tokens=1)
    second.complete(system="s", user="u", max_tokens=1)
    with pytest.raises(RequestCapReached):
        second.complete(system="s", user="u", max_tokens=1)


def test_a_request_that_fails_still_counts() -> None:
    class Failing:
        model_name = "failing"

        def complete(self, **_kwargs):
            raise RuntimeError("boom")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        budget = RequestBudget(5, Path(folder) / "ledger.json")
        llm = BudgetedLLMProvider(Failing(), budget)
        with pytest.raises(RuntimeError):
            llm.complete(system="s", user="u", max_tokens=1)
        assert budget.used == 1  # counted before it was sent


def test_hitting_the_cap_stops_the_suite_like_a_rate_limit_and_says_so() -> None:
    error = RequestCapReached("the request cap of 100 is spent (100 used); nothing was sent")
    assert is_rate_limit(error)
    assert "request cap" in str(error)
