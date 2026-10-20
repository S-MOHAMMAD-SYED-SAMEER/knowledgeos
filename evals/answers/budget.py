"""A hard cap on how many model requests an evaluation may make, kept on disk.

`python -m evals.run --suite answers --request-cap 100` refuses to send request number 101. The count
lives in a small ledger file, not in the process, so it survives a run that is interrupted and is shared
by every run that points at the same ledger: a cap of 100 is 100 for the whole piece of work, not 100
per command.

The request is counted BEFORE it is sent, so a request that hangs, fails or is never answered still
counts, and a crash cannot make the ledger lower than what was really sent. Nothing here retries.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from app.providers.llm import LLMProvider, LLMRateLimited, LLMResult

DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "var" / "request_ledger.json"


class RequestCapReached(LLMRateLimited):
    """The cap is spent. Raised instead of sending. A subclass of the rate-limit error so the suite
    stops at once and does not carry on to the next question."""


class RequestBudget:
    def __init__(self, limit: int, path: Path | None = None) -> None:
        self.limit = limit
        self.path = path or DEFAULT_LEDGER

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"limit": self.limit, "entries": []}

    @property
    def used(self) -> int:
        return len(self._read()["entries"])

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reserve(self, label: str) -> int:
        """Count one request and return how many have been used, or raise if the cap is spent."""
        ledger = self._read()
        used = len(ledger["entries"])
        if used >= self.limit:
            raise RequestCapReached(
                f"the request cap of {self.limit} is spent ({used} used); nothing was sent"
            )
        ledger["limit"] = self.limit
        ledger["entries"].append({"at": datetime.now(UTC).isoformat(timespec="seconds"), "label": label})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
        return used + 1


class BudgetedLLMProvider:
    """Counts every `complete()` call against a budget before the inner provider sees it."""

    def __init__(self, inner: LLMProvider, budget: RequestBudget, label: str = "") -> None:
        self._inner = inner
        self._budget = budget
        self._label = label

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        json_schema: dict | None = None,
    ) -> LLMResult:
        self._budget.reserve(self._label)
        return self._inner.complete(
            system=system, user=user, max_tokens=max_tokens, json_schema=json_schema
        )


__all__ = ["BudgetedLLMProvider", "DEFAULT_LEDGER", "RequestBudget", "RequestCapReached"]
