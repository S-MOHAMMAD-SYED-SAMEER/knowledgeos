"""Keeping what the model actually said, so a run can be scored again later without asking it again.

`RecordingLLMProvider` wraps a provider and remembers every `complete()` call that returned: the raw text
(before any parsing or validation), the token counts and the finish reason. The suite asks for the calls
made since it last asked (`take_calls`) after each question and stores them with that question's result,
so the saved report holds every raw answer, including the ones that were rejected and never shown.
"""

import time

from app.providers.llm import LLMProvider, LLMResult


class RecordingLLMProvider:
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self._calls: list[dict] = []
        # Every call made through this wrapper, including one that raised.
        self.total_calls = 0

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
        self.total_calls += 1
        started = time.monotonic()
        result = self._inner.complete(
            system=system, user=user, max_tokens=max_tokens, json_schema=json_schema
        )
        self._calls.append(
            {
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "finish_reason": result.finish_reason,
                "seconds": round(time.monotonic() - started, 3),
            }
        )
        return result

    def take_calls(self) -> list[dict]:
        """The calls that returned since the last time this was asked, oldest first."""
        calls, self._calls = self._calls, []
        return calls


__all__ = ["RecordingLLMProvider"]
