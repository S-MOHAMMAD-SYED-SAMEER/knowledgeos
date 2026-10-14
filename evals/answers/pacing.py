"""Keeping an answers run under a model's rate limit, and saying what it will cost before it starts.

The answers suite calls the model once per question (twice when a reply has to be retried).
Left alone it goes as fast as the model answers, a few seconds per call, which is faster than a
free tier allows. `PacedLLMProvider` spaces the calls; `describe_call_budget` tells the person
what they are about to spend, while they can still stop.

Pacing counts calls to `complete()`. The adapter's own retries inside one call (after a 429 or a
timeout) are not counted, and they wait on their own, so a run can briefly exceed the configured
rate when the service is struggling. That is the adapter's back-off doing its job.
"""

import time
from collections.abc import Callable

from app.providers.llm import LLMProvider, LLMResult

# Under the 15 requests a minute the Gemini free tier allowed when this was measured, with room
# for the adapter's own retries. Not a promise about any model's limit: check yours.
DEFAULT_REQUESTS_PER_MINUTE = 10.0


class PacedLLMProvider:
    """Wraps a provider so that `complete()` calls start at least `60 / requests_per_minute` seconds
    apart. The first call is not delayed."""

    def __init__(
        self,
        inner: LLMProvider,
        *,
        requests_per_minute: float = DEFAULT_REQUESTS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._inner = inner
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None
        self.calls = 0

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
        if self._last_start is not None:
            wait = self._last_start + self._interval - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_start = self._clock()
        self.calls += 1
        return self._inner.complete(
            system=system, user=user, max_tokens=max_tokens, json_schema=json_schema
        )


def describe_call_budget(*, question_count: int, already_done: int, requests_per_minute: float) -> str:
    """What a run will ask of the model, in words, before it starts.

    One canary call, then one call per question still to do; up to twice that if every first reply
    is rejected and retried. The time is the lower bound the pacing imposes.
    """
    remaining = question_count - already_done
    likely = 1 + remaining
    worst = 1 + 2 * remaining
    minutes = likely / requests_per_minute
    done = f" ({already_done} already done, resuming)" if already_done else ""
    return (
        f"Call budget: about {likely} model calls{done}: 1 availability check + {remaining} question(s), "
        f"up to {worst} if every first reply is rejected and retried. "
        f"Paced at {requests_per_minute:g} a minute, that is at least {minutes:.1f} minutes. "
        "Check that your quota allows it; a daily limit stops a run part-way (it can be resumed)."
    )


__all__ = [
    "DEFAULT_REQUESTS_PER_MINUTE",
    "PacedLLMProvider",
    "describe_call_budget",
]
