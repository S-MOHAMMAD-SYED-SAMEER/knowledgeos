"""Stage timing and latency percentiles (§13: "mean latency (p50/p95
broken out by stage)"; §14: "log stage latencies").

`stage_timer()` measures one stage's wall-clock duration; `percentile()`
is the pure function both the serving path's own logging and
`evals/answers/`'s suite-wide aggregation call, so "p95" means the same
arithmetic everywhere it is reported.

No provider call, no I/O, no database — timing a stage that already
happened cannot itself fail, so this module has no error type.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _Elapsed:
    """A mutable box: `ms` is `None` while the timed block is still
    running, and set once it exits — success or exception, timing is not
    conditional on the block succeeding."""

    ms: float | None = None


@contextmanager
def stage_timer() -> Iterator[_Elapsed]:
    """Time one stage. Usage:

        with stage_timer() as timer:
            do_the_stage()
        stage_ms = timer.ms

    `time.perf_counter()` is a monotonic clock meant exactly for measuring
    elapsed intervals — never wall-clock time-of-day, which can jump.
    """
    box = _Elapsed()
    start = time.perf_counter()
    try:
        yield box
    finally:
        box.ms = (time.perf_counter() - start) * 1000.0


def percentile(values: list[float], p: float) -> float:
    """The `p`-th percentile of `values` (0 <= p <= 100), by linear
    interpolation between the two closest ranks — the same default method
    `numpy.percentile` and most statistics packages use, chosen so a
    result here matches what an operator would get cross-checking it
    independently.

    Raises `ValueError` for an empty list or an out-of-range `p`, rather
    than returning a number that would silently mean nothing.
    """
    if not values:
        raise ValueError("percentile of an empty list is undefined")
    if not (0 <= p <= 100):
        raise ValueError(f"p must be between 0 and 100, got {p}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (p / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def p50(values: list[float]) -> float:
    return percentile(values, 50)


def p95(values: list[float]) -> float:
    return percentile(values, 95)


__all__ = ["p50", "p95", "percentile", "stage_timer"]
