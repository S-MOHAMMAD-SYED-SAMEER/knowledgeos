"""Observability: per-query stage timing, token accounting, and cost
(§14).

Two independent concerns, kept in separate modules because they fail
independently: `timing.py` measures wall-clock milliseconds and can never
fail (a stopwatch has no failure mode); `pricing.py` computes a dollar
figure from measured tokens and configured pricing, and refuses outright
— rather than guessing zero — when a model's pricing has not been
configured. The specification's own words: "Unknown pricing raises a
config error — it must never silently become zero."

Nothing here calls a provider or touches the database. `app/api/query.py`
times the stages of one request and hands the results to
`app/generation/persistence.py` to record; `evals/answers/` uses the same
two modules to compute the suite-wide latency percentiles and cost
projections §13/§14 ask for. One set of functions, two callers, so a
served query's numbers and an evaluation run's numbers can never disagree
about what "cost" or "p95" means.
"""

__all__: list[str] = []
