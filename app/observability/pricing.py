"""Per-model token pricing and cost computation (§14).

The specification's own words: "log … cost_usd from configurable
per-model pricing. Unknown pricing raises a config error — it must never
silently become zero." This module is that rule, made literal: computing
a cost for a model with no configured pricing raises `PricingError`
rather than returning `0.0` — a `0.0` would be indistinguishable from a
model that is genuinely free, which none of these are.

Pricing is configured, never invented: `Settings.llm_pricing_usd_per_million_tokens`
(`app/config.py`) defaults to an **empty** mapping. No entry is ever
written here for a model whose real published rate has not been verified
— "do not invent pricing for an unverified model" is a locked project
decision, not merely a style preference.
"""

from dataclasses import dataclass


class PricingError(RuntimeError):
    """No usable pricing is configured for a model.

    Raised for a model missing from the table entirely, and for an entry
    present but malformed — either way, the caller must not treat the
    cost as zero.
    """


@dataclass(frozen=True)
class ModelPricing:
    """USD per **million** tokens — the unit model pricing pages
    conventionally publish in, so a configured value can be checked
    against the vendor's own page without a unit conversion."""

    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


def get_pricing(model: str, table: dict[str, dict[str, float]]) -> ModelPricing:
    """Look up one model's configured pricing. Raises `PricingError` if
    the model is absent from `table`, or if its entry is present but
    missing/malformed `input`/`output` keys."""
    entry = table.get(model)
    if entry is None:
        raise PricingError(
            f"no configured pricing for model {model!r}; set "
            "KNOWLEDGEOS_LLM_PRICING_USD_PER_MILLION_TOKENS to add a "
            "verified entry rather than treating the cost as zero"
        )
    try:
        return ModelPricing(
            input_usd_per_million_tokens=float(entry["input"]),
            output_usd_per_million_tokens=float(entry["output"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PricingError(
            f"pricing entry for model {model!r} is malformed; expected "
            "{'input': <usd per 1M tokens>, 'output': <usd per 1M tokens>}"
        ) from exc


def compute_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    table: dict[str, dict[str, float]],
) -> float:
    """The cost of one request, from measured token counts and configured
    pricing only. Raises `PricingError` for an unconfigured model —
    callers decide what "unavailable" means for them (the serving path
    leaves `cost_usd` `NULL`; the evaluation suite refuses to report an
    official cost figure at all — see `evals/answers/report.py`)."""
    pricing = get_pricing(model, table)
    return (
        input_tokens / 1_000_000 * pricing.input_usd_per_million_tokens
        + output_tokens / 1_000_000 * pricing.output_usd_per_million_tokens
    )


def projected_monthly_cost_usd(
    *, mean_cost_per_query_usd: float, queries_per_day: int
) -> float:
    """§14: "projected monthly cost at 100/day and 1,000/day, only from
    measured token counts." `mean_cost_per_query_usd` must itself already
    be an average of real `compute_cost_usd` results — this function
    performs no measurement of its own, only the projection arithmetic
    (a fixed 30-day month, the plainest reading of "monthly" the
    specification leaves undefined further than that).
    """
    if queries_per_day < 0:
        raise ValueError("queries_per_day must not be negative")
    return mean_cost_per_query_usd * queries_per_day * 30


__all__ = [
    "ModelPricing",
    "PricingError",
    "compute_cost_usd",
    "get_pricing",
    "projected_monthly_cost_usd",
]
