"""`app/observability/pricing.py`: configured pricing, unknown-model
failure, cost computation, and monthly projection arithmetic (§14, D7).

An unconfigured model must raise `PricingError`, never return `0.0` — the
specification's own words: "unknown pricing raises a config error." Every
test here that expects a real cost value configures the table explicitly;
none rely on a default that could silently be zero.
"""

import pytest

from app.observability.pricing import (
    ModelPricing,
    PricingError,
    compute_cost_usd,
    get_pricing,
    projected_monthly_cost_usd,
)


def test_get_pricing_returns_the_configured_rates() -> None:
    table = {"gemini-test": {"input": 1.0, "output": 2.0}}
    pricing = get_pricing("gemini-test", table)
    assert pricing == ModelPricing(input_usd_per_million_tokens=1.0, output_usd_per_million_tokens=2.0)


def test_get_pricing_raises_for_an_unconfigured_model() -> None:
    with pytest.raises(PricingError):
        get_pricing("unknown-model", {})


def test_get_pricing_raises_for_a_malformed_entry() -> None:
    with pytest.raises(PricingError):
        get_pricing("bad-model", {"bad-model": {"input": "not-a-number"}})
    with pytest.raises(PricingError):
        get_pricing("missing-output", {"missing-output": {"input": 1.0}})


def test_compute_cost_usd_is_linear_in_tokens() -> None:
    table = {"m": {"input": 1.0, "output": 2.0}}
    # 1,000,000 input tokens at $1/1M + 500,000 output tokens at $2/1M.
    cost = compute_cost_usd(model="m", input_tokens=1_000_000, output_tokens=500_000, table=table)
    assert cost == pytest.approx(1.0 + 1.0)


def test_compute_cost_usd_zero_tokens_is_zero_cost() -> None:
    table = {"m": {"input": 5.0, "output": 5.0}}
    assert compute_cost_usd(model="m", input_tokens=0, output_tokens=0, table=table) == 0.0


def test_compute_cost_usd_never_silently_zero_for_unconfigured_model() -> None:
    with pytest.raises(PricingError):
        compute_cost_usd(model="unconfigured", input_tokens=100, output_tokens=100, table={})


def test_projected_monthly_cost_usd_is_a_fixed_30_day_month() -> None:
    assert projected_monthly_cost_usd(mean_cost_per_query_usd=0.01, queries_per_day=100) == pytest.approx(
        0.01 * 100 * 30
    )
    assert projected_monthly_cost_usd(mean_cost_per_query_usd=0.01, queries_per_day=1_000) == pytest.approx(
        0.01 * 1_000 * 30
    )


def test_projected_monthly_cost_usd_rejects_negative_queries_per_day() -> None:
    with pytest.raises(ValueError):
        projected_monthly_cost_usd(mean_cost_per_query_usd=0.01, queries_per_day=-1)


def test_projected_monthly_cost_usd_zero_queries_per_day_is_zero() -> None:
    assert projected_monthly_cost_usd(mean_cost_per_query_usd=0.01, queries_per_day=0) == 0.0
