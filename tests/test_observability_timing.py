"""`app/observability/timing.py`: `percentile()`'s arithmetic and
`stage_timer()`'s always-set-even-on-exception contract."""

import time

import pytest

from app.observability.timing import p50, p95, percentile, stage_timer


def test_percentile_of_a_single_value_is_that_value() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([42.0], 0) == 42.0


def test_percentile_interpolates_linearly() -> None:
    # Four ordered values 0, 10, 20, 30: p50 sits at rank 1.5 -> halfway
    # between values[1]=10 and values[2]=20.
    values = [30.0, 0.0, 20.0, 10.0]
    assert percentile(values, 50) == 15.0


def test_percentile_at_0_and_100_are_the_min_and_max() -> None:
    values = [5.0, 1.0, 9.0, 3.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 9.0


def test_percentile_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], -1)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_p50_and_p95_match_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert p50(values) == percentile(values, 50)
    assert p95(values) == percentile(values, 95)


def test_stage_timer_sets_a_positive_elapsed_ms_on_success() -> None:
    with stage_timer() as timer:
        time.sleep(0.001)
    assert timer.ms is not None
    assert timer.ms > 0


def test_stage_timer_ms_is_none_while_the_block_is_running() -> None:
    with stage_timer() as timer:
        assert timer.ms is None


def test_stage_timer_sets_ms_even_when_the_block_raises() -> None:
    timer_box = None
    with pytest.raises(RuntimeError):
        with stage_timer() as timer:
            timer_box = timer
            raise RuntimeError("boom")
    assert timer_box is not None
    assert timer_box.ms is not None
    assert timer_box.ms >= 0
