from typing import Literal
from unittest.mock import patch

import pytest
import torch

from fme.core.benchmark.timer import CUDATimer, TimerResult


@pytest.mark.parametrize("is_available", [True, False])
def test_new_if_available(is_available: bool):
    from fme.core.benchmark.timer import CUDATimer, NullTimer

    with patch("torch.cuda.is_available", return_value=is_available):
        timer = CUDATimer.new_if_available()
    if is_available:
        assert isinstance(timer, CUDATimer)
    else:
        assert isinstance(timer, NullTimer)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available, skipping CUDATimer tests.",
)
def test_timer_with_child():
    timer = CUDATimer()
    with timer:
        # get cuda to wait
        torch.cuda._sleep(100_000)
        with timer.child("child"):
            torch.cuda._sleep(100_000)
    result = timer.result
    assert "child" in result.children
    # parent time should include the child time, so it should be
    # at least 2x the child time (since we sleep for the same amount of time in both)
    assert result.avg_time >= 2.0 * result.children["child"].avg_time


def _create_parent_result(avg_time: float) -> TimerResult:
    return TimerResult(count=2, avg_time=avg_time, children={})


def _create_child_result(avg_time: float) -> TimerResult:
    return TimerResult(
        count=2,
        avg_time=1.0,
        children={"child": TimerResult(count=2, avg_time=avg_time, children={})},
    )


@pytest.mark.parametrize(
    "v1, v2, rtol, expect_raise",
    [
        (100, 101, 0.02, False),  # within 2%
        (100, 103, 0.02, True),  # outside 2%
        (100, 102, 0.02, False),  # exactly 2% is considered inside
        (10000, 10201, 0.02, True),  # more than 2% is considered outside
        (100, 102, 0.03, False),  # exactly 2% is within 3%
    ],
)
@pytest.mark.parametrize("kind", ["parent", "child"])
def test_assert_close(
    v1: int, v2: int, rtol: float, kind: Literal["parent", "child"], expect_raise: bool
):
    if kind == "child":
        result1 = _create_child_result(avg_time=v1)
        result2 = _create_child_result(avg_time=v2)
    else:
        result1 = _create_parent_result(avg_time=v1)
        result2 = _create_parent_result(avg_time=v2)
    if expect_raise:
        with pytest.raises(AssertionError):
            result2.assert_close(result1, rtol=rtol)
    else:
        result2.assert_close(result1, rtol=rtol)


def test_assert_close_different_count():
    # different count should raise regardless of rtol
    result1 = TimerResult(count=100, avg_time=100.0, children={})
    result2 = TimerResult(count=101, avg_time=100.0, children={})
    with pytest.raises(AssertionError):
        result2.assert_close(result1, rtol=0.5)


def test_assert_close_children_rtol():
    # test that children_rtol is used for child comparisons
    result1 = TimerResult(
        count=2,
        avg_time=100.0,
        children={"child": TimerResult(count=2, avg_time=100.0, children={})},
    )
    result2 = TimerResult(
        count=2,
        avg_time=110.0,
        children={"child": TimerResult(count=2, avg_time=103.0, children={})},
    )
    result2.assert_close(result1, rtol=0.2, children_rtol=0.05)


def test_assert_close_children_rtol_raises():
    # test that children_rtol is used for child comparisons
    result1 = TimerResult(
        count=2,
        avg_time=100.0,
        children={"child": TimerResult(count=2, avg_time=100.0, children={})},
    )
    result2 = TimerResult(
        count=2,
        avg_time=110.0,
        children={"child": TimerResult(count=2, avg_time=103.0, children={})},
    )
    with pytest.raises(AssertionError):
        result2.assert_close(result1, rtol=0.05, children_rtol=0.2)
