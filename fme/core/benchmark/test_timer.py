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
    report = timer.result()
    assert "child" in report.children
    # parent time should include the child time, so it should be
    # at least 2x the child time (since we sleep for the same amount of time in both)
    assert report.avg_time >= 2.0 * report.children["child"].avg_time


def test_combine_two_timer_results():
    result1 = TimerResult(
        avg_time=1.0, children={"a": TimerResult(avg_time=0.5, children={})}
    )
    result2 = TimerResult(
        avg_time=1.5, children={"a": TimerResult(avg_time=0.7, children={})}
    )
    combined = result1.combine(result2)
    torch.testing.assert_close(combined.avg_time, 1.25)
    assert set(combined.children.keys()) == {"a"}
    torch.testing.assert_close(combined.children["a"].avg_time, 0.6)


def test_combine_three_timer_results():
    result1 = TimerResult(
        avg_time=1.0, children={"a": TimerResult(avg_time=0.5, children={})}
    )
    result2 = TimerResult(
        avg_time=1.5, children={"a": TimerResult(avg_time=0.7, children={})}
    )
    result3 = TimerResult(
        avg_time=2.0, children={"a": TimerResult(avg_time=0.9, children={})}
    )
    combined = result1.combine(result2).combine(result3)
    torch.testing.assert_close(combined.avg_time, 1.5)
    assert set(combined.children.keys()) == {"a"}
    torch.testing.assert_close(combined.children["a"].avg_time, 0.7)


def test_combine_timer_results_different_children():
    result1 = TimerResult(
        avg_time=1.0, children={"a": TimerResult(avg_time=0.5, children={})}
    )
    result2 = TimerResult(
        avg_time=1.5, children={"b": TimerResult(avg_time=0.7, children={})}
    )
    with pytest.raises(
        ValueError, match="Cannot combine TimerResults with different children"
    ):
        result1.combine(result2)
