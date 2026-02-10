from unittest.mock import patch

import pytest
import torch

from fme.core.benchmark.timer import CUDATimer


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
    # get cuda to wait
    with timer.context("parent"):
        torch.cuda._sleep(100_000)
        child_timer = timer.child("child")
        with child_timer.context("child_time"):
            torch.cuda._sleep(100_000)
    report = timer.report()
    assert "parent" in report.average_time_seconds
    assert "child" in report.children
    assert "child_time" in report.children["child"].average_time_seconds
    # parent time should include the child time, so it should be
    # at least 2x the child time (since we sleep for the same amount of time in both)
    assert (
        report.average_time_seconds["parent"]
        >= 2.0 * report.children["child"].average_time_seconds["child_time"]
    )
