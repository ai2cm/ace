import logging
import time

import numpy as np
import pytest

from fme.core.timing import CumulativeTimer, GlobalTimer


def test_CumulativeTimer():
    category = "foo"
    cumulative_timer = CumulativeTimer(category)

    cumulative_timer.start()
    time.sleep(0.01)
    cumulative_timer.stop()

    time.sleep(0.01)

    cumulative_timer.start()
    time.sleep(0.01)
    cumulative_timer.stop()

    assert cumulative_timer.duration == pytest.approx(0.02, abs=0.005)


def test_CumulativeTimer_start_error():
    category = "foo"
    cumulative_timer = CumulativeTimer(category)
    cumulative_timer.start()
    with pytest.raises(RuntimeError, match=f"timer {category!r} is already running"):
        cumulative_timer.start()


def test_CumulativeTimer_stop_error():
    category = "foo"
    cumulative_timer = CumulativeTimer(category)
    with pytest.raises(
        RuntimeError, match=f"must call start for timer {category!r} before stop"
    ):
        cumulative_timer.stop()


def test_CumulativeTimer_duration_error():
    category = "foo"
    cumulative_timer = CumulativeTimer(category)
    cumulative_timer.start()
    with pytest.raises(RuntimeError, match=f"timer {category!r} is still running"):
        cumulative_timer.duration


def exercise_active_timer():
    timer = GlobalTimer.get_instance()

    timer.start("foo")
    time.sleep(0.01)
    timer.stop()

    timer = GlobalTimer.get_instance()

    timer.start("bar")
    time.sleep(0.02)
    timer.stop()

    assert timer.get_duration("foo") == pytest.approx(0.01, abs=0.005)
    assert timer.get_duration("bar") == pytest.approx(0.02, abs=0.005)

    with pytest.raises(KeyError):
        timer.get_duration("baz")

    durations = timer.get_durations()
    assert durations["foo"] == pytest.approx(0.01, abs=0.005)
    assert durations["bar"] == pytest.approx(0.02, abs=0.005)


def test_GlobalTimer():
    with GlobalTimer():
        exercise_active_timer()

    # Check that timer is reset within new context.
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        assert timer.get_durations() == {}


def test_GlobalTimer_resets_after_exception():
    with pytest.raises(ValueError):
        with GlobalTimer():
            timer = GlobalTimer.get_instance()
            timer.start("foo")
            raise ValueError()

    # Check that the context manager clears the state of the timer after an
    # exception. If it were not clear, starting the timer for "foo" would raise
    # an error.
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        timer.start("foo")
        timer.stop()


def test_GlobalTimer_multiple_context_error():
    with pytest.raises(RuntimeError, match="GlobalTimer is currently in use"):
        with GlobalTimer(), GlobalTimer():
            pass


def test_inactive_GlobalTimer_warning():
    with pytest.warns(UserWarning, match=r"inactive"):
        GlobalTimer.get_instance()


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_inactive_GlobalTimer_start():
    timer = GlobalTimer.get_instance()
    timer.start("foo")


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_inactive_GlobalTimer_stop():
    timer = GlobalTimer.get_instance()
    timer.stop()


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_inactive_GlobalTimer_get_duration():
    timer = GlobalTimer.get_instance()
    result = timer.get_duration("foo")
    assert np.isnan(result)


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_inactive_GlobalTimer_get_durations():
    timer = GlobalTimer.get_instance()
    result = timer.get_durations()
    assert result == {}


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_inactive_GlobalTimer_log_durations(caplog):
    timer = GlobalTimer.get_instance()
    with caplog.at_level(logging.INFO):
        timer.log_durations()
    assert len(caplog.records) == 0


@pytest.mark.filterwarnings("ignore:The GlobalTimer")
def test_GlobalTimer_inactive_then_active():
    # Make sure we can instantiate an inactive timer, but then still create and
    # use an active timer later.
    GlobalTimer.get_instance()

    with GlobalTimer():
        exercise_active_timer()


def test_GlobalTimer_context():
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        with timer.context("foo"):
            time.sleep(0.01)
    assert timer.get_duration("foo") > 0.01


def test_GlobalTimer_context_with_exception():
    with pytest.raises(ValueError):
        with GlobalTimer():
            timer = GlobalTimer.get_instance()
            with timer.context("foo"):
                time.sleep(0.01)
                raise ValueError()
    assert timer.get_duration("foo") > 0.01


def test_GlobalTimer_single_inner_timer():
    with pytest.raises(
        RuntimeError, match="GlobalTimer already has an active inner timer"
    ):
        with GlobalTimer():
            timer = GlobalTimer.get_instance()
            with timer.context("foo"):
                with timer.context("bar"):
                    pass


def test_GlobalTimer_nested_outer_context():
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        with timer.outer_context("foo"):
            with timer.context("bar"):
                pass


def test_GlobalTimer_double_nested_outer_context():
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        with timer.outer_context("foo"):
            with timer.outer_context("bar"):
                pass
