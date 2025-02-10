from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fme.core.dataset.config import RepeatedInterval


def test_repeated_interval_int():
    interval = RepeatedInterval(interval_length=3, block_length=6, start=0)
    mask = interval.get_boolean_mask(length=18)
    expected_mask = np.array([True, True, True, False, False, False] * 3)
    np.testing.assert_array_equal(mask, expected_mask)


def test_repeated_interval_str():
    interval = RepeatedInterval(interval_length="1d", block_length="7d", start="2d")
    mask = interval.get_boolean_mask(length=21, timestep=timedelta(days=1))
    expected_mask = np.array([False, False, True, False, False, False, False] * 3)
    np.testing.assert_array_equal(mask, expected_mask)


def test_repeated_interval_mixed_types():
    with pytest.raises(ValueError):
        RepeatedInterval(interval_length=3, block_length="6d", start=0)


@pytest.mark.parametrize("interval, block, start", [(4, 6, 3), ("2d", "3d", "2d")])
def test_repeated_interval_invalid_interval_start(interval, block, start):
    """start + interval exceeds length of block"""
    interval = RepeatedInterval(
        interval_length=interval, block_length=block, start=start
    )
    with pytest.raises(ValueError):
        interval.get_boolean_mask(length=18, timestep=timedelta(days=1))


def test_repeated_interval_zero_length():
    interval = RepeatedInterval(interval_length=0, block_length=6, start=0)
    mask = interval.get_boolean_mask(length=18)
    expected_mask = np.array([False] * 18)
    np.testing.assert_array_equal(mask, expected_mask)


def test_repeated_interval_partial_block():
    interval = RepeatedInterval(interval_length=3, block_length=6, start=0)
    mask = interval.get_boolean_mask(length=20)
    expected_mask = np.array([True, True, True, False, False, False] * 3 + [True, True])
    np.testing.assert_array_equal(mask, expected_mask)


def test_repeated_interval_no_timestep_fails_for_timedelta_lengths():
    interval = RepeatedInterval(interval_length="1d", block_length="7d", start="0d")
    with pytest.raises(ValueError):
        interval.get_boolean_mask(length=20)


@pytest.mark.parametrize("timestep", ["2h", "150m", "5h", "12h"])
def test_invalid_timesteps(timestep):
    """
    Test that timesteps that don't evenly divide into some or all
    arguments raise a ValueError
    """
    timestep = pd.to_timedelta(timestep)
    with pytest.raises(ValueError):
        RepeatedInterval(
            interval_length="5h", start="4h", block_length="10h"
        ).get_boolean_mask(length=18, timestep=timestep)
