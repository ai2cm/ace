from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from fme.ace.data_loading.config import DataLoaderConfig, GroupWeightsConfig
from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.schedule import WeightMilestone
from fme.core.dataset.time import RepeatedInterval
from fme.core.dataset.xarray import XarrayDataConfig


def _concat_config(n_members: int) -> ConcatDatasetConfig:
    return ConcatDatasetConfig(
        concat=[
            XarrayDataConfig(data_path=f"/does/not/exist/{i}") for i in range(n_members)
        ]
    )


def test_group_weights_config_round_trip():
    group_weights = GroupWeightsConfig(
        groups=[2, 1],
        start_value=[0.3, 0.7],
        milestones=[WeightMilestone(epoch=143, value=[0.0, 1.0])],
    )
    config = DataLoaderConfig(
        dataset=_concat_config(3),
        batch_size=1,
        group_weights=group_weights,
    )
    assert config.group_weights is group_weights
    assert config.group_weights.schedule.get_value(0) == [0.3, 0.7]
    assert config.group_weights.schedule.get_value(143) == [0.0, 1.0]


def test_group_weights_config_sum_mismatch_raises():
    with pytest.raises(ValueError, match="number of concat members"):
        DataLoaderConfig(
            dataset=_concat_config(3),
            batch_size=1,
            group_weights=GroupWeightsConfig(groups=[2, 2], start_value=[0.5, 0.5]),
        )


def test_group_weights_config_with_sample_with_replacement_raises():
    with pytest.raises(ValueError, match="cannot be combined"):
        DataLoaderConfig(
            dataset=_concat_config(2),
            batch_size=1,
            sample_with_replacement=10,
            group_weights=GroupWeightsConfig(groups=[1, 1], start_value=[0.5, 0.5]),
        )


def test_group_weights_config_non_concat_dataset_raises():
    with pytest.raises(ValueError, match="ConcatDatasetConfig"):
        DataLoaderConfig(
            dataset=XarrayDataConfig(data_path="/does/not/exist"),
            batch_size=1,
            group_weights=GroupWeightsConfig(groups=[1], start_value=[1.0]),
        )


def test_group_weights_config_non_positive_groups_raises():
    with pytest.raises(ValueError, match="positive"):
        GroupWeightsConfig(groups=[0, 2], start_value=[0.5, 0.5])


def test_group_weights_config_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        GroupWeightsConfig(groups=[1, 1], start_value=[1.0])


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
