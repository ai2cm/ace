import datetime

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.dummy import DummyDataset
from fme.core.dataset.schedule import IntSchedule


def _make_dummy_dataset(
    n_total_times: int = 10,
    n_timesteps: int = 3,
) -> DummyDataset:
    start_time = cftime.DatetimeGregorian(2000, 1, 1)
    end_time = start_time + datetime.timedelta(days=n_total_times - 1)
    timestep = datetime.timedelta(days=1)
    schedule = IntSchedule.from_constant(n_timesteps)
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.Tensor(np.arange(12)),
        lon=torch.Tensor(np.arange(6)),
    )
    return DummyDataset(
        start_time=start_time,
        end_time=end_time,
        timestep=timestep,
        n_timesteps=schedule,
        horizontal_coordinates=horizontal_coordinates,
    )


def test_dummy_dataset_has_expected_information():
    dataset = _make_dummy_dataset(n_total_times=10, n_timesteps=3)
    assert isinstance(dataset.all_times, xr.CFTimeIndex)
    assert len(dataset.all_times) == 10
    assert dataset.all_times[0] == cftime.DatetimeGregorian(2000, 1, 1)
    assert dataset.all_times[-1] == cftime.DatetimeGregorian(2000, 1, 10)
    assert len(dataset) == 8
    assert dataset[0][0]["__dummy__"].shape == (3, 12, 6)


def test_dummy_dataset_getitem_truncates_near_end():
    dataset = _make_dummy_dataset(n_total_times=10, n_timesteps=7)
    # 10 times total, sample_n_times=7, so valid full-window indices are 0..3
    assert len(dataset) == 4
    data, time, _, _ = dataset[0]
    assert data["__dummy__"].shape[0] == 7
    assert len(time) == 7

    # Access beyond the last valid index — the last window is truncated
    data, time, _, _ = dataset[5]
    assert len(time) == 5
    assert data["__dummy__"].shape[0] == 5
