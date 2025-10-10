import datetime

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.dummy import DummyDataset


def test_dummy_dataset_has_expected_information():
    start_time = cftime.DatetimeGregorian(2000, 1, 1)
    end_time = cftime.DatetimeGregorian(2000, 1, 10)
    timestep = datetime.timedelta(days=1)
    n_timesteps = 3
    horizontal_coordinates = LatLonCoordinates(
        lat=torch.Tensor(np.arange(12)),
        lon=torch.Tensor(np.arange(6)),
    )
    dataset = DummyDataset(
        start_time=start_time,
        end_time=end_time,
        timestep=timestep,
        n_timesteps=n_timesteps,
        horizontal_coordinates=horizontal_coordinates,
    )
    assert isinstance(dataset.all_times, xr.CFTimeIndex)
    assert len(dataset.all_times) == 10
    assert dataset.all_times[0] == start_time
    assert dataset.all_times[-1] == end_time
    assert len(dataset) == 8
    assert dataset[0][0]["__dummy__"].shape == (3, 12, 6)
