import datetime

import cftime
import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data.config import PairedDataLoaderConfig
from fme.downscaling.data.datasets import (
    PairedVideoBatchData,
    VideoBatchData,
    VideoBatchItem,
)
from fme.downscaling.data.time_encoding import compute_calendar_features
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.test_utils import data_paths_helper


def _times(n, start_hour=0, step_hours=3):
    base = cftime.DatetimeProlepticGregorian(2013, 1, 2, start_hour)
    vals = [base + datetime.timedelta(hours=step_hours * i) for i in range(n)]
    return xr.DataArray(vals, dims=["time"])


def _video_item(n_times=9, lat=4, lon=5, varnames=("x",)):
    data = {v: torch.rand(n_times, lat, lon) for v in varnames}
    time = _times(n_times)
    coords = LatLonCoordinates(
        lat=torch.linspace(0.0, 90.0, lat), lon=torch.linspace(0.0, 360.0, lon)
    )
    day_of_year, second_of_day = compute_calendar_features(time)
    return VideoBatchItem(data, time, coords, day_of_year, second_of_day)


def test_compute_calendar_features_24h_clip():
    # 9 frames at 0,3,...,24h starting Jan 2 2013 00:00 -> crosses into Jan 3.
    time = _times(9, start_hour=0, step_hours=3)
    day_of_year, second_of_day = compute_calendar_features(time)
    assert day_of_year.shape == (9,)
    assert second_of_day.shape == (9,)
    assert second_of_day[0].item() == 0
    assert second_of_day[1].item() == 3 * 3600
    assert second_of_day[-1].item() == 0  # 24h -> next midnight
    assert day_of_year[0].item() == 2  # Jan 2 is day-of-year 2
    assert day_of_year[-1].item() == 3  # rolled to Jan 3


def test_video_batch_item_shapes():
    item = _video_item()
    assert item.n_times == 9
    assert item.horizontal_shape == (4, 5)
    moved = item.to_device()
    assert moved.day_of_year.shape == (9,)


def test_video_batch_item_validation():
    good = _video_item()
    # 2-D (image) field is rejected -- video requires a leading time axis
    with pytest.raises(ValueError):
        VideoBatchItem(
            {"x": torch.rand(4, 5)},
            good.time,
            good.latlon_coordinates,
            good.day_of_year,
            good.second_of_day,
        )
    # time length must match the number of frames
    with pytest.raises(ValueError):
        VideoBatchItem(
            good.data,
            _times(3),
            good.latlon_coordinates,
            good.day_of_year,
            good.second_of_day,
        )


def test_video_batch_data_from_sequence():
    items = [_video_item(varnames=("x", "y")) for _ in range(3)]
    batch = VideoBatchData.from_sequence(items)
    assert batch.data["x"].shape == (3, 9, 4, 5)
    assert batch.data["y"].shape == (3, 9, 4, 5)
    assert batch.day_of_year.shape == (3, 9)
    assert batch.second_of_day.shape == (3, 9)
    assert batch.latlon_coordinates.lat.shape == (3, 4)
    assert len(batch) == 3
    batch.to_device()


def _video_config(paths, **kwargs):
    return PairedDataLoaderConfig(
        fine=[XarrayDataConfig(paths.fine)],
        coarse=[XarrayDataConfig(paths.coarse)],
        batch_size=1,
        num_data_workers=0,
        strict_ensemble=False,
        lat_extent=ClosedInterval(1, 4),
        lon_extent=ClosedInterval(0, 3),
        n_timesteps=9,
        **kwargs,
    )


def _requirements():
    return DataRequirements(
        fine_names=["var0", "var1"],
        coarse_names=["var0", "var1"],
        n_timesteps=1,  # ignored by build_video; clip length comes from config
        use_fine_topography=False,
    )


def test_build_video_shapes_and_time_features(tmp_path):
    paths = data_paths_helper(tmp_path, num_timesteps=18)
    data = _video_config(paths).build_video(
        train=False, requirements=_requirements()
    )
    assert data.n_timesteps == 9
    assert data.downscale_factor == 2

    batch = next(iter(data.loader))
    assert isinstance(batch, PairedVideoBatchData)
    fine = batch.fine.data["var0"]
    coarse = batch.coarse.data["var0"]
    # explicit (batch, T, lat, lon) layout
    assert fine.ndim == 4 and fine.shape[1] == 9 and coarse.shape[1] == 9
    # 2x spatial downscaling preserved on both spatial axes
    assert fine.shape[-1] == coarse.shape[-1] * 2
    assert fine.shape[-2] == coarse.shape[-2] * 2
    # per-frame physical-time features attached, aligned across fine/coarse
    assert batch.fine.day_of_year.shape == (1, 9)
    assert torch.equal(batch.fine.day_of_year, batch.coarse.day_of_year)
    # daily timesteps in the fixture -> every frame at midnight UTC
    assert torch.all(batch.fine.second_of_day == 0)


def test_build_video_default_is_per_day_non_overlapping(tmp_path):
    # 18 timesteps, clip length 9 -> 10 stride-one clip starts.
    paths = data_paths_helper(tmp_path, num_timesteps=18)
    requirements = _requirements()

    # Default: per-day non-overlapping (stride = n_timesteps - 1 = 8) -> starts 0, 8.
    default_data = _video_config(paths).build_video(
        train=False, requirements=requirements
    )
    assert len(default_data.loader) == 2

    # Opt-in full sliding window (stride 1) -> all 10 clips.
    sliding_data = _video_config(paths, time_stride=1).build_video(
        train=False, requirements=requirements
    )
    assert len(sliding_data.loader) == 10
