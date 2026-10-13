import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates, NullVerticalCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.schedule import IntMilestone, IntSchedule
from fme.core.dataset.testing import MockDataset, assert_dataset_item_length
from fme.core.dataset.time_coarsen import TimeCoarsenDataset, TimeCoarsenDatasetConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.spatial_mask_provider import SpatialMaskProvider

N_LAT, N_LON = 3, 4


def _get_properties(names, timestep):
    return DatasetProperties(
        variable_metadata={
            name: VariableMetadata(units="", long_name="") for name in names
        },
        vertical_coordinate=NullVerticalCoordinate(),
        horizontal_coordinates=LatLonCoordinates(
            lon=torch.arange(0, N_LON), lat=torch.arange(0, N_LAT)
        ),
        spatial_mask_provider=SpatialMaskProvider(),
        timestep=timestep,
        is_remote=False,
        all_labels=None,
    )


def _get_mock_dataset(
    n_times: int, sample_n_times: int, timestep=datetime.timedelta(days=1)
) -> MockDataset:
    names = ["snapshot", "window"]
    time = xr.date_range(
        MockDataset.START_DATE,
        periods=n_times,
        freq="D",
        calendar=MockDataset.CALENDAR,
        use_cftime=True,
    )
    data = {name: torch.randn(n_times, N_LAT, N_LON) for name in names}
    return MockDataset(
        time=time,
        data=data,
        sample_n_times=sample_n_times,
        properties=_get_properties(names, timestep),
        initial_epoch=3,
    )


@pytest.mark.parametrize("n_coarse, factor", [(2, 4), (3, 2), (4, 1)])
def test_time_coarsen_dataset(n_coarse: int, factor: int):
    n_times = 20
    base_dataset = _get_mock_dataset(n_times, sample_n_times=n_coarse * factor)
    dataset = TimeCoarsenDataset(base_dataset, factor, window_names=["window"])
    assert len(dataset) == len(base_dataset)
    assert dataset.sample_n_times == n_coarse
    assert dataset.properties.timestep == datetime.timedelta(days=factor)
    expected_starts = base_dataset.sample_start_times + datetime.timedelta(
        days=factor - 1
    )
    assert dataset.sample_start_times.equals(expected_starts)
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert_dataset_item_length(item)
        data, time, labels, epoch, missing_names = item
        assert epoch == 3
        assert labels is None
        assert missing_names is None
        assert time.shape == (n_coarse,)
        assert time.values[0] == dataset.sample_start_times[idx]
        base_data, base_time, *_ = base_dataset[idx]
        for k in range(n_coarse):
            fine_slice = slice(k * factor, (k + 1) * factor)
            assert time.values[k] == base_time.values[(k + 1) * factor - 1]
            torch.testing.assert_close(
                data["snapshot"][k], base_data["snapshot"][(k + 1) * factor - 1]
            )
            torch.testing.assert_close(
                data["window"][k], base_data["window"][fine_slice].mean(dim=0)
            )


def test_time_coarsen_dataset_requires_timestep():
    base_dataset = _get_mock_dataset(8, sample_n_times=4, timestep=None)
    with pytest.raises(ValueError, match="inferred timestep"):
        TimeCoarsenDataset(base_dataset, 2, window_names=["window"])


def test_time_coarsen_dataset_rejects_non_multiple_sample_length():
    base_dataset = _get_mock_dataset(8, sample_n_times=5)
    dataset = TimeCoarsenDataset(base_dataset, 2, window_names=["window"])
    with pytest.raises(ValueError, match="not a multiple"):
        dataset.sample_n_times
    with pytest.raises(ValueError, match="not a multiple"):
        dataset[0]


def test_time_coarsen_dataset_forwards_epoch():
    base_dataset = _get_mock_dataset(8, sample_n_times=4)
    dataset = TimeCoarsenDataset(base_dataset, 2, window_names=["window"])
    dataset.set_epoch(7)
    assert base_dataset.epoch == 7
    assert dataset[0][3] == 7


def test_time_coarsen_dataset_config_validation():
    inner = XarrayDataConfig(data_path="/nonexistent", file_pattern="*.nc")
    with pytest.raises(ValueError, match="positive integer"):
        TimeCoarsenDatasetConfig(dataset=inner, factor=0, window_names=["a"])
    with pytest.raises(ValueError, match="both snapshot and window"):
        TimeCoarsenDatasetConfig(
            dataset=inner, factor=2, window_names=["a"], snapshot_names=["a", "b"]
        )
    config = TimeCoarsenDatasetConfig(
        dataset=inner, factor=2, window_names=["a"], snapshot_names=["b"]
    )
    with pytest.raises(ValueError, match="neither snapshot_names nor window_names"):
        config.build(["a", "b", "c"], IntSchedule.from_constant(2))
    assert not config.zarr_engine_used
    assert config.available_labels is None


def _write_dataset(path, n_times: int):
    """Daily data with a constant (time-invariant) field, as netCDF."""
    time = xr.date_range(
        "2000-01-01", periods=n_times, freq="D", calendar="noleap", use_cftime=True
    )
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        data_vars={
            "snapshot": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_times, N_LAT, N_LON)).astype(np.float32),
            ),
            "window": (
                ("time", "lat", "lon"),
                rng.standard_normal((n_times, N_LAT, N_LON)).astype(np.float32),
            ),
            "constant": (
                ("lat", "lon"),
                rng.standard_normal((N_LAT, N_LON)).astype(np.float32),
            ),
        },
        coords={
            "time": time,
            "lat": np.arange(N_LAT, dtype=np.float32),
            "lon": np.arange(N_LON, dtype=np.float32),
        },
    )
    ds.to_netcdf(path / "data.nc", unlimited_dims=["time"])
    return ds


@pytest.mark.parametrize("factor", [1, 4])
def test_time_coarsen_dataset_config_builds_from_xarray(tmp_path, factor: int):
    n_times = 17
    raw = _write_dataset(tmp_path, n_times)
    names = ["snapshot", "window", "constant"]
    config = TimeCoarsenDatasetConfig(
        dataset=XarrayDataConfig(data_path=str(tmp_path), file_pattern="data.nc"),
        factor=factor,
        window_names=["window"],
        snapshot_names=["snapshot", "constant"],
    )
    # 1 initial condition + 1 forward step at the coarse resolution, growing
    # to 1 + 2 steps at epoch 2
    n_timesteps = IntSchedule(
        start_value=2, milestones=[IntMilestone(epoch=2, value=3)]
    )
    dataset, properties = config.build(names, n_timesteps)
    assert properties.timestep == datetime.timedelta(days=factor)
    assert dataset.properties.timestep == datetime.timedelta(days=factor)
    # the wrapped dataset is sized for the longest window of the schedule
    n_fine_max = 3 * factor
    assert len(dataset) == n_times - n_fine_max + 1
    dataset.set_epoch(0)
    assert dataset.sample_n_times == 2
    first_ic = raw.time.values[factor - 1]
    assert dataset.sample_start_times[0] == first_ic
    for idx in [0, len(dataset) - 1]:
        data, time, *_ = dataset[idx]
        assert time.shape == (2,)
        assert set(data.keys()) == set(names)
        for k in range(2):
            fine_end = idx + (k + 1) * factor - 1
            assert time.values[k] == raw.time.values[fine_end]
            np.testing.assert_allclose(
                data["snapshot"][k].numpy(), raw["snapshot"].values[fine_end]
            )
            np.testing.assert_allclose(
                data["constant"][k].numpy(), raw["constant"].values
            )
            np.testing.assert_allclose(
                data["window"][k].numpy(),
                raw["window"].values[idx + k * factor : fine_end + 1].mean(axis=0),
                rtol=1e-5,
            )
    # XarraySubset caches sample_n_times at construction, so only the items
    # reflect the schedule once the epoch advances
    dataset.set_epoch(2)
    data, time, *_ = dataset[0]
    assert time.shape == (3,)
    assert data["window"].shape[0] == 3
