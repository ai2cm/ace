"""This file contains unit tests related to creating torch Datasets from climate
data (e.g. netCDF files)."""

import datetime
import pathlib
from typing import List

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.getters import get_data_loader, get_inference_data
from fme.core.data_loading.inference import (
    InferenceDataLoaderParams,
    InferenceInitialConditionIndices,
)
from fme.core.data_loading.params import DataLoaderConfig, Slice, XarrayDataConfig
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData, get_times


def _get_coords(dim_sizes, calendar):
    coords = {}
    for dim_name, size in dim_sizes.items():
        if dim_name == "time":
            dtype = np.int64
            attrs = {"calendar": calendar, "units": "seconds since 1970-01-01"}
        else:
            dtype = np.float32
            attrs = {}
        coord_value = np.arange(size, dtype=dtype)
        coord = xr.DataArray(coord_value, dims=(dim_name,), attrs=attrs)
        coords[dim_name] = coord
    return coords


def _save_netcdf(filename, dim_sizes, variable_names, calendar):
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)  # type: ignore
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes), attrs={"units": "m", "long_name": name}
        )
    coords = _get_coords(dim_sizes, calendar)
    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")


def _create_dataset_on_disk(
    data_dir: pathlib.Path, calendar: str = "proleptic_gregorian", n_times: int = 3
) -> pathlib.Path:
    seed = 0
    np.random.seed(seed)
    in_variable_names = ["foo", "bar", "baz"]
    out_variable_names = ["foo", "bar"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names)) + [mask_name]
    data_dim_sizes = {"time": n_times, "grid_yt": 16, "grid_xt": 32}

    data_path = data_dir / "data.nc"
    _save_netcdf(data_path, data_dim_sizes, all_variable_names, calendar)

    return data_path


def test_ensemble_loader(tmp_path, num_ensemble_members=3):
    """Tests that the ensemble loader returns the correct number of samples."""

    # Create a dataset for each ensemble member. We assume that each member
    # corresponds to an initial condition.
    netcdfs: List[pathlib.Path] = []
    for i in range(num_ensemble_members):
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        _create_dataset_on_disk(ic_path)
        netcdfs.append(ic_path / "data")

    params = DataLoaderConfig(
        XarrayDataConfig(data_path=tmp_path, n_repeats=1),
        batch_size=1,
        num_data_workers=0,
        data_type="ensemble_xarray",
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    n_timesteps = 3  # hard coded to match `_create_dataset_on_disk`.
    samples_per_member = n_timesteps - window_timesteps + 1

    data = get_data_loader(params, True, requirements)
    assert len(data.loader) == samples_per_member * num_ensemble_members
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_ensemble_loader_n_samples(tmp_path, num_ensemble_members=3, n_samples=1):
    """Tests that the ensemble loader returns the correct number of samples
    when n_samples is set in params.
    """

    # Create a dataset for each ensemble member. We assume that each member
    # corresponds to an initial condition.
    netcdfs: List[pathlib.Path] = []
    for i in range(num_ensemble_members):
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        _create_dataset_on_disk(ic_path)
        netcdfs.append(ic_path / "data")

    params = DataLoaderConfig(
        XarrayDataConfig(data_path=tmp_path, n_repeats=1),
        batch_size=1,
        num_data_workers=0,
        data_type="ensemble_xarray",
        subset=Slice(stop=n_samples),
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    data = get_data_loader(params, True, requirements)
    assert len(data.loader) == n_samples * num_ensemble_members
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_xarray_loader(tmp_path):
    """Checks that sigma coordinates are present."""
    _create_dataset_on_disk(tmp_path)
    params = DataLoaderConfig(
        XarrayDataConfig(data_path=tmp_path, n_repeats=1),
        batch_size=1,
        num_data_workers=0,
        data_type="xarray",
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_data_loader(params, True, requirements)  # type: ignore
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_inference_data_loader(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=14)
    batch_size = 2
    step = 7
    params = InferenceDataLoaderParams(
        XarrayDataConfig(
            data_path=tmp_path,
            n_repeats=1,
        ),
        start_indices=InferenceInitialConditionIndices(
            first=0, n_initial_conditions=batch_size, interval=step
        ),
    )
    n_forward_steps_in_memory = 3
    requirements = DataRequirements(["foo"], n_timesteps=7)
    data_loader = get_inference_data(
        params,
        forward_steps_in_memory=n_forward_steps_in_memory,
        requirements=requirements,
    ).loader
    batch_data = next(iter(data_loader))
    assert isinstance(batch_data, BatchData)
    assert isinstance(batch_data.data["foo"], torch.Tensor)
    assert batch_data.data["foo"].shape[0] == batch_size
    assert batch_data.data["foo"].shape[1] == n_forward_steps_in_memory + 1
    assert batch_data.data["foo"].shape[2] == 16
    assert batch_data.data["foo"].shape[3] == 32
    assert isinstance(batch_data.times, xr.DataArray)
    assert list(batch_data.times.dims) == ["sample", "time"]
    assert batch_data.times.sizes["sample"] == batch_size
    assert batch_data.times.sizes["time"] == n_forward_steps_in_memory + 1
    assert batch_data.times.dt.calendar == "proleptic_gregorian"
    assert len(data_loader) == 2


@pytest.fixture(params=["julian", "proleptic_gregorian", "noleap"])
def calendar(request):
    """
    These are the calendars for the datasets we tend to use: 'julian'
    for FV3GFS, 'noleap' for E3SM, and 'proleptic_gregorian' for generic
    datetimes in testing.

    Check that datasets created with each calendar for their time coordinate
    are read by the data loader and the calendar is retained.
    """
    return request.param


def test_data_loader_outputs(tmp_path, calendar):
    _create_dataset_on_disk(tmp_path, calendar=calendar)
    n_samples = 2
    params = DataLoaderConfig(
        XarrayDataConfig(
            data_path=tmp_path,
        ),
        batch_size=n_samples,
        num_data_workers=0,
        data_type="xarray",
        subset=Slice(stop=n_samples),
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_data_loader(params, True, requirements)  # type: ignore
    batch_data = next(iter(data.loader))
    assert isinstance(batch_data, BatchData)
    assert isinstance(batch_data.data["foo"], torch.Tensor)
    assert batch_data.data["foo"].shape[0] == n_samples
    assert isinstance(batch_data.times, xr.DataArray)
    assert list(batch_data.times.dims) == ["sample", "time"]
    assert batch_data.times.sizes["sample"] == n_samples
    assert batch_data.times.sizes["time"] == window_timesteps
    assert batch_data.times.dt.calendar == calendar


def test_get_times_non_cftime():
    """
    Check that `get_times` raises an error when the time coordinate is not
    cftime.datetime
    """
    n_times = 5
    times = [datetime.datetime(2020, 1, 1, i, 0, 0) for i in range(n_times)]
    ds = xr.Dataset(
        {"foo": xr.DataArray(np.arange(n_times), dims=("time",))},
        coords={"time": times},
    )
    with pytest.raises(AssertionError):
        get_times(ds, 0, 1)
