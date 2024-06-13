"""This file contains unit tests related to creating torch Datasets from climate
data (e.g. netCDF files)."""

import datetime
import math
import pathlib
from typing import List

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.data_loading.config import DataLoaderConfig, Slice, XarrayDataConfig
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.getters import (
    get_data_loader,
    get_forcing_data,
    get_inference_data,
)
from fme.core.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData, get_times


def _get_coords(dim_sizes, calendar):
    coords = {}
    for dim_name, size in dim_sizes.items():
        if dim_name == "time":
            dtype = np.int64
            attrs = {"calendar": calendar, "units": "days since 1970-01-01"}
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
        netcdfs.append(ic_path)

    config = DataLoaderConfig(
        [XarrayDataConfig(data_path=str(path)) for path in netcdfs],
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    n_timesteps = 3  # hard coded to match `_create_dataset_on_disk`.
    samples_per_member = n_timesteps - window_timesteps + 1

    data = get_data_loader(config, True, requirements)
    assert len(data.loader) == samples_per_member * num_ensemble_members
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_ensemble_loader_n_samples(tmp_path, num_ensemble_members=3, n_samples=1):
    """Tests that the ensemble loader returns the correct number of samples
    when n_samples is set in config.
    """

    # Create a dataset for each ensemble member. We assume that each member
    # corresponds to an initial condition.
    netcdfs: List[pathlib.Path] = []
    for i in range(num_ensemble_members):
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        _create_dataset_on_disk(ic_path)
        netcdfs.append(ic_path)

    config = DataLoaderConfig(
        [
            XarrayDataConfig(data_path=str(path), subset=Slice(stop=n_samples))
            for path in netcdfs
        ],
        batch_size=1,
        num_data_workers=0,
    )

    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    data = get_data_loader(config, True, requirements)
    assert len(data.loader) == n_samples * num_ensemble_members
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_xarray_loader(tmp_path):
    """Checks that sigma coordinates are present."""
    _create_dataset_on_disk(tmp_path)
    config = DataLoaderConfig(
        [XarrayDataConfig(data_path=tmp_path, n_repeats=1)],
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_data_loader(config, True, requirements)  # type: ignore
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_loader_n_repeats_but_not_infer_timestep_error(tmp_path):
    _create_dataset_on_disk(tmp_path)
    with pytest.raises(ValueError, match="infer_timestep must be True"):
        DataLoaderConfig(
            [XarrayDataConfig(data_path=tmp_path, n_repeats=2, infer_timestep=False)],
            batch_size=1,
            num_data_workers=0,
        )


def test_inference_data_loader(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=14)
    batch_size = 2
    step = 7
    config = InferenceDataLoaderConfig(
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
        config,
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
    config = DataLoaderConfig(
        [XarrayDataConfig(data_path=tmp_path, subset=Slice(stop=n_samples))],
        batch_size=n_samples,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_data_loader(config, True, requirements)  # type: ignore
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


@pytest.mark.parametrize(
    (
        "first_ic_index,"
        "n_initial_conditions,"
        "ic_interval,"
        "num_forward_steps,"
        "raises_error"
    ),
    [
        (0, 1, 1, 9, False),
        (0, 1, 1, 10, True),
        (1, 1, 1, 8, False),
        (1, 1, 1, 9, True),
        (2, 3, 2, 3, False),
        (2, 3, 2, 4, True),
    ],
)
def test_inference_data_loader_validate_n_forward_steps(
    tmp_path,
    first_ic_index,
    n_initial_conditions,
    ic_interval,
    num_forward_steps,
    raises_error,
):
    """Check exception is raised if n_forward_steps exceeds number of
    forward steps in dataset available after last initial condition."""

    total_dataset_timesteps = 10
    _create_dataset_on_disk(tmp_path, n_times=total_dataset_timesteps)
    start_indices = InferenceInitialConditionIndices(
        first=first_ic_index,
        n_initial_conditions=n_initial_conditions,
        interval=ic_interval,
    )
    config = InferenceDataLoaderConfig(
        XarrayDataConfig(
            data_path=tmp_path,
            n_repeats=1,
        ),
        start_indices=start_indices,
    )
    n_forward_steps_in_memory = num_forward_steps
    requirements = DataRequirements(["foo"], n_timesteps=num_forward_steps + 1)

    if raises_error:
        with pytest.raises(ValueError):
            get_inference_data(
                config,
                forward_steps_in_memory=n_forward_steps_in_memory,
                requirements=requirements,
            )
    else:
        get_inference_data(
            config,
            forward_steps_in_memory=n_forward_steps_in_memory,
            requirements=requirements,
        )


@pytest.mark.parametrize(
    "start, stop, batch_size, raises_error",
    [
        pytest.param(0, 3, 1, False, id="valid"),
        pytest.param(10000, 10100, 1, True, id="no_samples"),
        pytest.param(0, 25, 50, True, id="batch_size_larger_than_nsamples"),
    ],
)
def test_zero_batches_raises_error(tmp_path, start, stop, batch_size, raises_error):
    _create_dataset_on_disk(tmp_path)
    config = DataLoaderConfig(
        [XarrayDataConfig(data_path=tmp_path, n_repeats=10, subset=Slice(start, stop))],
        batch_size=batch_size,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    if raises_error:
        with pytest.raises(ValueError):
            get_data_loader(config, True, requirements)  # type: ignore
    else:
        get_data_loader(config, True, requirements)  # type: ignore


@pytest.mark.parametrize("n_initial_conditions", [1, 2])
def test_get_forcing_data(tmp_path, n_initial_conditions):
    calendar = "proleptic_gregorian"
    total_forward_steps = 5
    forward_steps_in_memory = 2
    _create_dataset_on_disk(tmp_path, calendar=calendar, n_times=10)
    config = ForcingDataLoaderConfig(XarrayDataConfig(data_path=tmp_path))
    requirements = DataRequirements(["foo"], total_forward_steps + 1)
    time_values = [
        cftime.datetime(1970, 1, 1 + 2 * n, calendar=calendar)
        for n in range(n_initial_conditions)
    ]
    initial_times = xr.DataArray(time_values, dims=["sample"])
    data = get_forcing_data(
        config, forward_steps_in_memory, requirements, initial_times
    )
    assert len(data.loader.dataset) == math.ceil(
        total_forward_steps / forward_steps_in_memory
    )
    batch_data = next(iter(data.loader))
    assert isinstance(batch_data, BatchData)
    assert isinstance(batch_data.data["foo"], torch.Tensor)
    assert set(batch_data.data.keys()) == {"foo"}
    assert batch_data.data["foo"].shape[0] == len(time_values)
    assert batch_data.data["foo"].shape[1] == forward_steps_in_memory + 1
    assert list(batch_data.times.dims) == ["sample", "time"]
    xr.testing.assert_allclose(batch_data.times.isel(time=0), initial_times)
    assert batch_data.times.dt.calendar == calendar


def test_inference_loader_raises_if_subset():
    with pytest.raises(ValueError):
        InferenceDataLoaderConfig(
            XarrayDataConfig(data_path="foo", subset=Slice(stop=1)),
            start_indices=ExplicitIndices([0, 1]),
        )


def test_forcing_loader_raises_if_subset():
    with pytest.raises(ValueError):
        ForcingDataLoaderConfig(XarrayDataConfig(data_path="foo", subset=Slice(stop=1)))
