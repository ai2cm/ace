"""This file contains unit tests related to creating torch Datasets from climate
data (e.g. netCDF files)."""

import math
import os
import pathlib

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.config import (
    ConcatDatasetConfig,
    DataLoaderConfig,
    MergeDatasetConfig,
)
from fme.ace.data_loading.getters import (
    get_forcing_data,
    get_gridded_data,
    get_inference_data,
)
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.data_loading.perturbation import PerturbationSelector, SSTPerturbation
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset.config import MergeNoConcatDatasetConfig, XarrayDataConfig
from fme.core.typing_ import Slice


def _get_coords(dim_sizes, calendar, timestep_size=1, timestep_start=0):
    coords = {}
    for dim_name, size in dim_sizes.items():
        if dim_name == "time":
            dtype = np.int64
            start = timestep_start
            step = timestep_size
            size = size * step
            attrs = {"calendar": calendar, "units": "days since 1970-01-01"}
        else:
            dtype = np.float32
            start = 0
            step = 1
            attrs = {}
        coord_value = np.arange(start, start + size, step, dtype=dtype)
        coord = xr.DataArray(coord_value, dims=(dim_name,), attrs=attrs)
        coords[dim_name] = coord
    return coords


def _save_netcdf(
    filename, dim_sizes, variable_names, calendar, timestep_size=1, timestep_start=0
):
    data_vars = {}
    for name in variable_names:
        if name == "constant_mask":
            data = np.ones(list(dim_sizes.values()))
        else:
            data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)  # type: ignore
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes), attrs={"units": "m", "long_name": name}
        )
    coords = _get_coords(dim_sizes, calendar, timestep_size, timestep_start)
    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")
    return ds


def _create_dataset_on_disk(
    data_dir: pathlib.Path,
    calendar: str = "proleptic_gregorian",
    data_dim_sizes=None,
    n_times: int = 3,
    timestep_size: int = 1,
    in_variable_names=["foo", "bar", "baz"],
    out_variable_names=["foo", "bar"],
    filename="data.nc",
) -> pathlib.Path:
    if data_dim_sizes is None:
        data_dim_sizes = {"time": n_times, "lat": 16, "lon": 32}
    seed = 0
    np.random.seed(seed)
    mask_name = "mask"
    constant_mask_name = "constant_mask"
    all_variable_names = list(set(in_variable_names + out_variable_names)) + [
        mask_name,
        constant_mask_name,
    ]

    data_path = data_dir / filename
    _save_netcdf(data_path, data_dim_sizes, all_variable_names, calendar, timestep_size)

    return data_path


def test_ensemble_loader(tmp_path, num_ensemble_members=3):
    """Tests that the ensemble loader returns the correct number of samples."""

    # Create a dataset for each ensemble member. We assume that each member
    # corresponds to an initial condition.
    netcdfs: list[pathlib.Path] = []
    for i in range(num_ensemble_members):
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        _create_dataset_on_disk(ic_path)
        netcdfs.append(ic_path)

    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[XarrayDataConfig(data_path=str(path)) for path in netcdfs]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    n_timesteps = 3  # hard coded to match `_create_dataset_on_disk`.
    samples_per_member = n_timesteps - window_timesteps + 1

    data = get_gridded_data(config, True, requirements)
    assert data.n_batches == samples_per_member * num_ensemble_members
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)


def test_ensemble_loader_n_samples(tmp_path, num_ensemble_members=3, n_samples=1):
    """Tests that the ensemble loader returns the correct number of samples
    when n_samples is set in config.
    """

    # Create a dataset for each ensemble member. We assume that each member
    # corresponds to an initial condition.
    netcdfs: list[pathlib.Path] = []
    for i in range(num_ensemble_members):
        ic_path = tmp_path / f"ic{i}"
        ic_path.mkdir()
        _create_dataset_on_disk(ic_path)
        netcdfs.append(ic_path)

    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[
                XarrayDataConfig(data_path=str(path), subset=Slice(stop=n_samples))
                for path in netcdfs
            ]
        ),
        batch_size=1,
        num_data_workers=0,
    )

    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)

    data = get_gridded_data(config, True, requirements)
    assert data.n_batches == n_samples * num_ensemble_members
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)


def test_xarray_loader(tmp_path):
    """Checks that vertical coordinates are present."""
    _create_dataset_on_disk(tmp_path)
    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[XarrayDataConfig(data_path=tmp_path, n_repeats=1)]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_gridded_data(config, True, requirements)  # type: ignore
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)
    assert data._vertical_coordinate.ak.device == fme.get_device()


def test_xarray_loader_sample_with_replacement(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=3)
    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[XarrayDataConfig(data_path=tmp_path, n_repeats=1)]
        ),
        batch_size=1,
        num_data_workers=0,
        sample_with_replacement=10,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_gridded_data(config, True, requirements)  # type: ignore
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)
    assert data._vertical_coordinate.ak.device == fme.get_device()
    epoch_samples = list(data.loader)
    assert len(epoch_samples) == 10


def test_xarray_loader_using_merged_dataset(tmp_path, tmp_path_factory):
    _create_dataset_on_disk(tmp_path)
    other_path = tmp_path_factory.mktemp("other")
    _create_dataset_on_disk(
        other_path,
        in_variable_names=["foo2", "bar2"],
        out_variable_names=["foo2", "bar2"],
        filename="other_source.nc",
    )

    config = DataLoaderConfig(
        dataset=MergeDatasetConfig(
            merge=[
                XarrayDataConfig(
                    data_path=tmp_path,
                    file_pattern="data*.nc",
                    n_repeats=1,
                ),
                XarrayDataConfig(
                    data_path=other_path, file_pattern="other_source*.nc", n_repeats=1
                ),
            ]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo", "foo2"], window_timesteps)
    data = get_gridded_data(config, True, requirements)  # type: ignore
    assert "foo" in data.variable_metadata.keys()
    assert "foo2" in data.variable_metadata.keys()


def test_xarray_loader_using_merged_dataset_errors_if_different_time(
    tmp_path, tmp_path_factory
):
    _create_dataset_on_disk(tmp_path)
    other_path = tmp_path_factory.mktemp("other")
    _create_dataset_on_disk(
        other_path,
        in_variable_names=["foo2", "bar2"],
        out_variable_names=["foo2", "bar2"],
        filename="other_source.nc",
        n_times=4,
    )

    config = DataLoaderConfig(
        dataset=MergeDatasetConfig(
            merge=[
                XarrayDataConfig(
                    data_path=tmp_path,
                    file_pattern="data*.nc",
                    n_repeats=1,
                ),
                XarrayDataConfig(
                    data_path=other_path,
                    file_pattern="other_source*.nc",
                    n_repeats=1,
                ),
            ]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo", "foo2"], window_timesteps)
    with pytest.raises(
        ValueError,
        match="All datasets in a merged dataset must have the same sample start times",
    ):
        get_gridded_data(config, True, requirements)  # type: ignore

    # subset source2 to have the same time stamps as source1
    config = DataLoaderConfig(
        dataset=MergeDatasetConfig(
            merge=[
                XarrayDataConfig(
                    data_path=tmp_path,
                    file_pattern="data*.nc",
                    n_repeats=1,
                ),
                XarrayDataConfig(
                    data_path=other_path,
                    file_pattern="other_source*.nc",
                    n_repeats=1,
                    subset=Slice(stop=2),
                ),
            ]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    get_gridded_data(config, True, requirements)  # type: ignore


def test_xarray_loader_hpx(tmp_path):
    """Checks that vertical coordinates are present."""
    n_times = 3
    data_dim_sizes = {"time": n_times, "face": 12, "width": 16, "height": 16}
    _create_dataset_on_disk(tmp_path, data_dim_sizes=data_dim_sizes, n_times=n_times)
    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[
                XarrayDataConfig(
                    data_path=tmp_path, n_repeats=1, spatial_dimensions="healpix"
                )
            ]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_gridded_data(config, True, requirements)  # type: ignore
    for batch in data.loader:
        assert batch is not None
        # expect healpix shape
        assert batch.data["foo"].shape == (1, window_timesteps, 12, 16, 16)
        break
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)
    assert data._vertical_coordinate.ak.device == fme.get_device()


def test_loader_n_repeats_but_not_infer_timestep_error(tmp_path):
    _create_dataset_on_disk(tmp_path)
    with pytest.raises(ValueError, match="infer_timestep must be True"):
        DataLoaderConfig(
            dataset=ConcatDatasetConfig(
                concat=[
                    XarrayDataConfig(
                        data_path=tmp_path, n_repeats=2, infer_timestep=False
                    )
                ]
            ),
            batch_size=1,
            num_data_workers=0,
        )


def test_inference_data_loader(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=14)
    batch_size = 2
    step = 7
    config = InferenceDataLoaderConfig(
        dataset=XarrayDataConfig(
            data_path=tmp_path,
            n_repeats=1,
        ),
        start_indices=InferenceInitialConditionIndices(
            first=0, n_initial_conditions=batch_size, interval=step
        ),
    )
    n_forward_steps_in_memory = 3
    window_requirements = DataRequirements(
        names=["foo", "bar"],
        n_timesteps=n_forward_steps_in_memory + 1,
    )
    initial_condition_requirements = PrognosticStateDataRequirements(
        names=["foo"],
        n_timesteps=1,
    )
    data = get_inference_data(
        config,
        total_forward_steps=6,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
    )
    data_loader = data.loader
    batch_data = next(iter(data_loader))
    assert isinstance(batch_data, BatchData)
    for name in ["foo", "bar"]:
        assert isinstance(batch_data.data[name], torch.Tensor)
        assert batch_data.data[name].shape == (
            batch_size,
            n_forward_steps_in_memory + 1,
            16,
            32,
        )
    assert isinstance(batch_data.time, xr.DataArray)
    assert list(batch_data.time.dims) == ["sample", "time"]
    assert batch_data.time.sizes["sample"] == batch_size
    assert batch_data.time.sizes["time"] == n_forward_steps_in_memory + 1
    assert batch_data.time.dt.calendar == "proleptic_gregorian"
    assert isinstance(data._vertical_coordinate, HybridSigmaPressureCoordinate)
    assert data._vertical_coordinate.ak.device == fme.get_device()
    initial_condition = data.initial_condition.as_batch_data()
    assert isinstance(initial_condition, BatchData)
    assert "bar" not in initial_condition.data
    assert initial_condition.data["foo"].shape == (batch_size, 1, 16, 32)


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
        dataset=ConcatDatasetConfig(
            concat=[XarrayDataConfig(data_path=tmp_path, subset=Slice(stop=n_samples))]
        ),
        batch_size=n_samples,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    data = get_gridded_data(config, True, requirements)  # type: ignore
    batch_data = next(iter(data.loader))
    assert isinstance(batch_data, BatchData)
    assert isinstance(batch_data.data["foo"], torch.Tensor)
    assert batch_data.data["foo"].shape[0] == n_samples
    assert isinstance(batch_data.time, xr.DataArray)
    assert list(batch_data.time.dims) == ["sample", "time"]
    assert batch_data.time.sizes["sample"] == n_samples
    assert batch_data.time.sizes["time"] == window_timesteps
    assert batch_data.time.dt.calendar == calendar


@pytest.mark.parametrize(
    ("first_ic_index,n_initial_conditions,ic_interval,num_forward_steps,raises_error"),
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
        dataset=XarrayDataConfig(
            data_path=tmp_path,
            n_repeats=1,
        ),
        start_indices=start_indices,
    )
    n_forward_steps_in_memory = num_forward_steps
    window_requirements = DataRequirements(
        names=["foo", "bar"],
        n_timesteps=n_forward_steps_in_memory + 1,
    )
    initial_condition_requirements = PrognosticStateDataRequirements(
        names=["foo"],
        n_timesteps=1,
    )

    if raises_error:
        with pytest.raises(ValueError):
            get_inference_data(
                config,
                total_forward_steps=num_forward_steps,
                window_requirements=window_requirements,
                initial_condition=initial_condition_requirements,
            )
    else:
        get_inference_data(
            config,
            total_forward_steps=num_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition_requirements,
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
        dataset=ConcatDatasetConfig(
            concat=[
                XarrayDataConfig(
                    data_path=tmp_path, n_repeats=10, subset=Slice(start, stop)
                )
            ]
        ),
        batch_size=batch_size,
        num_data_workers=0,
    )
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], window_timesteps)
    if raises_error:
        with pytest.raises(ValueError):
            get_gridded_data(config, True, requirements)  # type: ignore
    else:
        get_gridded_data(config, True, requirements)  # type: ignore


@pytest.mark.parametrize("n_initial_conditions", [1, 2])
def test_get_forcing_data(tmp_path, n_initial_conditions):
    calendar = "proleptic_gregorian"
    total_forward_steps = 5
    forward_steps_in_memory = 2
    _create_dataset_on_disk(tmp_path, calendar=calendar, n_times=10)
    config = ForcingDataLoaderConfig(dataset=XarrayDataConfig(data_path=tmp_path))
    window_requirements = DataRequirements(
        names=["foo"],
        n_timesteps=forward_steps_in_memory + 1,
    )
    time_values = [
        [cftime.datetime(1970, 1, 1 + 2 * n, calendar=calendar)]
        for n in range(n_initial_conditions)
    ]
    initial_condition = BatchData.new_on_cpu(
        data={"foo": torch.randn(n_initial_conditions, 1, 1, 1)},
        time=xr.DataArray(time_values, dims=["sample", "time"]),
    )
    data = get_forcing_data(
        config,
        total_forward_steps,
        window_requirements=window_requirements,
        initial_condition=PrognosticState(initial_condition),
    )
    assert len(data._loader.dataset) == math.ceil(  # type: ignore
        total_forward_steps / forward_steps_in_memory
    )
    batch_data = next(iter(data.loader))
    assert isinstance(batch_data, BatchData)
    assert isinstance(batch_data.data["foo"], torch.Tensor)
    assert set(batch_data.data.keys()) == {"foo"}
    assert batch_data.data["foo"].shape[0] == len(time_values)
    assert batch_data.data["foo"].shape[1] == forward_steps_in_memory + 1
    assert list(batch_data.time.dims) == ["sample", "time"]
    xr.testing.assert_allclose(batch_data.time[:, 0], initial_condition.time[:, 0])
    assert batch_data.time.dt.calendar == calendar
    xr.testing.assert_equal(
        data.initial_condition.as_batch_data().time,
        initial_condition.time,
    )
    np.testing.assert_allclose(
        data.initial_condition.as_batch_data().data["foo"].cpu().numpy(),
        initial_condition.data["foo"].cpu().numpy(),
    )


def test_inference_loader_raises_if_subset():
    with pytest.raises(ValueError):
        InferenceDataLoaderConfig(
            dataset=XarrayDataConfig(data_path="foo", subset=Slice(stop=1)),
            start_indices=ExplicitIndices([0, 1]),
        )


def test_inference_loader_merged_inputs(tmp_path, tmp_path_factory):
    _create_dataset_on_disk(tmp_path)
    other_path = tmp_path_factory.mktemp("other")
    _create_dataset_on_disk(
        other_path,
        in_variable_names=["foo2", "bar2"],
        out_variable_names=["foo2", "bar2"],
        filename="other_source.nc",
    )
    merged = InferenceDataLoaderConfig(
        dataset=MergeNoConcatDatasetConfig(
            merge=[
                XarrayDataConfig(data_path=tmp_path),
                XarrayDataConfig(data_path=other_path),
            ]
        ),
        start_indices=ExplicitIndices([0, 1]),
    )
    window_requirements = DataRequirements(
        names=["foo", "foo2"],
        n_timesteps=2,
    )
    initial_condition_requirements = PrognosticStateDataRequirements(
        names=["foo", "foo2"],
        n_timesteps=1,
    )
    data = get_inference_data(
        merged,
        total_forward_steps=1,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
        surface_temperature_name="foo",
        ocean_fraction_name="constant_mask",
    )
    data_loader = data.loader
    batch_data = next(iter(data_loader))
    assert "foo" in batch_data.data.keys()
    assert "foo2" in batch_data.data.keys()


def test_forcing_loader_raises_if_subset():
    with pytest.raises(ValueError):
        ForcingDataLoaderConfig(
            dataset=XarrayDataConfig(data_path="foo", subset=Slice(stop=1))
        )


def test_forcing_loader_config_merged_dataset_inputs():
    ForcingDataLoaderConfig(
        dataset=MergeNoConcatDatasetConfig(
            merge=[
                XarrayDataConfig(data_path="foo"),
                XarrayDataConfig(data_path="bar"),
            ]
        )
    )


def test_forcing_loader_loads_merged_dataset(tmp_path, tmp_path_factory):
    calendar = "proleptic_gregorian"
    _create_dataset_on_disk(tmp_path, calendar=calendar)
    other_path = tmp_path_factory.mktemp("other")
    _create_dataset_on_disk(
        other_path,
        calendar=calendar,
        in_variable_names=["foo2", "bar2"],
        out_variable_names=["foo2", "bar2"],
    )
    config = ForcingDataLoaderConfig(dataset=XarrayDataConfig(data_path=tmp_path))
    config = ForcingDataLoaderConfig(
        dataset=MergeNoConcatDatasetConfig(
            merge=[
                XarrayDataConfig(data_path=tmp_path),
                XarrayDataConfig(data_path=other_path),
            ]
        ),
    )
    window_requirements = DataRequirements(
        names=["foo", "foo2"],
        n_timesteps=2,
    )
    time_values = [[cftime.datetime(1970, 1, 1, calendar=calendar)]]
    initial_condition = BatchData.new_on_cpu(
        data={"foo": torch.randn(1, 1, 1, 1), "foo2": torch.randn(1, 1, 1, 1)},
        time=xr.DataArray(time_values, dims=["sample", "time"]),
    )
    data = get_forcing_data(
        config,
        1,
        window_requirements=window_requirements,
        initial_condition=PrognosticState(initial_condition),
    )
    assert "foo" in data.variable_metadata.keys()
    assert "foo2" in data.variable_metadata.keys()


@pytest.mark.parametrize(
    "timestamps, expected_indices",
    [
        (
            [
                "2020-01-01T00:00:00",
                "2020-01-02T00:00:00",
            ],
            [0, 2],
        ),
        (
            [
                "2020-01-01T00:00:00",
                "2021-01-02T00:00:00",
            ],
            None,
        ),
        (
            [
                "2021-01-02T00:00:00",
            ],
            None,
        ),
    ],
)
def test_TimestampList_as_indices(timestamps, expected_indices):
    time_index = xr.CFTimeIndex(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 12, 0, 0),
            cftime.DatetimeJulian(2020, 1, 2, 0, 0, 0),
            cftime.DatetimeJulian(2020, 1, 2, 12, 0, 0),
        ]
    )
    timestamp_list = TimestampList(timestamps)
    if expected_indices is None:
        with pytest.raises(ValueError):
            timestamp_list.as_indices(time_index)
    else:
        np.testing.assert_equal(
            timestamp_list.as_indices(time_index), np.array(expected_indices)
        )


def test_inference_data_with_perturbations(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=14)
    batch_size = 1
    step = 7
    config = InferenceDataLoaderConfig(
        dataset=XarrayDataConfig(
            data_path=tmp_path,
            n_repeats=1,
        ),
        start_indices=InferenceInitialConditionIndices(
            first=0, n_initial_conditions=batch_size, interval=step
        ),
        perturbations=SSTPerturbation(
            sst=[PerturbationSelector(type="constant", config={"amplitude": 2.0})]
        ),
    )
    n_forward_steps_in_memory = 3
    original_foo = xr.open_dataset(
        os.path.join(tmp_path, "data.nc"), decode_times=False
    )["foo"].values[0 : n_forward_steps_in_memory + 1, :, :]
    window_requirements = DataRequirements(
        names=["foo", "constant_mask"],
        n_timesteps=n_forward_steps_in_memory + 1,
    )
    initial_condition_requirements = PrognosticStateDataRequirements(
        names=["foo"],
        n_timesteps=1,
    )
    data = get_inference_data(
        config,
        total_forward_steps=6,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
        surface_temperature_name="foo",
        ocean_fraction_name="constant_mask",
    )
    batch_data = next(iter(data.loader))
    np.testing.assert_allclose(
        original_foo + 2.0,
        batch_data.data["foo"].cpu().numpy()[0, :, :, :],
    )
    np.testing.assert_allclose(
        original_foo[:1, :, :] + 2.0,
        data.initial_condition.as_batch_data().data["foo"].cpu().numpy()[0, :, :, :],
    )


def test_inference_persistence_names(tmp_path):
    _create_dataset_on_disk(tmp_path, n_times=14)

    config = InferenceDataLoaderConfig(
        dataset=XarrayDataConfig(data_path=tmp_path),
        start_indices=ExplicitIndices([0, 3]),
        persistence_names=["foo"],
    )
    window_requirements = DataRequirements(
        names=["foo", "bar"],
        n_timesteps=3,
    )
    dataset = InferenceDataset(
        config,
        9,
        requirements=window_requirements,
    )
    first_item = dataset[0].data
    second_item = dataset[1].data
    # ensure first and second time steps are the same
    torch.testing.assert_close(first_item["foo"][:, 0], first_item["foo"][:, 1])
    # ensure the entire first and second returned items
    torch.testing.assert_close(first_item["foo"], second_item["foo"])
    # ensure this is not the case for another variable
    assert not torch.all(first_item["bar"] == second_item["bar"])


def test_zarr_engine_used_sequence():
    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[
                XarrayDataConfig(data_path="foo", file_pattern=".zarr", engine="zarr"),
                XarrayDataConfig(data_path="bar"),
            ]
        ),
        batch_size=1,
    )
    assert config.zarr_engine_used


def test_zarr_engine_used_mapping():
    config = DataLoaderConfig(
        dataset=MergeDatasetConfig(
            merge=[
                ConcatDatasetConfig(
                    concat=[
                        XarrayDataConfig(
                            data_path="foo", file_pattern=".zarr", engine="zarr"
                        ),
                        XarrayDataConfig(data_path="bar"),
                    ]
                ),
                XarrayDataConfig(data_path="bar"),
            ]
        ),
        batch_size=1,
    )
    assert config.zarr_engine_used


def test_data_loader_maintains_backward_concat_compatibility(recwarn):
    data_loader = DataLoaderConfig(
        dataset=[XarrayDataConfig(data_path="some_path")],
        batch_size=1,
    )
    assert isinstance(data_loader.dataset, ConcatDatasetConfig)
    warning = recwarn.pop(DeprecationWarning)
    assert "Dataset list format is deprecated" in str(warning.message)


@pytest.mark.parametrize(
    (
        "shuffle",
        "steps_on_disk",
        "n_timesteps_requirement",
        "time_buffer",
        "expected_start_indices",
    ),
    (
        pytest.param(
            False,
            7,
            2,
            0,
            list(range(0, 6)),
            id="time_buffer=0",
        ),
        pytest.param(
            False,
            7,
            2,
            2,
            list(range(0, 6)),
            id="time_buffer=2",
        ),
        pytest.param(
            False,
            10,
            2,
            2,
            list(range(0, 9)),
            id="time_buffer=2, steps_on_disk=10",
        ),
        pytest.param(
            False,
            6,
            2,
            2,
            list(range(0, 3)),
            id=(
                "time_buffer=2, steps_on_disk=6 "
                "(lose some samples due to dropping last 'preloaded' batch)"
            ),
        ),
        pytest.param(
            False,
            8,
            3,
            2,
            list(range(0, 6)),
            id="time_buffer=2, steps_on_disk=10, n_timesteps_requirement=3",
        ),
        pytest.param(
            False,
            10,
            3,
            2,
            list(range(0, 6)),
            id=(
                "time_buffer=2, steps_on_disk=10, n_timesteps_requirement=3 "
                "(lose some samples due to dropping last 'preloaded' batch)"
            ),
        ),
        pytest.param(
            True,
            13,
            2,
            2,
            list(range(0, 12)),
            id="time_buffer=2, shuffle=True",
        ),
    ),
)
def test_time_buffer(
    tmp_path,
    shuffle,
    steps_on_disk,
    n_timesteps_requirement,
    time_buffer,
    expected_start_indices,
):
    _create_dataset_on_disk(tmp_path, n_times=steps_on_disk)
    dataset_times = xr.open_dataset(
        os.path.join(tmp_path, "data.nc"),
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )["time"].values
    window_size = time_buffer + 1
    i_window_starts = [
        i
        for i in range(0, len(dataset_times), window_size)
        if i + window_size <= len(dataset_times) - n_timesteps_requirement + 1
    ]
    dataset_window_times = [dataset_times[i : i + window_size] for i in i_window_starts]
    config = DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=tmp_path),
        batch_size=1,
        num_data_workers=0,
        time_buffer=time_buffer,
    )
    requirements = DataRequirements(["foo"], n_timesteps_requirement)
    data = get_gridded_data(config, shuffle, requirements)
    assert len(data.loader) == len(expected_start_indices)
    start_times = []
    for batch in data.loader:
        start_times.append(batch.time[0, 0].item())
        assert batch.n_timesteps == n_timesteps_requirement
    if shuffle:
        assert set(start_times) == set(dataset_times[expected_start_indices])
    else:
        assert start_times == list(dataset_times[expected_start_indices])
    for i_window_start in i_window_starts:
        # confirm that a "window's worth" of data is loaded consecutively;
        # with shuffling we know that it will match one window, but not which one
        loaded_times = start_times[i_window_start : i_window_start + window_size]
        window_matches = [
            set(loaded_times) == set(window_times)
            for window_times in dataset_window_times
        ]
        assert sum(window_matches) == 1
