"""This file contains unit tests of XarrayDataset."""

import dataclasses
import datetime
import os
from collections import namedtuple
from collections.abc import Iterable

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

from fme.core.coordinates import (
    DepthCoordinate,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    NullVerticalCoordinate,
)
from fme.core.dataset.concat import XarrayConcat, get_dataset
from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.dataset.utils import FillNaNsConfig
from fme.core.dataset.xarray import (
    GET_RAW_TIMES_NUM_FILES_PARALLELIZATION_THRESHOLD,
    OverwriteConfig,
    XarrayDataConfig,
    XarrayDataset,
    XarraySubset,
    _get_cumulative_timesteps,
    _get_file_local_index,
    _get_raw_times,
    _get_timestep,
    _get_vertical_coordinate,
    _repeat_and_increment_time,
    get_xarray_dataset,
)
from fme.core.mask_provider import MaskProvider
from fme.core.typing_ import Slice

from .utils import as_broadcasted_tensor

SLICE_NONE = slice(None)
MOCK_DATA_FREQ = "3h"
MOCK_DATA_START_DATE = "2003-03"
MOCK_DATA_LAT_DIM, MOCK_DATA_LON_DIM = ("lat", "lon")


@dataclasses.dataclass
class VariableNames:
    time_dependent_names: Iterable[str]
    time_invariant_names: Iterable[str]
    initial_condition_names: Iterable[str]

    def _concat(self, *lists):
        return_value = []
        for list in lists:
            return_value.extend(list)
        return return_value

    @property
    def all_names(self) -> list[str]:
        return self._concat(
            self.time_dependent_names,
            self.time_invariant_names,
            self.initial_condition_names,
        )

    @property
    def spatial_resolved_names(self) -> list[str]:
        return self._concat(self.time_dependent_names, self.time_invariant_names)


MockData = namedtuple(
    "MockData", ("tmpdir", "obs_times", "start_times", "start_indices", "var_names")
)


def _get_data(
    tmp_path_factory,
    dirname,
    start,
    end,
    file_freq,
    step_freq,
    calendar,
    with_nans=False,
    var_names=["foo", "bar"],
    write_extra_vars=True,
    add_ensemble_dim=False,
) -> MockData:
    """Constructs an xarray dataset and saves to disk in netcdf format."""
    obs_times = xr.date_range(
        start,
        end,
        freq=step_freq,
        calendar=calendar,
        inclusive="left",
        use_cftime=True,
    )
    start_times = xr.date_range(
        start,
        end,
        freq=file_freq,
        calendar=calendar,
        inclusive="left",
        use_cftime=True,
    )
    obs_delta = obs_times[1] - obs_times[0]
    n_levels = 2
    n_lat, n_lon = 4, 8
    n_sample = 3

    non_time_dims = ("sample", "lat", "lon") if add_ensemble_dim else ("lat", "lon")
    non_time_shape = (n_sample, n_lat, n_lon) if add_ensemble_dim else (n_lat, n_lon)

    constant_var = xr.DataArray(
        np.random.randn(*non_time_shape).astype(np.float32),
        dims=non_time_dims,
    )
    constant_scalar_var = xr.DataArray(1.0).astype(np.float32)
    ak = {f"ak_{i}": float(i) for i in range(n_levels)}
    bk = {f"bk_{i}": float(i + 1) for i in range(n_levels)}
    tmpdir = tmp_path_factory.mktemp(dirname)
    filenames = []
    for i, first in enumerate(start_times):
        if first != start_times[-1]:
            last = start_times[i + 1]
        else:
            last = obs_times[-1] + obs_delta
        time = xr.date_range(
            first,
            last,
            freq=step_freq,
            calendar=calendar,
            inclusive="left",
            use_cftime=True,
        )
        data_vars: dict[str, float | xr.DataArray] = {**ak, **bk}
        for var_name in var_names:
            data = np.random.randn(len(time), *non_time_shape).astype(np.float32)
            if with_nans:
                data[0, :, 0] = np.nan
            data_vars[var_name] = xr.DataArray(data, dims=("time", *non_time_dims))

        data_varying_scalar = np.random.randn(len(time)).astype(np.float32)
        if with_nans:
            constant_var[0, 0] = np.nan

        if write_extra_vars:
            data_vars["varying_scalar_var"] = xr.DataArray(
                data_varying_scalar, dims=("time",)
            )
            data_vars["constant_var"] = constant_var
            data_vars["constant_scalar_var"] = constant_scalar_var

        coords = {
            "time": xr.DataArray(time, dims=("time",)),
            "lat": xr.DataArray(np.arange(n_lat, dtype=np.float32), dims=("lat",)),
            "lon": xr.DataArray(np.arange(n_lon, dtype=np.float32), dims=("lon",)),
        }
        if add_ensemble_dim:
            coords["sample"] = xr.DataArray(
                np.arange(n_sample, dtype=np.float32), dims=("sample",)
            )
            # variable without the ensemble dimension is useful for checking
            # broadcast behavior
            data_vars["var_no_ensemble_dim"] = xr.DataArray(
                np.random.randn(len(time), n_lat, n_lon).astype(np.float32),
                dims=("time", "lat", "lon"),
            )
            # set values to sample index for testing convenience
            sample_index_values = np.broadcast_to(
                np.arange(n_sample).reshape(1, n_sample, 1, 1),  # shape [1, ns, 1, 1],
                (len(time), n_sample, n_lat, n_lon),
            )
            data_vars["var_matches_sample_index"] = (
                xr.zeros_like(data_vars["foo"]) + sample_index_values
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        filename = tmpdir / f"{first.strftime('%Y%m%d%H')}.nc"
        ds.to_netcdf(
            filename,
            unlimited_dims=["time"],
            format="NETCDF4",
        )
        filenames.append(filename)

    initial_condition_names = ()
    start_indices = _get_cumulative_timesteps(_get_raw_times(filenames, "netcdf4"))
    if write_extra_vars:
        variable_names = VariableNames(
            time_dependent_names=(*var_names, "varying_scalar_var"),
            time_invariant_names=("constant_var", "constant_scalar_var"),
            initial_condition_names=initial_condition_names,
        )
    else:
        variable_names = VariableNames(
            time_dependent_names=var_names,
            time_invariant_names=(),
            initial_condition_names=initial_condition_names,
        )
    return MockData(tmpdir, obs_times, start_times, start_indices, variable_names)


def get_mock_monthly_netcdfs(
    tmp_path_factory,
    dirname,
    with_nans=False,
    end_date="2003-06",
    var_names=["foo", "bar"],
    write_extra_vars=True,
    add_ensemble_dim=False,
) -> MockData:
    return _get_data(
        tmp_path_factory,
        dirname,
        start=MOCK_DATA_START_DATE,
        end=end_date,
        file_freq="MS",
        step_freq=MOCK_DATA_FREQ,
        calendar="standard",
        with_nans=with_nans,
        var_names=var_names,
        write_extra_vars=write_extra_vars,
        add_ensemble_dim=add_ensemble_dim,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month")


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_another_source(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory, "month_another_source", var_names=["baz", "qux"]
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_another_source_diff_time(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory,
        "month_another_source",
        end_date="2003-08",
        var_names=["baz", "qux"],
        write_extra_vars=False,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_with_nans(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month_with_nans", with_nans=True)


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_ensemble_dim(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(
        tmp_path_factory,
        "month_with_ensemble_dim",
        add_ensemble_dim=True,
        var_names=["foo", "bar", "var_no_ensemble_dim", "var_matches_sample_index"],
    )


@pytest.fixture(scope="session")
def mock_monthly_zarr_ensemble_dim(
    tmp_path_factory, mock_monthly_netcdfs_ensemble_dim
) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(
        mock_monthly_netcdfs_ensemble_dim.tmpdir.glob("*.nc")
    )
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs_ensemble_dim.obs_times,
        mock_monthly_netcdfs_ensemble_dim.start_times,
        mock_monthly_netcdfs_ensemble_dim.start_indices,
        mock_monthly_netcdfs_ensemble_dim.var_names,
    )


def load_files_without_dask(files, engine="netcdf4") -> xr.Dataset:
    """Load a sequence of files without dask, concatenating along the time dimension.

    We load the data from the files into memory to ensure Datasets are properly closed,
    since xarray cannot concatenate Datasets lazily without dask anyway. This should be
    acceptable for the small datasets we construct for test purposes.
    """
    datasets = []
    for file in sorted(files):
        with xr.open_dataset(
            file,
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
            engine=engine,
        ) as ds:
            datasets.append(ds.load())
    return xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")


@pytest.fixture(scope="session")
def mock_monthly_zarr(tmp_path_factory, mock_monthly_netcdfs) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(mock_monthly_netcdfs.tmpdir.glob("*.nc"))
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs.obs_times,
        mock_monthly_netcdfs.start_times,
        mock_monthly_netcdfs.start_indices,
        mock_monthly_netcdfs.var_names,
    )


@pytest.fixture(scope="session")
def mock_monthly_zarr_with_nans(
    tmp_path_factory, mock_monthly_netcdfs_with_nans
) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = load_files_without_dask(mock_monthly_netcdfs_with_nans.tmpdir.glob("*.nc"))
    data.to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs_with_nans.obs_times,
        mock_monthly_netcdfs_with_nans.start_times,
        mock_monthly_netcdfs_with_nans.start_indices,
        mock_monthly_netcdfs_with_nans.var_names,
    )


@pytest.fixture(scope="session")
def mock_yearly_netcdfs(tmp_path_factory):
    return _get_data(
        tmp_path_factory,
        "yearly",
        start="1999",
        end="2001",
        file_freq="YS",
        step_freq="1D",
        calendar="noleap",
    )


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(0, (0, 0), id="monthly_file_local_idx_2003_03_01_00"),
        pytest.param(1, (0, 1), id="monthly_file_local_idx_2003_03_01_03"),
        pytest.param(30 * 8, (0, 30 * 8), id="monthly_file_local_idx_2003_03_31_00"),
        pytest.param(31 * 8, (1, 0), id="monthly_file_local_idx_2003_04_01_00"),
        pytest.param(
            (31 + 30 + 20) * 8 - 1,
            (2, 20 * 8 - 1),
            id="monthly_file_local_idx_2003_05_20_21",
        ),
    ],
)
def test_monthly_file_local_index(
    mock_monthly_netcdfs, global_idx, expected_file_local_idx
):
    mock_data: MockData = mock_monthly_netcdfs
    file_local_idx = _get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = np.datetime64(
        cftime.DatetimeGregorian(2003, 3, 1, 0, 0, 0, 0, has_year_zero=False)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(
        full_paths[file_idx],
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(31 * 8 - 1, id="monthly_XarrayDataset_2003_03_31_21"),
        pytest.param((31 + 30 + 20) * 8 - 1, id="monthly_XarrayDataset_2003_05_20_21"),
        pytest.param((31 + 30) * 8 - 1, id="2003_04_30_21 (test for GH #1942)"),
    ],
)
@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern, labels",
    [
        ("mock_monthly_netcdfs", "netcdf4", "*.nc", set()),
        ("mock_monthly_zarr", "zarr", "*.zarr", {"foo_label"}),
    ],
)
def test_XarrayDataset_monthly(
    global_idx, mock_data_fixture, engine, file_pattern, request, labels
):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    var_names: VariableNames = mock_data.var_names
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        file_pattern=file_pattern,
        engine=engine,
        labels=labels,
    )
    dataset = XarrayDataset(config, var_names.all_names, 2)
    expected_n_samples = len(mock_data.obs_times) - 1

    assert len(dataset) == expected_n_samples
    arrays, time, dataset_labels = dataset[global_idx]
    assert dataset_labels == labels
    ds = load_files_without_dask(mock_data.tmpdir.glob(file_pattern), engine=engine)
    target_times = ds["time"][global_idx : global_idx + 2].drop_vars("time")
    xr.testing.assert_equal(time, target_times)
    lat_dim, lon_dim = MOCK_DATA_LAT_DIM, MOCK_DATA_LON_DIM
    dims = ("time", str(lat_dim), str(lon_dim))
    shape = (2, ds.sizes[lat_dim], ds.sizes[lon_dim])
    time_slice = slice(global_idx, global_idx + 2)
    for var_name in var_names.spatial_resolved_names:
        data = arrays[var_name]
        assert data.shape[0] == 2
        da = ds[var_name]
        if var_name in var_names.time_dependent_names:
            da = da.isel(time=time_slice)
        target_data = as_broadcasted_tensor(da.variable, dims, shape)
        assert torch.equal(data, target_data)

    for var_name in mock_data.var_names.initial_condition_names:
        data = arrays[var_name].detach().numpy()
        assert data.shape[0] == 1
        target_data = ds[var_name][global_idx : global_idx + 1, :, :].values
        assert np.all(data == target_data)


@pytest.mark.parametrize("n_samples", [None, 1])
@pytest.mark.parametrize("labels", [set(), {"foo"}])
def test_XarrayDataset_monthly_n_timesteps(mock_monthly_netcdfs, n_samples, labels):
    """Test that increasing n_timesteps decreases the number of samples."""
    mock_data: MockData = mock_monthly_netcdfs
    if len(mock_data.var_names.initial_condition_names) != 0:
        return
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=Slice(stop=n_samples), labels=labels
    )
    n_forward_steps = 4
    dataset, properties = get_xarray_dataset(
        config, mock_data.var_names.all_names + ["x"], n_forward_steps + 1
    )
    assert properties.all_labels == labels
    if n_samples is None:
        assert len(dataset) == len(mock_data.obs_times) - n_forward_steps
    else:
        assert len(dataset) == n_samples
    assert "x" in dataset[0][0]


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(365 + 59, (1, 59), id="yearly_file_local_idx_2000_03_01"),
        pytest.param(365, (1, 0), id="yearly_file_local_idx_2000_01_01"),
        pytest.param(364, (0, 364), id="yearly_file_local_idx_1999_12_31"),
    ],
)
def test_yearly_file_local_index(
    mock_yearly_netcdfs, global_idx, expected_file_local_idx
):
    mock_data: MockData = mock_yearly_netcdfs
    file_local_idx = _get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = (
        cftime.DatetimeNoLeap(1999, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(
        full_paths[file_idx],
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(364, id="yearly_XarrayDataset_1999_12_31"),
        pytest.param(365 + 31 + 28, id="yearly_XarrayDataset_2000_02_28"),
    ],
)
@pytest.mark.parametrize("labels", [set(), {"foo"}])
def test_XarrayDataset_yearly(mock_yearly_netcdfs, global_idx, labels):
    mock_data: MockData = mock_yearly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir, labels=labels)
    ds = load_files_without_dask(mock_data.tmpdir.glob("*.nc"))
    for n_steps in [3, 50]:
        dataset = XarrayDataset(config, mock_data.var_names.all_names, n_steps)
        assert len(dataset) == len(mock_data.obs_times) - n_steps + 1
        lon_dim, lat_dim = MOCK_DATA_LON_DIM, MOCK_DATA_LAT_DIM
        dims = ("time", lat_dim, lon_dim)
        shape = (n_steps, ds.sizes[lat_dim], ds.sizes[lon_dim])
        time_slice = slice(global_idx, global_idx + n_steps)
        for var_name in mock_data.var_names.spatial_resolved_names:
            da = ds[var_name]
            if var_name in mock_data.var_names.time_dependent_names:
                da = da.isel(time=time_slice)
            target_data = as_broadcasted_tensor(da.variable, dims, shape)
            target_times = ds["time"][global_idx : global_idx + n_steps].drop_vars(
                "time"
            )
            data, time, labels = dataset[global_idx]
            assert labels == labels
            data_tensor = data[var_name]
            assert data_tensor.shape[0] == n_steps
            assert torch.equal(data_tensor, target_data)
            xr.testing.assert_equal(time, target_times)


def test_dataset_dtype_casting(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir, dtype="bfloat16")
    dataset = XarrayDataset(config, mock_data.var_names.all_names, 2)
    data_properties = dataset.properties
    assert isinstance(data_properties.horizontal_coordinates, LatLonCoordinates)
    assert data_properties.horizontal_coordinates.lat.dtype == torch.bfloat16
    assert data_properties.horizontal_coordinates.lon.dtype == torch.bfloat16
    assert isinstance(
        data_properties.vertical_coordinate, HybridSigmaPressureCoordinate
    )
    assert data_properties.vertical_coordinate.ak.dtype == torch.bfloat16
    assert data_properties.vertical_coordinate.bk.dtype == torch.bfloat16
    data, _, _ = dataset[0]
    for tensor in data.values():
        assert tensor.dtype == torch.bfloat16


def test_time_invariant_variable_is_repeated(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    dataset = XarrayDataset(config, mock_data.var_names.all_names, 15)
    data = dataset[0][0]
    assert data["constant_var"].shape[0] == 15
    assert data["constant_scalar_var"].shape == (15, 4, 8)


def _get_repeat_dataset(
    mock_data: MockData, n_timesteps: int, n_repeats: int
) -> XarrayDataset:
    config = XarrayDataConfig(data_path=mock_data.tmpdir, n_repeats=n_repeats)
    return XarrayDataset(config, mock_data.var_names.all_names, n_timesteps)


@pytest.mark.parametrize("n_timesteps", [1, 4])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_repeat_dataset_num_timesteps(
    mock_monthly_netcdfs: MockData, n_timesteps, n_repeats
):
    unrepeated_dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, 1)
    data = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)
    offset = n_timesteps - 1
    expected_length = n_repeats * (len(unrepeated_dataset) + offset) - offset
    assert len(data) == expected_length


@pytest.mark.parametrize(
    "glob_pattern, expected_num_files, expected_year_month_tuples",
    [
        ("*.nc", None, None),
        ("2003030100.nc", 1, [(2003, 3)]),
        ("2003??0100.nc", 3, [(2003, i) for i in range(3, 6)]),
    ],
    ids=["all_files", "single_file", "all_2003_files"],
)
def test_glob_file_pattern(
    mock_monthly_netcdfs: MockData,
    glob_pattern,
    expected_num_files,
    expected_year_month_tuples,
):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, file_pattern=glob_pattern
    )
    dataset = XarrayDataset(config, mock_monthly_netcdfs.var_names.all_names, 2)
    if expected_num_files is None:
        expected_num_files = len(mock_monthly_netcdfs.start_times)
    assert expected_num_files == len(dataset.full_paths)

    if expected_year_month_tuples is not None:
        for i, (year, month) in enumerate(expected_year_month_tuples):
            assert f"{year}{month:02d}" in dataset.full_paths[i]


def test_time_slice():
    time_slice = TimeSlice("2001-01-01", "2001-01-05", 2)
    time_index = xr.date_range(
        "2000", "2002", freq="D", calendar="noleap", use_cftime=True
    )
    slice_ = time_slice.slice(time_index)
    assert slice_ == slice(365, 370, 2)


def test_time_index(mock_monthly_netcdfs):
    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = XarrayDataset(config, names, n_timesteps)
    last_sample_init_time = len(mock_monthly_netcdfs.obs_times) - n_timesteps + 1
    obs_times = mock_monthly_netcdfs.obs_times[:last_sample_init_time]
    assert dataset.sample_start_times.equals(xr.CFTimeIndex(obs_times))


@pytest.mark.parametrize("infer_timestep", [True, False])
def test_XarrayDataset_timestep(mock_monthly_netcdfs, infer_timestep):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, infer_timestep=infer_timestep
    )
    names = mock_monthly_netcdfs.var_names.all_names
    n_timesteps = 2
    dataset = XarrayDataset(config, names, n_timesteps)
    if infer_timestep:
        expected_timestep = pd.Timedelta(MOCK_DATA_FREQ).to_pytimedelta()
        assert dataset.timestep == expected_timestep
    else:
        assert dataset.timestep is None


@pytest.mark.parametrize(
    ("periods", "freq", "reverse", "expected", "exception"),
    [
        pytest.param(
            2,
            "3h",
            False,
            datetime.timedelta(hours=3),
            None,
            id="2 timesteps, regular freq",
        ),
        pytest.param(
            3,
            "9h",
            False,
            datetime.timedelta(hours=9),
            None,
            id="3 timesteps, regular freq",
        ),
        pytest.param(3, "3h", True, None, ValueError, id="3 timesteps, negative freq"),
        pytest.param(
            3, "MS", False, None, ValueError, id="3 timesteps, irregular freq"
        ),
        pytest.param(1, "D", False, None, ValueError, id="1 timestep"),
    ],
)
def test_get_timestep(periods, freq, reverse, expected, exception):
    index = xr.date_range("2000", periods=periods, freq=freq, use_cftime=True)

    if reverse:
        index = index[::-1]

    if exception is None:
        result = _get_timestep(index.values)
        assert result == expected
    else:
        with pytest.raises(exception):
            _get_timestep(index.values)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_repeat_and_increment_times(n_repeats):
    freq = "5h"
    delta = pd.Timedelta(freq).to_pytimedelta()

    start_a = cftime.DatetimeGregorian(2000, 1, 1)
    periods_a = 2
    segment_a = xr.date_range(
        start_a, periods=periods_a, freq=freq, use_cftime=True
    ).values

    start_b = segment_a[-1] + delta
    periods_b = 3
    segment_b = xr.date_range(
        start_b, periods=periods_b, freq=freq, use_cftime=True
    ).values

    raw_times = [segment_a, segment_b]
    raw_periods = [periods_a, periods_b]
    raw_total_periods = sum(raw_periods)

    result = _repeat_and_increment_time(raw_times, n_repeats, delta)
    full_periods = [len(times) for times in result]
    full_total_periods = sum(full_periods)

    result_concatenated = np.concatenate(result)
    expected_concatenated = xr.date_range(
        start_a, periods=full_total_periods, freq=freq, use_cftime=True
    ).values

    assert full_periods == n_repeats * raw_periods
    assert full_total_periods == n_repeats * raw_total_periods
    np.testing.assert_equal(result_concatenated, expected_concatenated)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_all_times(mock_monthly_netcdfs, n_repeats):
    n_timesteps = 2  # Arbitrary for this test
    dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)
    expected_periods = n_repeats * len(mock_monthly_netcdfs.obs_times)
    expected = xr.date_range(
        MOCK_DATA_START_DATE,
        periods=expected_periods,
        freq=MOCK_DATA_FREQ,
        use_cftime=True,
    )
    result = dataset.all_times
    assert result.equals(expected)


def test_get_sample_by_time_slice_times_n_repeats(mock_monthly_netcdfs: MockData):
    n_timesteps = 2  # Arbitrary for this test
    n_repeats = 3
    repeated_dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)

    # Pick a slice that is outside the range of the unrepeated data
    unrepeated_length = len(repeated_dataset.all_times) // n_repeats
    time_slice = slice(unrepeated_length, unrepeated_length + 3)

    _, result, _ = repeated_dataset.get_sample_by_time_slice(time_slice)
    expected = xr.DataArray(
        repeated_dataset.all_times[time_slice].values, dims=["time"]
    )
    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "dtype,expected_torch_dtype", [("int16", torch.int16), (None, None)]
)
def test_dataset_config_dtype(dtype, expected_torch_dtype):
    config = XarrayDataConfig(data_path="path/to/data", dtype=dtype)
    assert config.torch_dtype == expected_torch_dtype


def test_dataset_config_dtype_raises():
    with pytest.raises(ValueError):
        XarrayDataConfig(data_path="path/to/data", dtype="invalid_dtype")


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_with_nans", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_with_nans", "zarr", "*.zarr"),
    ],
)
def test_fill_nans(mock_data_fixture, engine, file_pattern, request):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    nan_config = FillNaNsConfig()
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        fill_nans=nan_config,
        engine=engine,
        file_pattern=file_pattern,
    )
    names = mock_data.var_names.all_names
    dataset = XarrayDataset(config, names, 2)
    data, _, _ = dataset[0]
    assert torch.all(data["foo"][0, :, 0] == 0)
    assert torch.all(data["constant_var"][:, 0, 0] == 0)


def test_keep_nans(mock_monthly_netcdfs_with_nans):
    config_keep_nan = XarrayDataConfig(data_path=mock_monthly_netcdfs_with_nans.tmpdir)
    names = mock_monthly_netcdfs_with_nans.var_names.all_names
    dataset = XarrayDataset(config_keep_nan, names, 2)
    data_with_nan, _, _ = dataset[0]
    assert torch.all(torch.isnan(data_with_nan["foo"][0, :, 0]))
    assert torch.all(torch.isnan(data_with_nan["constant_var"][:, 0, 0]))


def test_overwrite(mock_monthly_netcdfs):
    const = -10
    multiple = 3.5

    overwrite_config = OverwriteConfig(
        constant={"foo": const},
        multiply_scalar={"bar": multiple},
    )

    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = XarrayDataset(config, names, n_timesteps)[0][0]

    config_overwrite = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, overwrite=overwrite_config
    )
    n_timesteps = 2
    dataset_overwrite = XarrayDataset(config_overwrite, names, n_timesteps)[0][0]

    for v in ["foo", "bar"]:
        assert dataset_overwrite[v].dtype == dataset[v].dtype
        assert dataset_overwrite[v].device == dataset[v].device
    assert torch.equal(
        dataset_overwrite["foo"], torch.ones_like(dataset["foo"]) * const
    )
    assert torch.equal(dataset_overwrite["bar"], dataset["bar"] * multiple)


def test_repeated_interval_boolean_mask_subset(mock_monthly_netcdfs):
    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    names = mock_monthly_netcdfs.var_names.all_names
    dataset = XarrayDataset(config, names, 1)
    interval = RepeatedInterval(interval_length="1D", block_length="7D", start="3D")
    boolean_mask = interval.get_boolean_mask(len(dataset), dataset.timestep)
    subset = XarraySubset(dataset, boolean_mask)

    # Check that the subset length matches the expected number of intervals
    expected_length = boolean_mask.sum().item()
    assert len(subset) == expected_length


def test_multi_source_xarray_dataset_has_no_duplicates(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source
):
    monthly_netcdfs = [mock_monthly_netcdfs, mock_monthly_netcdfs_another_source]
    datasets = []

    for mock_data in monthly_netcdfs:
        config_source = XarrayDataConfig(data_path=mock_data.tmpdir)
        names = mock_data.var_names.all_names
        dataset = XarrayDataset(config_source, names, 1)
        datasets.append(dataset)

    with pytest.raises(ValueError):
        # duplicate variable names
        MergedXarrayDataset(datasets=datasets)


def test_multi_source_xarray_dataset_has_same_time(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source_diff_time
):
    monthly_netcdfs = [
        mock_monthly_netcdfs,
        mock_monthly_netcdfs_another_source_diff_time,
    ]

    datasets = []
    for mock_data in monthly_netcdfs:
        config_source = XarrayDataConfig(data_path=mock_data.tmpdir)
        names = mock_data.var_names.all_names
        dataset = XarrayDataset(config_source, names, 1)
        datasets.append(dataset)
    # different time index
    with pytest.raises(ValueError):
        MergedXarrayDataset(datasets=datasets)


def test_multi_source_xarray_returns_merged_data(
    mock_monthly_netcdfs, mock_monthly_netcdfs_another_source
):
    config_source1 = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    names1 = mock_monthly_netcdfs.var_names.all_names
    dataset1 = XarrayDataset(config_source1, names1, 1)

    config_source2 = XarrayDataConfig(
        data_path=mock_monthly_netcdfs_another_source.tmpdir
    )
    names2 = mock_monthly_netcdfs_another_source.var_names.all_names
    # remove duplicates in source 2 requirements
    for name in names1:
        if name in names2:
            names2.remove(name)
    dataset2 = XarrayDataset(config_source2, names2, 1)
    merged_dataset = MergedXarrayDataset(datasets=[dataset1, dataset2])
    assert len(merged_dataset) == len(dataset1)
    assert type(merged_dataset[0]) is type(dataset1[0])
    assert type(merged_dataset[0]) is type(dataset2[0])
    for key in merged_dataset[0][0].keys():
        if key in dataset1[0][0].keys():
            assert torch.equal(merged_dataset[0][0][key], dataset1[0][0][key])
            assert merged_dataset[0][1].equals(dataset1[0][1])
        if key in dataset2[0][0].keys():
            assert torch.equal(merged_dataset[0][0][key], dataset2[0][0][key])
            assert merged_dataset[0][1].equals(dataset2[0][1])
        else:
            assert KeyError(f"Key {key} is missing in merged dataset")


def test_xarray_subset_has_correct_sample(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    config2 = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(stop=1))
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    dataset, _ = get_xarray_dataset(config, names, n_timesteps)
    dataset2, _ = get_xarray_dataset(config2, names, n_timesteps)
    assert dataset.sample_start_times[0:1].equals(dataset2.sample_start_times)
    assert dataset[0][0]["foo"].equal(dataset2[0][0]["foo"])
    assert dataset[0][1].equals(dataset2[0][1])


def test_xarray_concat_has_correct_sample(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    config1 = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=TimeSlice("2003-03-01", "2003-03-31")
    )

    config2 = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=TimeSlice("2003-05-01", "2003-05-31")
    )
    concat, properties = get_dataset([config1, config2], names, n_timesteps)
    expected1, _ = get_xarray_dataset(config1, names, n_timesteps)
    expected2, _ = get_xarray_dataset(config2, names, n_timesteps)
    expected_times = np.concatenate(
        [expected1.sample_start_times, expected2.sample_start_times]
    )
    expected = xr.CFTimeIndex(expected_times)
    assert concat.sample_start_times.equals(expected)


def test__get_vertical_coordinate_raises():
    data = xr.Dataset({"ak_0": 1.0, "bk_0": 0.5, "idepth_0": 1.0})
    with pytest.raises(ValueError, match="Dataset contains both hybrid"):
        _get_vertical_coordinate(data, dtype=None)


def test__get_vertical_coordinate_null():
    data = xr.Dataset()
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert vertical_coordinate == NullVerticalCoordinate()


def test__get_vertical_coordinate_hybrid_sigma_pressure():
    data = xr.Dataset({"ak_0": 1.0, "bk_0": 0.5, "ak_1": 2.0, "bk_1": 1.5})
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, HybridSigmaPressureCoordinate)
    assert vertical_coordinate.ak[0] == 1.0
    assert vertical_coordinate.bk[0] == 0.5


def test__get_vertical_coordinate_depth_no_mask():
    data = xr.Dataset({"idepth_0": 1.0, "idepth_1": 2.0})
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, DepthCoordinate)
    assert vertical_coordinate.idepth[0] == 1.0
    assert vertical_coordinate.mask[0] == 1.0


def test__get_vertical_coordinate_depth_with_lat_dependent_mask():
    data = xr.Dataset(
        data_vars={
            "idepth_0": 1.0,
            "idepth_1": 2.0,
            "idepth_2": 3.0,
            "mask_0": ("lat", np.array([1.0, 1.0])),
            "mask_1": ("lat", np.array([0.0, 1.0])),
        },
        coords={
            "lat": np.array([1.0, 2.0]),
        },
    )
    vertical_coordinate = _get_vertical_coordinate(data, dtype=None)
    assert isinstance(vertical_coordinate, DepthCoordinate)
    assert vertical_coordinate.idepth[0] == 1.0
    assert vertical_coordinate.idepth.shape == (3,)
    assert vertical_coordinate.mask.shape == (2, 2)


def test__get_vertical_coordinate_depth_with_time_dependent_mask():
    data = xr.Dataset(
        data_vars={
            "idepth_0": 1.0,
            "idepth_1": 2.0,
            "idepth_2": 3.0,
            "mask_0": ("time", np.array([1.0, 1.0])),
            "mask_1": ("time", np.array([0.0, 1.0])),
        },
        coords={
            "time": np.array([1.0, 2.0]),
        },
    )
    with pytest.raises(ValueError, match="The ocean mask must by time-independent."):
        _get_vertical_coordinate(data, dtype=None)


@pytest.mark.parametrize(
    "kwargs,",
    [
        pytest.param({"spatial_dimensions": "xyz"}, id="invalid_spatial_dimensions"),
        pytest.param(
            {"engine": "zarr", "file_pattern": "*.nc"},
            id="engine_file_pattern_mismatch",
        ),
        pytest.param(
            {"n_repeats": 2, "infer_timestep": False}, id="n_repeats_infer_timestep"
        ),
        pytest.param({"dtype": "foo"}, id="invalid_dtype"),
    ],
)
def test_invalid_config_field_raises_error(kwargs):
    """Runs shape and length checks on the dataset."""
    with pytest.raises(ValueError):
        XarrayDataConfig(data_path="path", **kwargs)


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_dataset_with_nonspacetime_dim(
    mock_data_fixture, engine, file_pattern, request
):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        dtype="bfloat16",
        engine=engine,
        file_pattern=file_pattern,
    )
    # Omit the test variable that has mismatch dimensions
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    dataset = XarrayDataset(config, vars, 2)
    data, _, _ = dataset[0]
    assert len(data["foo"].shape) == 4
    assert dataset.dims == ["time", "sample", "lat", "lon"]


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_dataset_raise_error_on_dim_mismatch(
    mock_data_fixture, engine, file_pattern, request
):
    # Should raise error when trying to broadcast variable that is missing
    # ensemble 'sample' dim
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        dtype="bfloat16",
        engine=engine,
        file_pattern=file_pattern,
    )
    dataset = XarrayDataset(config, mock_data.var_names.all_names, 2)
    with pytest.raises(ValueError):
        dataset[0]


def test_xarray_raise_error_on_concat_dim_mismatch(
    mock_monthly_netcdfs, mock_monthly_netcdfs_ensemble_dim
):
    mock_data: MockData = mock_monthly_netcdfs
    mock_data_ensemble: MockData = mock_monthly_netcdfs_ensemble_dim
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    config1 = XarrayDataConfig(
        data_path=mock_data.tmpdir, subset=TimeSlice("2003-03-01", "2003-03-31")
    )
    config2 = XarrayDataConfig(
        data_path=mock_data_ensemble.tmpdir,
        subset=TimeSlice("2003-05-01", "2003-05-31"),
    )
    with pytest.raises(ValueError):
        get_dataset([config1, config2], names, n_timesteps)


@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs_ensemble_dim", "netcdf4", "*.nc"),
        ("mock_monthly_zarr_ensemble_dim", "zarr", "*.zarr"),
    ],
)
def test_xarray_dataset_isel(mock_data_fixture, engine, file_pattern, request):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        engine=engine,
        file_pattern=file_pattern,
        subset=Slice(start=None, stop=2),
        isel={"sample": 1},
    )
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    dataset = XarrayDataset(config, vars, 2)
    data, _, _ = dataset[0]
    # Original lat/lon sizes are 4, 8
    assert data["var_matches_sample_index"].shape == (2, 4, 8)
    assert data["constant_var"].shape == (2, 4, 8)
    assert "sample" not in dataset.dims
    assert torch.all(data["var_matches_sample_index"] == 1.0)


@pytest.mark.parametrize(
    "isel",
    [
        {"lat": 0},
        {"time": 0},
        {"grid_x": 0},
    ],
)
def test_xarray_dataset_invalid_isel_raises_error(
    mock_monthly_netcdfs_ensemble_dim, isel
):
    mock_data: MockData = mock_monthly_netcdfs_ensemble_dim
    names = mock_data.var_names.all_names

    with pytest.raises(ValueError):
        config = XarrayDataConfig(
            data_path=mock_data.tmpdir,
            subset=TimeSlice("2003-03-01", "2003-03-31"),
            isel=isel,
        )
        get_dataset([config], names, 5)


@pytest.mark.parametrize(
    "isel_value",
    [3, Slice(3, 13)],
)
def test_XarrayDataset_error_on_isel_outside_data(
    mock_monthly_netcdfs_ensemble_dim, isel_value
):
    # mock data has sample dimension size 3
    mock_data: MockData = mock_monthly_netcdfs_ensemble_dim
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir,
        subset=Slice(start=None, stop=2),
        isel={"sample": isel_value},
    )
    vars = list(set(mock_data.var_names.all_names) - {"var_no_ensemble_dim"})
    with pytest.raises(ValueError):
        XarrayDataset(config, vars, 2)


def test_concat_of_XarrayConcat(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    n_timesteps = 5
    names = mock_data.var_names.all_names + ["x"]
    config = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
    concat, _ = get_dataset([config, config], names, n_timesteps)
    concat2 = XarrayConcat(datasets=[concat, concat])
    assert len(concat2) == 16


def test_parallel__get_raw_times(tmpdir):
    times_per_file = 2
    n_files = GET_RAW_TIMES_NUM_FILES_PARALLELIZATION_THRESHOLD + 1
    n_times = n_files * times_per_file

    times = xr.date_range("2000", freq="6h", periods=n_times, use_cftime=True)
    da = xr.DataArray(range(len(times)), dims=["time"], coords=[times], name="foo")
    ds = da.to_dataset()

    paths = []
    for i in range(n_files):
        path = os.path.join(tmpdir, f"file_{i}.nc")
        time_slice = slice(times_per_file * i, times_per_file * (i + 1))
        ds.isel(time=time_slice).to_netcdf(path)
        paths.append(path)

    result = np.concatenate(_get_raw_times(paths, engine="netcdf4"))
    np.testing.assert_equal(result, times)


def test_dataset_properties_update_masks(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    dataset = XarrayDataset(config, mock_data.var_names.all_names, 2)
    data_properties = dataset.properties
    assert not data_properties.mask_provider.masks
    existing_mask = MaskProvider(masks={"mask_0": torch.ones(4, 8)})
    data_properties.update_mask_provider(existing_mask)
    assert "mask_0" in dataset.properties.mask_provider.masks
