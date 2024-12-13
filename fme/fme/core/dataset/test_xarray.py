"""This file contains unit tests of XarrayDataset."""

import dataclasses
import datetime
from collections import namedtuple
from typing import Dict, Iterable, List, Union

import cftime
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.config import (
    FillNaNsConfig,
    OverwriteConfig,
    TimeSlice,
    XarrayDataConfig,
)
from fme.core.dataset.requirements import DataRequirements
from fme.core.dataset.xarray import (
    XarrayDataset,
    get_cumulative_timesteps,
    get_file_local_index,
    get_raw_times,
    get_timestep,
    get_xarray_dataset,
    repeat_and_increment_time,
)
from fme.core.typing_ import Slice

from .utils import as_broadcasted_tensor, infer_horizontal_dimension_names

SLICE_NONE = slice(None)
MOCK_DATA_FREQ = "3h"
MOCK_DATA_START_DATE = "2003-03"


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
    def all_names(self) -> List[str]:
        return self._concat(
            self.time_dependent_names,
            self.time_invariant_names,
            self.initial_condition_names,
        )

    @property
    def spatial_resolved_names(self) -> List[str]:
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
) -> MockData:
    """Constructs an xarray dataset and saves to disk in netcdf format."""
    obs_times = xr.cftime_range(
        start,
        end,
        freq=step_freq,
        calendar=calendar,
        inclusive="left",
    )
    start_times = xr.cftime_range(
        start,
        end,
        freq=file_freq,
        calendar=calendar,
        inclusive="left",
    )
    obs_delta = obs_times[1] - obs_times[0]
    n_levels = 2
    n_lat, n_lon = 4, 8
    var_names = ["foo", "bar"]
    constant_var = xr.DataArray(
        np.random.randn(n_lat, n_lon).astype(np.float32),
        dims=("lat", "lon"),
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
        time = xr.cftime_range(
            first, last, freq=step_freq, calendar=calendar, inclusive="left"
        )
        data_vars: Dict[str, Union[float, xr.DataArray]] = {**ak, **bk}
        for var_name in var_names:
            data = np.random.randn(len(time), n_lat, n_lon).astype(np.float32)
            if with_nans:
                data[0, :, 0] = np.nan
            data_vars[var_name] = xr.DataArray(data, dims=("time", "lat", "lon"))

        data_varying_scalar = np.random.randn(len(time)).astype(np.float32)
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

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        filename = tmpdir / f"{first.strftime('%Y%m%d%H')}.nc"
        ds.to_netcdf(
            filename,
            unlimited_dims=["time"],
            format="NETCDF4",
        )
        filenames.append(filename)

    initial_condition_names = ()
    start_indices = get_cumulative_timesteps(get_raw_times(filenames, "netcdf4"))

    variable_names = VariableNames(
        time_dependent_names=("foo", "bar", "varying_scalar_var"),
        time_invariant_names=("constant_var", "constant_scalar_var"),
        initial_condition_names=initial_condition_names,
    )
    return MockData(tmpdir, obs_times, start_times, start_indices, variable_names)


def get_mock_monthly_netcdfs(tmp_path_factory, dirname, with_nans=False) -> MockData:
    return _get_data(
        tmp_path_factory,
        dirname,
        start=MOCK_DATA_START_DATE,
        end="2003-06",
        file_freq="MS",
        step_freq=MOCK_DATA_FREQ,
        calendar="standard",
        with_nans=with_nans,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month")


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_with_nans(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month_with_nans", with_nans=True)


@pytest.fixture(scope="session")
def mock_monthly_zarr(tmp_path_factory, mock_monthly_netcdfs) -> MockData:
    zarr_parent = tmp_path_factory.mktemp("zarr")
    filename = "data.zarr"
    data = xr.open_mfdataset(
        mock_monthly_netcdfs.tmpdir.glob("*.nc"),
        use_cftime=True,
        data_vars="minimal",
        coords="minimal",
    )
    data.chunk({"time": 240}).to_zarr(zarr_parent / filename)
    return MockData(
        zarr_parent,
        mock_monthly_netcdfs.obs_times,
        mock_monthly_netcdfs.start_times,
        mock_monthly_netcdfs.start_indices,
        mock_monthly_netcdfs.var_names,
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
    file_local_idx = get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = np.datetime64(
        cftime.DatetimeGregorian(2003, 3, 1, 0, 0, 0, 0, has_year_zero=False)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(full_paths[file_idx], use_cftime=True) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


def _test_monthly_values(
    mock_data: MockData,
    global_idx,
    expected_n_samples=None,
    file_pattern="*.nc",
    engine="netcdf4",
):
    """Runs shape and length checks on the dataset."""
    var_names: VariableNames = mock_data.var_names
    config = XarrayDataConfig(
        data_path=mock_data.tmpdir, file_pattern=file_pattern, engine=engine
    )
    requirements = DataRequirements(names=var_names.all_names, n_timesteps=2)
    dataset = XarrayDataset(config=config, requirements=requirements)
    if expected_n_samples is None:
        expected_n_samples = len(mock_data.obs_times) - 1

    assert len(dataset) == expected_n_samples
    arrays, time = dataset[global_idx]
    with xr.open_mfdataset(
        mock_data.tmpdir.glob(file_pattern),
        engine=engine,
        use_cftime=True,
        data_vars="minimal",
        coords="minimal",
    ) as ds:
        target_times = ds["time"][global_idx : global_idx + 2].drop_vars("time")
        xr.testing.assert_equal(time, target_times)
        lon_dim, lat_dim = infer_horizontal_dimension_names(ds)
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


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(31 * 8 - 1, id="monthly_XarrayDataset_2003_03_31_21"),
        pytest.param((31 + 30 + 20) * 8 - 1, id="monthly_XarrayDataset_2003_05_20_21"),
    ],
)
@pytest.mark.parametrize(
    "mock_data_fixture, engine, file_pattern",
    [
        ("mock_monthly_netcdfs", "netcdf4", "*.nc"),
        ("mock_monthly_zarr", "zarr", "*.zarr"),
    ],
)
def test_XarrayDataset_monthly(
    global_idx, mock_data_fixture, engine, file_pattern, request
):
    mock_data: MockData = request.getfixturevalue(mock_data_fixture)
    _test_monthly_values(
        mock_data, global_idx, file_pattern=file_pattern, engine=engine
    )


@pytest.mark.parametrize("n_samples", [None, 1])
def test_XarrayDataset_monthly_n_timesteps(mock_monthly_netcdfs, n_samples):
    """Test that increasing n_timesteps decreases the number of samples."""
    mock_data: MockData = mock_monthly_netcdfs
    if len(mock_data.var_names.initial_condition_names) != 0:
        return
    config = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(stop=n_samples))
    n_forward_steps = 4
    requirements = DataRequirements(
        names=mock_data.var_names.all_names + ["x"],
        n_timesteps=n_forward_steps + 1,
    )
    dataset, _ = get_xarray_dataset(config, requirements)
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
    file_local_idx = get_file_local_index(global_idx, mock_data.start_indices)
    assert file_local_idx == expected_file_local_idx
    delta = mock_data.obs_times[1] - mock_data.obs_times[0]
    target_timestamp = (
        cftime.DatetimeNoLeap(1999, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(mock_data.tmpdir.glob("*.nc")))
    with xr.open_dataset(full_paths[file_idx], use_cftime=True) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(364, id="yearly_XarrayDataset_1999_12_31"),
        pytest.param(365 + 31 + 28, id="yearly_XarrayDataset_2000_02_28"),
    ],
)
def test_XarrayDataset_yearly(mock_yearly_netcdfs, global_idx):
    mock_data: MockData = mock_yearly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    with xr.open_mfdataset(
        mock_data.tmpdir.glob("*.nc"),
        use_cftime=True,
        data_vars="minimal",
        coords="minimal",
    ) as ds:
        for n_steps in [3, 50]:
            requirements = DataRequirements(
                names=mock_data.var_names.all_names,
                n_timesteps=n_steps,
            )
            dataset = XarrayDataset(config=config, requirements=requirements)
            assert len(dataset) == len(mock_data.obs_times) - n_steps + 1
            lon_dim, lat_dim = infer_horizontal_dimension_names(ds)
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
                data, time = dataset[global_idx]
                data_tensor = data[var_name]
                assert data_tensor.shape[0] == n_steps
                assert torch.equal(data_tensor, target_data)
                xr.testing.assert_equal(time, target_times)


def test_dataset_dtype_casting(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir, dtype="bfloat16")
    requirements = DataRequirements(names=mock_data.var_names.all_names, n_timesteps=2)
    dataset = XarrayDataset(config=config, requirements=requirements)
    assert isinstance(dataset.horizontal_coordinates, LatLonCoordinates)
    assert dataset.horizontal_coordinates.lat.dtype == torch.bfloat16
    assert dataset.horizontal_coordinates.lon.dtype == torch.bfloat16
    assert dataset.vertical_coordinate.ak.dtype == torch.bfloat16
    assert dataset.vertical_coordinate.bk.dtype == torch.bfloat16
    data, _ = dataset[0]
    for tensor in data.values():
        assert tensor.dtype == torch.bfloat16


def test_time_invariant_variable_is_repeated(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    config = XarrayDataConfig(data_path=mock_data.tmpdir)
    requirements = DataRequirements(names=mock_data.var_names.all_names, n_timesteps=15)
    dataset = XarrayDataset(config=config, requirements=requirements)
    data = dataset[0][0]
    assert data["constant_var"].shape[0] == 15
    assert data["constant_scalar_var"].shape == (15, 4, 8)


def _get_repeat_dataset(
    mock_data: MockData, n_timesteps: int, n_repeats: int
) -> XarrayDataset:
    config = XarrayDataConfig(data_path=mock_data.tmpdir, n_repeats=n_repeats)
    requirements = DataRequirements(
        names=mock_data.var_names.all_names, n_timesteps=n_timesteps
    )
    return XarrayDataset(config, requirements)


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
    requirements = DataRequirements(
        names=mock_monthly_netcdfs.var_names.all_names, n_timesteps=2
    )
    dataset = XarrayDataset(config, requirements)
    if expected_num_files is None:
        expected_num_files = len(mock_monthly_netcdfs.start_times)
    assert expected_num_files == len(dataset.full_paths)

    if expected_year_month_tuples is not None:
        for i, (year, month) in enumerate(expected_year_month_tuples):
            assert f"{year}{month:02d}" in dataset.full_paths[i]


def test_time_slice():
    time_slice = TimeSlice("2001-01-01", "2001-01-05", 2)
    time_index = xr.cftime_range("2000", "2002", freq="D", calendar="noleap")
    slice_ = time_slice.slice(time_index)
    assert slice_ == slice(365, 370, 2)


def test_time_index(mock_monthly_netcdfs):
    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    dataset = XarrayDataset(
        config,
        DataRequirements(
            names=mock_monthly_netcdfs.var_names.all_names, n_timesteps=n_timesteps
        ),
    )
    last_sample_init_time = len(mock_monthly_netcdfs.obs_times) - n_timesteps + 1
    obs_times = mock_monthly_netcdfs.obs_times[:last_sample_init_time]
    assert dataset.sample_start_time.equals(xr.CFTimeIndex(obs_times))


@pytest.mark.parametrize("infer_timestep", [True, False])
def test_XarrayDataset_timestep(mock_monthly_netcdfs, infer_timestep):
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, infer_timestep=infer_timestep
    )
    n_timesteps = 2
    dataset = XarrayDataset(
        config,
        DataRequirements(
            names=mock_monthly_netcdfs.var_names.all_names, n_timesteps=n_timesteps
        ),
    )
    if infer_timestep:
        expected_timestep = pd.Timedelta(MOCK_DATA_FREQ).to_pytimedelta()
        assert dataset.timestep == expected_timestep
    else:
        with pytest.raises(ValueError, match="Timestep was not inferred"):
            assert dataset.timestep


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
    index = xr.cftime_range("2000", periods=periods, freq=freq)

    if reverse:
        index = index[::-1]

    if exception is None:
        result = get_timestep(index.values)
        assert result == expected
    else:
        with pytest.raises(exception):
            get_timestep(index.values)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_repeat_and_increment_times(n_repeats):
    freq = "5h"
    delta = pd.Timedelta(freq).to_pytimedelta()

    start_a = cftime.DatetimeGregorian(2000, 1, 1)
    periods_a = 2
    segment_a = xr.cftime_range(start_a, periods=periods_a, freq=freq).values

    start_b = segment_a[-1] + delta
    periods_b = 3
    segment_b = xr.cftime_range(start_b, periods=periods_b, freq=freq).values

    raw_times = [segment_a, segment_b]
    raw_periods = [periods_a, periods_b]
    raw_total_periods = sum(raw_periods)

    result = repeat_and_increment_time(raw_times, n_repeats, delta)
    full_periods = [len(times) for times in result]
    full_total_periods = sum(full_periods)

    result_concatenated = np.concatenate(result)
    expected_concatenated = xr.cftime_range(
        start_a, periods=full_total_periods, freq=freq
    ).values

    assert full_periods == n_repeats * raw_periods
    assert full_total_periods == n_repeats * raw_total_periods
    np.testing.assert_equal(result_concatenated, expected_concatenated)


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_all_times(mock_monthly_netcdfs, n_repeats):
    n_timesteps = 2  # Arbitrary for this test
    dataset = _get_repeat_dataset(mock_monthly_netcdfs, n_timesteps, n_repeats)
    expected_periods = n_repeats * len(mock_monthly_netcdfs.obs_times)
    expected = xr.cftime_range(
        MOCK_DATA_START_DATE, periods=expected_periods, freq=MOCK_DATA_FREQ
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

    _, result = repeated_dataset.get_sample_by_time_slice(time_slice)
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


def test_renaming(mock_monthly_netcdfs):
    # stepper in/out names should be variables after renaming
    stepper_variables = ["foo", "bar_new"]
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, renamed_variables={"bar": "bar_new"}
    )
    dataset = XarrayDataset(
        config,
        DataRequirements(names=stepper_variables, n_timesteps=2),
    )
    data, _ = dataset[0]
    assert "bar_new" in data
    assert "bar" not in data


def test_fill_nans(mock_monthly_netcdfs_with_nans):
    nan_config = FillNaNsConfig()
    config = XarrayDataConfig(
        data_path=mock_monthly_netcdfs_with_nans.tmpdir, fill_nans=nan_config
    )
    requirements = DataRequirements(
        names=mock_monthly_netcdfs_with_nans.var_names.all_names, n_timesteps=2
    )
    dataset = XarrayDataset(config, requirements)
    data, _ = dataset[0]
    assert torch.all(data["foo"][0, :, 0] == 0)


def test_keep_nans(mock_monthly_netcdfs_with_nans):
    config_keep_nan = XarrayDataConfig(data_path=mock_monthly_netcdfs_with_nans.tmpdir)
    requirements = DataRequirements(
        names=mock_monthly_netcdfs_with_nans.var_names.all_names, n_timesteps=2
    )
    dataset = XarrayDataset(config_keep_nan, requirements)
    data_with_nan, _ = dataset[0]
    assert torch.all(torch.isnan(data_with_nan["foo"][0, :, 0]))


def test_overwrite(mock_monthly_netcdfs):
    const = -10
    multiple = 3.5

    overwrite_config = OverwriteConfig(
        constant={"foo": const},
        multiply_scalar={"bar": multiple},
    )

    config = XarrayDataConfig(data_path=mock_monthly_netcdfs.tmpdir)
    n_timesteps = 2
    dataset = XarrayDataset(
        config,
        DataRequirements(
            names=mock_monthly_netcdfs.var_names.all_names, n_timesteps=n_timesteps
        ),
    )[0][0]

    config_overwrite = XarrayDataConfig(
        data_path=mock_monthly_netcdfs.tmpdir, overwrite=overwrite_config
    )
    n_timesteps = 2
    dataset_overwrite = XarrayDataset(
        config_overwrite,
        DataRequirements(
            names=mock_monthly_netcdfs.var_names.all_names, n_timesteps=n_timesteps
        ),
    )[0][0]

    for v in ["foo", "bar"]:
        assert dataset_overwrite[v].dtype == dataset[v].dtype
        assert dataset_overwrite[v].device == dataset[v].device
    assert torch.equal(
        dataset_overwrite["foo"], torch.ones_like(dataset["foo"]) * const
    )
    assert torch.equal(dataset_overwrite["bar"], dataset["bar"] * multiple)


def test_overwrite_raises_error_on_original_name(mock_monthly_netcdfs):
    overwrite_config = OverwriteConfig(
        constant={"foo": 3},
    )
    with pytest.raises(ValueError):
        XarrayDataConfig(
            data_path=mock_monthly_netcdfs.tmpdir,
            overwrite=overwrite_config,
            renamed_variables={"foo": "foo_new"},
        )
