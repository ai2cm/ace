"""This file contains unit tests of XarrayDataset."""

import contextlib
import dataclasses
from collections import namedtuple
from typing import Dict, Iterable, List, Union

import cftime
import numpy as np
import pytest
import xarray as xr

from fme.core.data_loading._xarray import (
    XarrayDataset,
    get_cumulative_timesteps,
    get_file_local_index,
)
from fme.core.data_loading.get_loader import get_data_loader
from fme.core.data_loading.params import DataLoaderParams, Slice
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.typing import GriddedData


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
    def time_resolved_names(self) -> List[str]:
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
    has_ragged_var,
) -> MockData:
    """Constructs an xarray dataset and saves to disk in netcdf format."""
    obs_times = xr.cftime_range(
        start,
        end,
        freq=step_freq,
        calendar=calendar,
        closed="left",
    )
    start_times = xr.cftime_range(
        start,
        end,
        freq=file_freq,
        calendar=calendar,
        closed="left",
    )
    obs_delta = obs_times[1] - obs_times[0]
    n_levels = 2
    n_lat, n_lon = 4, 8
    var_names = ["foo", "bar"]
    constant_var = xr.DataArray(
        np.random.randn(n_lat, n_lon).astype(np.float32),
        dims=("lat", "lon"),
    )
    ak = {f"ak_{i}": float(i) for i in range(n_levels)}
    bk = {f"bk_{i}": float(i + 1) for i in range(n_levels)}
    tmpdir = tmp_path_factory.mktemp(dirname)
    filenames = []
    for i, first in enumerate(start_times):
        if first != start_times[-1]:
            last = start_times[i + 1]
        else:
            last = obs_times[-1] + obs_delta
        times = xr.cftime_range(
            first, last, freq=step_freq, calendar=calendar, closed="left"
        )
        data_vars: Dict[str, Union[float, xr.DataArray]] = {**ak, **bk}
        for varname in var_names:
            data = np.random.randn(len(times), n_lat, n_lon).astype(np.float32)
            data_vars[varname] = xr.DataArray(data, dims=("time", "lat", "lon"))

        data_vars["constant_var"] = constant_var
        if i == 0 and has_ragged_var:
            rand_data = np.random.randn(n_lat, n_lon).astype(np.float32)
            data_vars["ragged_var"] = xr.DataArray(
                np.expand_dims(rand_data, axis=0),
                dims=["initial_condition", "lat", "lon"],
            )

        coords = {
            "time": xr.DataArray(times, dims=("time",)),
            "lat": xr.DataArray(np.arange(n_lat, dtype=np.float32), dims=("lat",)),
            "lon": xr.DataArray(np.arange(n_lon, dtype=np.float32), dims=("lon",)),
        }

        if i == 0 and has_ragged_var:
            coords["initial_condition"] = xr.DataArray(
                [times[0]], dims=("initial_condition",)
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        filename = tmpdir / f"{first.strftime('%Y%m%d%H')}.nc"
        ds.to_netcdf(
            filename,
            unlimited_dims=["time"],
            format="NETCDF4",
        )
        filenames.append(filename)

    if has_ragged_var:
        initial_condition_names: Iterable[str] = ("ragged_var",)
    else:
        initial_condition_names = ()

    start_indices = get_cumulative_timesteps(filenames)

    variable_names = VariableNames(
        time_dependent_names=("foo", "bar"),
        time_invariant_names=("constant_var",),
        initial_condition_names=initial_condition_names,
    )
    return MockData(tmpdir, obs_times, start_times, start_indices, variable_names)


def get_mock_monthly_netcdfs(tmp_path_factory, dirname, has_ragged_var) -> MockData:
    return _get_data(
        tmp_path_factory,
        dirname,
        start="2003-03",
        end="2005-06",
        file_freq="MS",
        step_freq="3H",
        calendar="standard",
        has_ragged_var=has_ragged_var,
    )


@pytest.fixture(scope="session")
def mock_monthly_netcdfs(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "month", False)


@pytest.fixture(scope="session")
def mock_monthly_netcdfs_ragged_time_dim(tmp_path_factory) -> MockData:
    return get_mock_monthly_netcdfs(tmp_path_factory, "ragged", True)


@pytest.fixture(scope="session")
def mock_yearly_netcdfs(tmp_path_factory):
    return _get_data(
        tmp_path_factory,
        "yearly",
        start="1999",
        end="2005",
        file_freq="YS",
        step_freq="1D",
        calendar="noleap",
        has_ragged_var=False,
    )


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(0, (0, 0), id="monthly_file_local_idx_2003_03_01_00"),
        pytest.param(1, (0, 1), id="monthly_file_local_idx_2003_03_01_03"),
        pytest.param(30 * 8, (0, 30 * 8), id="monthly_file_local_idx_2003_03_31_00"),
        pytest.param(31 * 8, (1, 0), id="monthly_file_local_idx_2003_04_01_00"),
        pytest.param(
            366 * 8 - 7,
            (11, 28 * 8 + 1),
            id="monthly_file_local_idx_2004_02_29_03",
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
    mock_data: MockData, global_idx, expected_n_samples=None, is_ragged=False
):
    """Runs shape and length checks on the dataset."""
    var_names: VariableNames = mock_data.var_names
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(names=var_names.all_names, n_timesteps=2)
    dataset = XarrayDataset(params=params, requirements=requirements)
    if expected_n_samples is None:
        expected_n_samples = len(mock_data.obs_times) - 1

    assert len(dataset) == expected_n_samples
    arrays, times = dataset[global_idx]
    with xr.open_mfdataset(mock_data.tmpdir.glob("*.nc"), use_cftime=True) as ds:
        target_times = ds["time"][global_idx : global_idx + 2].drop_vars("time")
        xr.testing.assert_equal(times, target_times)
        for var_name in var_names.time_resolved_names:
            data = arrays[var_name].detach().numpy()
            assert data.shape[0] == 2
            target_data = ds[var_name][global_idx : global_idx + 2, :, :].values
            assert np.all(data == target_data)

        for var_name in mock_data.var_names.initial_condition_names:
            data = arrays[var_name].detach().numpy()
            assert data.shape[0] == 1
            target_data = ds[var_name][global_idx : global_idx + 1, :, :].values
            assert np.all(data == target_data)


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(31 * 8 - 1, id="monthly_XarrayDataset_2003_03_31_21"),
        pytest.param(366 * 8 - 1, id="monthly_XarrayDataset_2004_02_29_21"),
    ],
)
def test_XarrayDataset_monthly(mock_monthly_netcdfs, global_idx):
    mock_data: MockData = mock_monthly_netcdfs
    _test_monthly_values(mock_data, global_idx)


@pytest.mark.parametrize(
    "global_idx,error_context",
    [(0, contextlib.nullcontext()), (1, pytest.raises(IndexError))],
)
def test_ragged_dataset_indexing(
    mock_monthly_netcdfs_ragged_time_dim, global_idx, error_context
):
    """
    Should only be allowed to select from a dataset with ragged time_dim at
    index = 0.
    """
    mock_data: MockData = mock_monthly_netcdfs_ragged_time_dim
    assert len(list(mock_data.var_names.initial_condition_names)) > 0  # sanity check
    with error_context:
        _test_monthly_values(mock_data, global_idx=global_idx, expected_n_samples=1)


@pytest.mark.parametrize(
    "n_samples,error_context",
    [
        (None, contextlib.nullcontext()),
        (1, contextlib.nullcontext()),
        (2, pytest.raises(ValueError)),
    ],
)
def test_XarrayDataset_ragged_raises_error(
    mock_monthly_netcdfs_ragged_time_dim, n_samples, error_context
):
    """Check that you are only allowed to specify n_samples = 1 or None."""
    mock_data: MockData = mock_monthly_netcdfs_ragged_time_dim

    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        n_samples=n_samples,
    )
    requirements = DataRequirements(names=mock_data.var_names.all_names, n_timesteps=2)

    with error_context:
        XarrayDataset(params=params, requirements=requirements)


@pytest.mark.parametrize("n_samples", [None, 1])
def test_XarrayDataset_monthly_n_timesteps(mock_monthly_netcdfs, n_samples):
    """Test that increasing n_timesteps decreases the number of samples."""
    mock_data: MockData = mock_monthly_netcdfs
    if len(mock_data.var_names.initial_condition_names) != 0:
        return
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        n_samples=n_samples,
    )
    n_forward_steps = 4
    requirements = DataRequirements(
        names=mock_data.var_names.all_names,
        n_timesteps=n_forward_steps + 1,
    )
    dataset = XarrayDataset(params=params, requirements=requirements)
    if n_samples is None:
        assert len(dataset) == len(mock_data.obs_times) - n_forward_steps
    else:
        assert len(dataset) == n_samples


def test_XarrayDataset_monthly_start_slice(mock_monthly_netcdfs):
    """
    When initial conditions are only taken from a certain start point, there should
    be fewer samples.
    """
    mock_data: MockData = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        window_starts=Slice(5, None),
    )
    requirements = DataRequirements(names=mock_data.var_names.all_names, n_timesteps=2)
    dataset = XarrayDataset(params=params, requirements=requirements)
    assert len(dataset) == len(mock_data.obs_times) - 1 - 5


@pytest.mark.parametrize(
    "n_forward_steps",
    [1, 2, 5],
)
def test_XarrayDataset_monthly_step_slice(mock_monthly_netcdfs, n_forward_steps):
    """
    When we subsample initial conditions every N steps, there should be fewer samples.
    """
    mock_data: MockData = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        window_starts=Slice(None, None, 2),
    )
    requirements = DataRequirements(
        names=mock_data.var_names.all_names,
        n_timesteps=n_forward_steps + 1,
    )
    dataset = XarrayDataset(params=params, requirements=requirements)
    n_all_samples = len(mock_data.obs_times) - n_forward_steps
    # +1 because if the number of samples is odd, we include the first and last sample
    assert len(dataset) == int((n_all_samples + 1) / 2)


def test_XarrayDataset_monthly_time_window_sample_length(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(
        names=mock_data.var_names.all_names, n_timesteps=120
    )
    data = get_data_loader(
        params=params,
        train=False,
        requirements=requirements,
        window_time_slice=slice(80, 120),
    )
    batch, times = data.loader.dataset[129]
    assert batch["foo"].shape[0] == 40  # time window should be length 40
    assert batch["bar"].shape[0] == 40
    assert len(times) == 40


@pytest.mark.parametrize(
    "global_idx,expected_file_local_idx",
    [
        pytest.param(365 + 59, (1, 59), id="yearly_file_local_idx_2000_03_01"),
        pytest.param(2 * 365, (2, 0), id="yearly_file_local_idx_2001_01_01"),
        pytest.param(4 * 365 + 364, (4, 364), id="yearly_file_local_idx_2003_12_31"),
        pytest.param(5 * 365 + 235, (5, 235), id="yearly_file_local_idx_2004_08_24"),
        pytest.param(5 * 365 + 358, (5, 358), id="yearly_file_local_idx_2004_12_25"),
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
        pytest.param(365 * 2 + 364, id="yearly_XarrayDataset_2001_12_31"),
        pytest.param(365 * 3 + 363, id="yearly_XarrayDataset_2002_12_30"),
    ],
)
def test_XarrayDataset_yearly(mock_yearly_netcdfs, global_idx):
    mock_data: MockData = mock_yearly_netcdfs
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    with xr.open_mfdataset(mock_data.tmpdir.glob("*.nc"), use_cftime=True) as ds:
        for n_steps in [3, 2 * 365]:
            requirements = DataRequirements(
                names=mock_data.var_names.all_names,
                n_timesteps=n_steps,
            )
            dataset = XarrayDataset(params=params, requirements=requirements)
            assert len(dataset) == len(mock_data.obs_times) - n_steps + 1
            for varname in mock_data.var_names.time_resolved_names:
                target_data = ds[varname][
                    global_idx : global_idx + n_steps, :, :
                ].values
                target_times = ds["time"][global_idx : global_idx + n_steps].drop_vars(
                    "time"
                )
                data, times = dataset[global_idx]
                data = data[varname].detach().numpy()
                assert data.shape[0] == n_steps
                assert np.all(data == target_data)
                xr.testing.assert_equal(times, target_times)


def test_time_invariant_variable_is_repeated(mock_monthly_netcdfs):
    mock_data: MockData = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(names=mock_data.var_names.all_names, n_timesteps=15)
    data = get_data_loader(params=params, train=False, requirements=requirements)
    batch, _ = data.loader.dataset[0]
    assert batch["constant_var"].shape[0] == 15


def _get_repeat_data_loader(
    mock_data: MockData, n_timesteps: int, n_repeats: int
) -> GriddedData:
    params = DataLoaderParams(
        data_path=mock_data.tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        n_repeats=n_repeats,
    )
    requirements = DataRequirements(
        names=mock_data.var_names.all_names, n_timesteps=n_timesteps
    )
    return get_data_loader(params=params, train=False, requirements=requirements)


@pytest.mark.parametrize("n_timesteps", [1, 2, 4])
@pytest.mark.parametrize("n_repeats", [1, 2])
def test_repeat_dataset_num_timesteps(
    mock_monthly_netcdfs: MockData, n_timesteps, n_repeats
):
    unrepeated_loader = _get_repeat_data_loader(mock_monthly_netcdfs, n_timesteps, 1)
    data = _get_repeat_data_loader(mock_monthly_netcdfs, n_timesteps, n_repeats)
    offset = n_timesteps - 1
    expected_length = (
        n_repeats * (len(unrepeated_loader.loader.dataset) + offset) - offset
    )
    assert len(data.loader.dataset) == expected_length
