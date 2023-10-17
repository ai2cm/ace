"""This file contains unit tests of XarrayDataset."""

from typing import Dict, Union

import cftime
import numpy as np
import pytest
import xarray as xr

from fme.core.data_loading._xarray import XarrayDataset, get_file_local_index
from fme.core.data_loading.get_loader import get_data_loader
from fme.core.data_loading.params import DataLoaderParams, Slice
from fme.core.data_loading.requirements import DataRequirements


def _mock_netcdf_factory(tmpdir, start, end, file_freq, step_freq, calendar):
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
    horizontal_dim_sizes = {"lat": 4, "lon": 8}
    var_names = ["foo", "bar"]
    constant_var = xr.DataArray(
        np.random.randn(
            horizontal_dim_sizes["lat"], horizontal_dim_sizes["lon"]
        ).astype(np.float32),
        dims=list(horizontal_dim_sizes),
    )
    ak = {f"ak_{i}": float(i) for i in range(n_levels)}
    bk = {f"bk_{i}": float(i + 1) for i in range(n_levels)}
    for i, first in enumerate(start_times):
        if first != start_times[-1]:
            last = start_times[i + 1]
        else:
            last = obs_times[-1] + obs_delta
        times = xr.cftime_range(
            first, last, freq=step_freq, calendar=calendar, closed="left"
        )
        dim_sizes = {"time": len(times), **horizontal_dim_sizes}
        data_vars: Dict[str, Union[float, xr.DataArray]] = {**ak, **bk}
        for varname in var_names:
            data = np.random.randn(
                dim_sizes["time"], dim_sizes["lat"], dim_sizes["lon"]
            ).astype(np.float32)
            data_vars[varname] = xr.DataArray(data, dims=list(dim_sizes))
        data_vars["constant_var"] = constant_var
        coords = {
            "time": xr.DataArray(times, dims=("time",)),
            "lat": xr.DataArray(
                np.arange(dim_sizes["lat"], dtype=np.float32), dims=("lat",)
            ),
            "lon": xr.DataArray(
                np.arange(dim_sizes["lon"], dtype=np.float32), dims=("lon",)
            ),
        }
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        ds.to_netcdf(
            tmpdir / f"{first.strftime('%Y%m%d%H')}.nc",
            unlimited_dims=["time"],
            format="NETCDF4",
        )

    all_var_names = var_names + ["constant_var"]
    return obs_times, start_times, all_var_names


@pytest.fixture(scope="session")
def mock_monthly_netcdfs(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("monthly")
    obs_times, start_times, var_names = _mock_netcdf_factory(
        tmpdir,
        start="2003-03",
        end="2005-06",
        file_freq="MS",
        step_freq="3H",
        calendar="standard",
    )
    return tmpdir, obs_times, start_times, var_names


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
    tmpdir, obs_times, start_times, _ = mock_monthly_netcdfs
    file_local_idx = get_file_local_index(global_idx, obs_times, start_times)
    assert file_local_idx == expected_file_local_idx
    delta = obs_times[1] - obs_times[0]
    target_timestamp = np.datetime64(
        cftime.DatetimeGregorian(2003, 3, 1, 0, 0, 0, 0, has_year_zero=False)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(tmpdir.glob("*.nc")))
    with xr.open_dataset(full_paths[file_idx], use_cftime=True) as ds:
        assert ds["time"][local_idx].item() == target_timestamp


@pytest.mark.parametrize(
    "global_idx",
    [
        pytest.param(31 * 8 - 1, id="monthly_XarrayDataset_2003_03_31_21"),
        pytest.param(366 * 8 - 1, id="monthly_XarrayDataset_2004_02_29_21"),
    ],
)
def test_XarrayDataset_monthly(mock_monthly_netcdfs, global_idx):
    tmpdir, obs_times, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(names=var_names, n_timesteps=2)
    dataset = XarrayDataset(params=params, requirements=requirements)
    assert len(dataset) == len(obs_times) - 1
    arrays, times = dataset[global_idx]
    with xr.open_mfdataset(tmpdir.glob("*.nc"), use_cftime=True) as ds:
        target_times = ds["time"][global_idx : global_idx + 2].drop_vars("time")
        xr.testing.assert_equal(times, target_times)
        for var_name in var_names:
            data = arrays[var_name].detach().numpy()
            assert data.shape[0] == 2
            target_data = ds[var_name][global_idx : global_idx + 2, :, :].values
            assert np.all(data == target_data)


def test_XarrayDataset_monthly_n_timesteps(mock_monthly_netcdfs):
    """Test that increasing n_timesteps decreases the number of samples."""
    tmpdir, obs_times, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    n_forward_steps = 4
    requirements = DataRequirements(
        names=var_names,
        n_timesteps=n_forward_steps + 1,
    )
    dataset = XarrayDataset(params=params, requirements=requirements)
    assert len(dataset) == len(obs_times) - n_forward_steps


def test_XarrayDataset_monthly_start_slice(mock_monthly_netcdfs):
    """
    When initial conditions are only taken from a certain start point, there should
    be fewer samples.
    """
    tmpdir, obs_times, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        window_starts=Slice(5, None),
    )
    requirements = DataRequirements(names=var_names, n_timesteps=2)
    dataset = XarrayDataset(params=params, requirements=requirements)
    assert len(dataset) == len(obs_times) - 1 - 5


@pytest.mark.parametrize(
    "n_forward_steps",
    [1, 2, 5],
)
def test_XarrayDataset_monthly_step_slice(mock_monthly_netcdfs, n_forward_steps):
    """
    When we subsample initial conditions every N steps, there should be fewer samples.
    """
    tmpdir, obs_times, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
        window_starts=Slice(None, None, 2),
    )
    requirements = DataRequirements(
        names=var_names,
        n_timesteps=n_forward_steps + 1,
    )
    dataset = XarrayDataset(params=params, requirements=requirements)
    n_all_samples = len(obs_times) - n_forward_steps
    # +1 because if the number of samples is odd, we include the first and last sample
    assert len(dataset) == int((n_all_samples + 1) / 2)


def test_XarrayDataset_monthly_time_window_sample_length(mock_monthly_netcdfs):
    tmpdir, _, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(names=var_names, n_timesteps=120)
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


@pytest.fixture(scope="session")
def mock_yearly_netcdfs(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("yearly")
    obs_times, start_times, var_names = _mock_netcdf_factory(
        tmpdir,
        start="1999",
        end="2005",
        file_freq="YS",
        step_freq="1D",
        calendar="noleap",
    )
    return tmpdir, obs_times, start_times, var_names


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
    tmpdir, obs_times, start_times, _ = mock_yearly_netcdfs
    file_local_idx = get_file_local_index(global_idx, obs_times, start_times)
    assert file_local_idx == expected_file_local_idx
    delta = obs_times[1] - obs_times[0]
    target_timestamp = (
        cftime.DatetimeNoLeap(1999, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        + global_idx * delta
    )
    file_idx, local_idx = file_local_idx
    full_paths = sorted(list(tmpdir.glob("*.nc")))
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
    tmpdir, obs_times, _, var_names = mock_yearly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    with xr.open_mfdataset(tmpdir.glob("*.nc"), use_cftime=True) as ds:
        for n_steps in [3, 2 * 365]:
            requirements = DataRequirements(
                names=var_names,
                n_timesteps=n_steps,
            )
            dataset = XarrayDataset(params=params, requirements=requirements)
            assert len(dataset) == len(obs_times) - n_steps + 1
            for varname in var_names:
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
    tmpdir, _, _, var_names = mock_monthly_netcdfs
    params = DataLoaderParams(
        data_path=tmpdir,
        data_type="xarray",
        batch_size=1,
        num_data_workers=0,
    )
    requirements = DataRequirements(names=var_names, n_timesteps=15)
    data = get_data_loader(params=params, train=False, requirements=requirements)
    batch, _ = data.loader.dataset[0]
    assert batch["constant_var"].shape[0] == 15
