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
from fme.core.distributed import Distributed
import torch_harmonics.distributed as thd
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

from fme.core.dataset.test_helper import gather_helper_conv, relative_error, init_seed
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
    #NOTE: fixing random seed
    np.random.seed(333)
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
# TODO: Make this run with a Bash script; I am running this test manually.
# 1. Get an interactive node in PM.
# 2. Then srun -n 4 pytest test_xarray_sp_dist.py.
def test_concat_of_XarrayConcat_w_spatial_parallel(mock_monthly_netcdfs):
    # We must use the same random seed because this code will be executed several times.
    init_seed(333)
    mock_data: MockData = mock_monthly_netcdfs
    n_timesteps = 5
    names = mock_data.var_names.all_names
    ## without domain decomposition
    dist = Distributed()
    config_ref = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
    ref, _ = get_dataset([config_ref], names, n_timesteps)
    niters= len(ref)
    tensor_refs=[]
    for i in range(niters):
      ref_t, _, _=ref[i]
      for var in ref_t:
        reft = ref_t[var]
        # NOTE: We need to make a hard copy because the reference gets overwritten.
        tensor_refs.append(reft.clone())

    dist.shutdown()
    # from mpi4py import MPI
    # mpi_comm = MPI.COMM_WORLD.Dup()
    # mpi_comm.Barrier()
    # mpi_comm_rank = mpi_comm.Get_rank()
    ## with domain decomposition
    dist = Distributed()
    h_parallel_size=2
    w_parallel_size=2
    dist._init_distributed(h_parallel_size =  h_parallel_size, w_parallel_size=w_parallel_size)
    thd.init(h_parallel_size, w_parallel_size)
    comm = dist.get_comm()
    w_group = comm.get_group("w")
    h_group = comm.get_group("h")
    config_c1 = XarrayDataConfig(data_path=mock_data.tmpdir, subset=Slice(None, 4))
    c1, _ = get_dataset([config_c1], names, n_timesteps)

    # mpi_comm.Barrier()
    with torch.no_grad():
      niters= len(ref)
      j=0
      for i in range(niters):
        t1,_,_=c1[i]
        for var in ref_t:
          reft = tensor_refs[j]
          j+=1
          c1t = t1[var]
          # NOTE: only check variables w time, lat, and lon
          if len(c1t.shape) > 3:
            #gather_helper_conv assumes that the distribution is across the GPUs.
            c1t=c1t.to(dist.local_rank)
            c1t_full = gather_helper_conv(c1t, hdim=-2, wdim=-1, w_group=w_group, h_group=h_group)
            # Get back to the CPU so that it can be compared with the reference.
            c1t_full_cpu=c1t_full.to("cpu")
            err = relative_error(c1t_full_cpu, reft)
            if (dist.local_rank == 0):
              print(var, f"final relative error of output: {err.item()}")
              this_shape=c1t_full_cpu.shape
              for f in range(this_shape[0]):
                for g in range(this_shape[1]):
                  for k in range(this_shape[2]):
                    diff = abs(c1t_full_cpu[f,g,k]-reft[f,g,k])
                    if diff > 1e-12:
                      print(f,g, k, " : " ,c1t_full_cpu[f,g,k], reft[f,g,k])
            assert torch.equal(c1t_full_cpu,reft)
