"""This file contains unit tests related to creating torch Datasets from climate
data (e.g. netCDF files)."""

import datetime
import pathlib
from typing import List

import numpy as np
import pytest
import xarray as xr

from fme.core.data_loading.get_loader import get_data_loader
from fme.core.data_loading.params import DataLoaderParams
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.data_loading.utils import apply_slice


def _coord_value(name, size):
    # xarray data loader requires time to be a datetime or cftime.datetime object
    if name == "time":
        return [
            datetime.datetime(2000, 1, 1) + datetime.timedelta(hours=i)
            for i in range(size)
        ]
    else:
        return np.arange(size, dtype=np.float32)


def _save_netcdf(filename, dim_sizes, variable_names):
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)  # type: ignore
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes), attrs={"units": "m", "long_name": name}
        )
    coords = {
        dim_name: xr.DataArray(
            _coord_value(dim_name, size),
            dims=(dim_name,),
        )
        for dim_name, size in dim_sizes.items()
    }

    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")


def _create_dataset_on_disk(data_dir: pathlib.Path) -> pathlib.Path:
    seed = 0
    np.random.seed(seed)
    in_variable_names = ["foo", "bar", "baz"]
    out_variable_names = ["foo", "bar"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names)) + [mask_name]
    data_dim_sizes = {"time": 3, "grid_yt": 16, "grid_xt": 32}

    data_path = data_dir / "data.nc"
    _save_netcdf(data_path, data_dim_sizes, all_variable_names)

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

    params = DataLoaderParams(tmp_path, "ensemble_xarray", 1, 0, None)
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], [], [], window_timesteps)

    n_timesteps = 3  # hard coded to match `_create_dataset_on_disk`.
    samples_per_member = n_timesteps - window_timesteps + 1

    data = get_data_loader(params, True, requirements)
    assert len(data.loader) == samples_per_member * num_ensemble_members
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


def test_xarray_loader(tmp_path):
    """Checks that sigma coordinates are present."""
    _create_dataset_on_disk(tmp_path)
    params = DataLoaderParams(tmp_path, "xarray", 1, 0, None)
    window_timesteps = 2  # 1 initial condition and 1 step forward
    requirements = DataRequirements(["foo"], [], [], window_timesteps)
    data = get_data_loader(params, True, requirements)  # type: ignore
    assert isinstance(data.sigma_coordinates, SigmaCoordinates)


@pytest.mark.parametrize(
    "outer_slice, inner_slice, expected",
    [
        pytest.param(
            slice(0, 2),
            slice(0, 2),
            slice(0, 2),
            id="slice_0_2",
        ),
        pytest.param(
            slice(0, 2),
            slice(0, 1),
            slice(0, 1),
            id="slice_0_1",
        ),
        pytest.param(
            slice(0, 2),
            slice(1, 2),
            slice(1, 2),
            id="slice_1_2",
        ),
        pytest.param(
            slice(1, 3),
            slice(0, 5),
            slice(1, 3),
            id="slice_inner_past_end",
        ),
        pytest.param(
            slice(5, 10),
            slice(1, 3),
            slice(6, 8),
            id="slice_5_10_1_3",
        ),
        pytest.param(
            slice(5, 10),
            slice(7, 9),
            slice(10, 10),
            id="slice_out_of_range",
        ),
    ],
)
def test_apply_slice(outer_slice, inner_slice, expected):
    result = apply_slice(outer_slice, inner_slice)
    assert result == expected
