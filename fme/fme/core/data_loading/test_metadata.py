import datetime
import tempfile
from pathlib import Path
from typing import Literal, Mapping, Optional

import numpy as np
import pytest
import xarray as xr

from fme.core.data_loading.get_loader import get_data_loader
from fme.core.data_loading.params import DataLoaderParams
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.typing import VariableMetadata

METADATA = [
    pytest.param(
        {"bar": None},
        id="one_var_no_metadata",
    ),
    pytest.param(
        {"bar": VariableMetadata("km", "bar_long_name")},
        id="one_var_metadata",
    ),
    pytest.param(
        {"foo": VariableMetadata("m", "foo_long_name"), "bar": None},
        id="two_vars_one_metadata",
    ),
    pytest.param(
        {
            "foo": VariableMetadata("m", "foo_long_name"),
            "bar": VariableMetadata("km", "bar_long_name"),
        },
        id="two_vars_two_metadata",
    ),
]


def _coord_value(name, size):
    # xarray data loader requires time to be a datetime or cftime.datetime object
    if name == "time":
        return [
            datetime.datetime(2000, 1, 1) + datetime.timedelta(hours=i)
            for i in range(size)
        ]
    else:
        return np.arange(size, dtype=np.float32)


def _save_netcdf(
    filename, metadata: Mapping[str, Optional[VariableMetadata]], num_members=1
):
    dim_sizes = {"time": 3, "grid_yt": 16, "grid_xt": 32}
    data_vars = {}
    for name in metadata:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)
        item_metadata = metadata[name]
        if item_metadata is None:
            attrs = {}
        else:
            attrs = {
                "units": item_metadata.units,
                "long_name": item_metadata.long_name,
            }
        data_vars[name] = xr.DataArray(data, dims=list(dim_sizes), attrs=attrs)
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


def _save_netcdf_ensemble(
    basename,
    metadata: Mapping[str, Optional[VariableMetadata]],
    num_members=1,
    member_prefix="member",
):
    for i in range(num_members):
        member_path = basename / f"{member_prefix}{i}"
        member_path.mkdir()
        _save_netcdf(member_path / "data.nc", metadata)


def _create_data(path, metadata, data_type: Literal["xarray", "ensemble_xarray"]):
    """Looks up function to create toy data for the given data type and runs it."""
    thunks = dict(
        xarray=lambda: _save_netcdf(path / "data.nc", metadata),
        ensemble_xarray=lambda: _save_netcdf_ensemble(path, metadata, num_members=2),
    )
    create_data_fn = thunks[data_type]
    create_data_fn()


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("data_type", ["xarray", "ensemble_xarray"])
def test_metadata(metadata, data_type):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        _create_data(path, metadata, data_type)

        params = DataLoaderParams(
            data_path=str(path), data_type=data_type, batch_size=1, num_data_workers=0
        )
        var_names = list(metadata.keys())
        requirements = DataRequirements(
            names=var_names, in_names=var_names, out_names=var_names, n_timesteps=2
        )
        data = get_data_loader(params=params, train=True, requirements=requirements)
        target_metadata = {
            name: metadata[name] for name in metadata if metadata[name] is not None
        }
        assert data.metadata == target_metadata  # type: ignore
