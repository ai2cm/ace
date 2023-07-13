from fme.fcn_training.utils.data_loader_fv3gfs import VariableMetadata
from fme.fcn_training.utils.data_loader_params import DataLoaderParams
from fme.fcn_training.utils.data_requirements import DataRequirements
from fme.fcn_training.utils.data_loader_multifiles import get_data_loader
import numpy as np
import xarray as xr
from typing import Literal, Mapping, Optional
import tempfile
from pathlib import Path
import pytest


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
            np.arange(size, dtype=np.float32),
            dims=(dim_name,),
        )
        for dim_name, size in dim_sizes.items()
    }
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


def _create_data(path, metadata, data_type: Literal["FV3GFS", "ensemble"]):
    """Looks up function to create toy data for the given data type and runs it."""
    thunks = dict(
        FV3GFS=lambda: _save_netcdf(path / "data.nc", metadata),
        ensemble=lambda: _save_netcdf_ensemble(path, metadata, num_members=2),
    )
    create_data_fn = thunks[data_type]
    create_data_fn()


@pytest.mark.parametrize("metadata", METADATA)
@pytest.mark.parametrize("data_type", ["FV3GFS", "ensemble"])
def test_metadata(metadata, data_type):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        _create_data(path, metadata, data_type)

        params = DataLoaderParams(
            data_path=str(path), data_type=data_type, batch_size=1, num_data_workers=0
        )
        varnames = list(metadata.keys())
        requirements = DataRequirements(
            names=varnames, in_names=varnames, out_names=varnames, n_timesteps=2
        )
        _, dataset, _ = get_data_loader(  # type: ignore
            params=params, train=True, requirements=requirements
        )
        target_metadata = {
            name: metadata[name] for name in metadata if metadata[name] is not None
        }
        assert dataset.metadata == target_metadata  # type: ignore
