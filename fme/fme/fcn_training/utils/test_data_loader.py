from fme.fcn_training.utils.data_loader_fv3gfs import VariableMetadata, FV3GFSDataset
from fme.fcn_training.utils.data_loader_params import DataLoaderParams
from fme.fcn_training.utils.data_requirements import DataRequirements
import numpy as np
import xarray as xr
from typing import Mapping, Optional
import tempfile
from pathlib import Path
import pytest


def _save_netcdf(filename, metadata: Mapping[str, Optional[VariableMetadata]]):
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


@pytest.mark.parametrize(
    "metadata",
    [
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
    ],
)
def test_metadata(metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        _save_netcdf(path / "data.nc", metadata)
        params = DataLoaderParams(
            data_path=path, data_type="fv3gfs", batch_size=1, num_data_workers=0
        )
        varnames = list(metadata.keys())
        requirements = DataRequirements(
            names=varnames, in_names=varnames, out_names=varnames, n_timesteps=2
        )

        dataset = FV3GFSDataset(params=params, requirements=requirements)
        target_metadata = {
            name: metadata[name] for name in metadata if metadata[name] is not None
        }
        assert dataset.metadata == target_metadata
