import datetime
from typing import Mapping, Optional

import cftime
import numpy as np
import pytest
import xarray as xr

from fme.core.data_loading.config import DataLoaderConfig, XarrayDataConfig
from fme.core.data_loading.data_typing import VariableMetadata
from fme.core.data_loading.getters import get_data_loader
from fme.core.data_loading.requirements import DataRequirements

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
    # xarray data loader requires time to be cftime.datetime object
    if name == "time":
        return [
            cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(hours=i)
            for i in range(size)
        ]
    else:
        return np.arange(size, dtype=np.float32)


def _save_netcdf(
    filename,
    metadata: Mapping[str, Optional[VariableMetadata]],
    num_members=1,
    dim_sizes=None,
):
    if dim_sizes is None:
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


@pytest.mark.parametrize("n_ensemble_members", [1, 2])
@pytest.mark.parametrize("metadata", METADATA)
def test_metadata(tmp_path, metadata, n_ensemble_members):
    paths = []
    for i in range(n_ensemble_members):
        path = tmp_path / f"ic{i}"
        path.mkdir(exist_ok=True)
        paths.append(path)
        _save_netcdf(path / "data.nc", metadata)

    config = DataLoaderConfig(
        [XarrayDataConfig(data_path=str(path)) for path in paths],
        batch_size=1,
        num_data_workers=0,
    )
    var_names = list(metadata.keys())
    requirements = DataRequirements(names=var_names, n_timesteps=2)
    data = get_data_loader(config=config, train=True, requirements=requirements)
    target_metadata = {
        name: metadata[name] for name in metadata if metadata[name] is not None
    }
    assert data.metadata == target_metadata  # type: ignore
