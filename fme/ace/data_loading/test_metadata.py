import datetime
from collections.abc import Mapping

import cftime
import numpy as np
import pytest
import xarray as xr

from fme.ace.data_loading.config import ConcatDatasetConfig, DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.requirements import DataRequirements
from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset.data_typing import VariableMetadata


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
    variable_metadata: Mapping[str, VariableMetadata | None],
    num_members=1,
    dim_sizes=None,
):
    if dim_sizes is None:
        dim_sizes = {"time": 3, "lat": 16, "lon": 32}
    data_vars = {}
    for name in variable_metadata:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)
        item_metadata = variable_metadata[name]
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
@pytest.mark.parametrize(
    "variable_metadata",
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
def test_metadata(tmp_path, variable_metadata, n_ensemble_members):
    paths = []
    for i in range(n_ensemble_members):
        path = tmp_path / f"ic{i}"
        path.mkdir(exist_ok=True)
        paths.append(path)
        _save_netcdf(path / "data.nc", variable_metadata)

    config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(
            concat=[XarrayDataConfig(data_path=str(path)) for path in paths]
        ),
        batch_size=1,
        num_data_workers=0,
    )
    var_names = list(variable_metadata.keys())
    requirements = DataRequirements(names=var_names, n_timesteps=2)
    data = get_gridded_data(config=config, train=True, requirements=requirements)
    target_metadata = {
        name: variable_metadata[name]
        for name in variable_metadata
        if variable_metadata[name] is not None
    }
    assert data.variable_metadata == target_metadata  # type: ignore
