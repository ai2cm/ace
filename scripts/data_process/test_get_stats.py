import json

import distributed
import numpy as np
import pandas as pd
import xarray as xr
from get_stats import StatsConfig, get_stats


def test_get_stats_writes_dict_attrs_as_json(tmp_path, monkeypatch):
    monkeypatch.setattr(distributed, "Client", lambda **_: None)
    root_attr = {"spread": 1.0, "residual": {"first": 0.1, "last": 0.2}}
    variable_attr = {"source": "test"}
    ds = xr.Dataset(
        {
            "thetao": (
                ["time", "lat", "lon"],
                np.random.rand(4, 2, 3),
                {"checks": variable_attr},
            )
        },
        coords={"time": pd.date_range("2000-01-01", periods=4)},
        attrs={"native_area_checks": root_attr},
    )
    input_zarr = str(tmp_path / "input.zarr")
    ds.to_zarr(input_zarr, zarr_format=3)
    out_dir = str(tmp_path / "stats")

    get_stats(
        StatsConfig(output_directory=out_dir, data_type="CM4"),
        input_zarr=input_zarr,
        out_dir=out_dir,
        debug=False,
    )

    for filename in [
        "centering.nc",
        "scaling-full-field.nc",
        "scaling-residual.nc",
        "time-mean.nc",
    ]:
        stats = xr.open_dataset(f"{out_dir}/{filename}")
        assert json.loads(stats.attrs["native_area_checks"]) == root_attr
        assert json.loads(stats["thetao"].attrs["checks"]) == variable_attr
