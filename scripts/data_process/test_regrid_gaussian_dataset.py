import os

import cftime
import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner
from regrid_gaussian_dataset import _conservative_weights, _make_gaussian_grid, main


def _cell_areas(grid: xr.Dataset) -> np.ndarray:
    return np.outer(
        np.diff(np.sin(np.radians(grid["lat_b"].values))),
        np.diff(np.radians(grid["lon_b"].values)),
    )


def _regrid(weights, array):
    lat_weights, lon_weights = weights
    return lat_weights @ array @ lon_weights.T


@pytest.mark.parametrize("source, target", [("F90", "F22.5"), ("F45", "F22.5")])
def test_conservative_weights(source, target):
    source_grid = _make_gaussian_grid(source)
    target_grid = _make_gaussian_grid(target)
    weights = _conservative_weights(source_grid, target_grid)
    field = np.random.default_rng(0).normal(size=_cell_areas(source_grid).shape)
    np.testing.assert_allclose(_regrid(weights, np.ones_like(field)), 1.0)
    source_mean = np.average(field, weights=_cell_areas(source_grid))
    target_mean = np.average(_regrid(weights, field), weights=_cell_areas(target_grid))
    np.testing.assert_allclose(target_mean, source_mean, rtol=0, atol=1e-12)


def test_main(tmp_path):
    grid = _make_gaussian_grid("F45")
    n_time = 5
    rng = np.random.default_rng(0)
    shape = (n_time, grid.sizes["lat"], grid.sizes["lon"])
    time = [cftime.DatetimeNoLeap(1, 1, 1 + i) for i in range(n_time)]
    ds = xr.Dataset(
        {
            "T": (("time", "lat", "lon"), rng.normal(size=shape).astype(np.float32)),
            "ICEFRAC": (
                ("time", "lat", "lon"),
                rng.uniform(-0.1, 1.1, size=shape).astype(np.float32),
            ),
            "PHIS": (("lat", "lon"), rng.normal(size=shape[1:]).astype(np.float32)),
            "ak_0": ((), 1.0),
        },
        coords={"time": time, "lat": grid["lat"].values, "lon": grid["lon"].values},
    )
    input_zarr = str(tmp_path / "input.zarr")
    output_zarr = str(tmp_path / "output.zarr")
    ds.to_zarr(input_zarr)
    args = [
        input_zarr,
        output_zarr,
        "--input-grid=F45",
        "--output-grid=F22.5",
        "--fraction-variables=ICEFRAC",
        "--time-block-size=2",
        "--n-workers=2",
    ]
    result = CliRunner().invoke(main, args + ["--max-blocks=1"])
    assert result.exit_code == 0, result.output
    assert os.path.isdir(output_zarr + ".progress")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert not os.path.exists(output_zarr + ".progress")

    output = xr.open_zarr(output_zarr)
    target_grid = _make_gaussian_grid("F22.5")
    weights = _conservative_weights(grid, target_grid)
    assert output.sizes == {"time": n_time, "lat": 45, "lon": 90}
    xr.testing.assert_equal(output["time"], ds["time"])
    np.testing.assert_allclose(
        output["T"].values, _regrid(weights, ds["T"].values), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        output["ICEFRAC"].values,
        np.clip(_regrid(weights, ds["ICEFRAC"].values), 0, 1),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        output["PHIS"].values, _regrid(weights, ds["PHIS"].values), rtol=1e-5, atol=1e-6
    )
    assert float(output["ak_0"]) == 1.0
    assert output["T"].encoding["shards"] == (2, 45, 90)
