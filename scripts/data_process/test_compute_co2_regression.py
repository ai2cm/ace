import datetime

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from compute_co2_regression import compute_co2_regression

CO2_NAME = "global_mean_co2"


def _get_synthetic_dataset(intercept, slope, co2_ref, spatial_co2: bool):
    """Monthly data whose annual means are exactly
    intercept + slope * ln(co2 / co2_ref), plus a seasonal cycle with zero
    annual mean and data outside the 1990-1999 window that does not follow the
    fit."""
    time = pd.date_range("1988-01-01", "2001-12-01", freq="MS")
    year = time.year.values
    month = time.month.values
    n_lat, n_lon = intercept.shape
    co2 = 350.0 * np.exp(0.01 * (year - 1990))
    log_ratio = np.log(co2 / co2_ref)
    seasonal = np.sin(2 * np.pi * (month - 1) / 12)
    field = (
        intercept[None]
        + slope[None] * log_ratio[:, None, None]
        + 5.0 * seasonal[:, None, None]
    )
    outside = (year < 1990) | (year > 1999)
    field[outside] = 1e6
    if spatial_co2:
        co2_var = (
            ["time", "lat", "lon"],
            np.broadcast_to(co2[:, None, None], (len(time), n_lat, n_lon)).copy(),
        )
    else:
        co2_var = (["time"], co2)
    return xr.Dataset(
        {"T": (["time", "lat", "lon"], field), CO2_NAME: co2_var},
        coords={"time": time},
    )


@pytest.mark.parametrize("spatial_co2", [False, True])
def test_compute_co2_regression_recovers_fit(spatial_co2: bool):
    rng = np.random.default_rng(0)
    intercept = 280 + rng.random((3, 4))
    slope = rng.random((3, 4))
    co2_ref = 360.0
    ds = _get_synthetic_dataset(intercept, slope, co2_ref, spatial_co2)
    out = compute_co2_regression(
        ds,
        names=["T"],
        co2_name=CO2_NAME,
        start_date="1990-01-01",
        end_date="1999-12-31",
        co2_ref=co2_ref,
    )
    np.testing.assert_allclose(out["T_intercept"].values, intercept, rtol=1e-5)
    np.testing.assert_allclose(out["T_slope"].values, slope, rtol=1e-3, atol=1e-3)
    assert out["T_intercept"].dims == ("lat", "lon")
    assert out.attrs["co2_name"] == CO2_NAME
    assert out.attrs["co2_ref"] == co2_ref
    assert out.attrs["n_years"] == 10


def test_compute_co2_regression_default_co2_ref_is_window_mean():
    rng = np.random.default_rng(0)
    intercept = 280 + rng.random((3, 4))
    slope = rng.random((3, 4))
    ds = _get_synthetic_dataset(intercept, slope, 360.0, spatial_co2=False)
    out = compute_co2_regression(
        ds,
        names=["T"],
        co2_name=CO2_NAME,
        start_date="1990-01-01",
        end_date="1999-12-31",
    )
    expected_co2_ref = float(np.mean(350.0 * np.exp(0.01 * np.arange(10))))
    assert out.attrs["co2_ref"] == pytest.approx(expected_co2_ref)
    # the intercept is the fitted field at co2_ref
    expected_intercept = intercept + slope * np.log(expected_co2_ref / 360.0)
    np.testing.assert_allclose(out["T_intercept"].values, expected_intercept, rtol=1e-5)


def test_compute_co2_regression_output_loads_into_corrector(tmp_path):
    from fme.core.corrector.co2_relaxation import CO2RegressionRelaxationConfig

    rng = np.random.default_rng(0)
    intercept = 280 + rng.random((3, 4))
    slope = rng.random((3, 4))
    ds = _get_synthetic_dataset(intercept, slope, 360.0, spatial_co2=False)
    out = compute_co2_regression(
        ds,
        names=["T"],
        co2_name=CO2_NAME,
        start_date="1990-01-01",
        end_date="1999-12-31",
        co2_ref=360.0,
    )
    path = tmp_path / "reference.nc"
    out.to_netcdf(path)
    config = CO2RegressionRelaxationConfig(
        timescale_days={"T": 1.0}, co2_name=CO2_NAME, path=str(path)
    )
    config.load()
    assert config.co2_ref == 360.0
    assert config.intercept is not None
    np.testing.assert_allclose(config.intercept["T"].numpy(), intercept, rtol=1e-5)
    relaxation = config.build(datetime.timedelta(hours=6))
    relaxation.train(False)
    x = torch.zeros(2, 3, 4, dtype=torch.float64)
    relaxed, _ = relaxation(
        {"T": x}, {"T": x}, {CO2_NAME: torch.full((2, 3, 4), 360.0)}, None
    )
    assert relaxed["T"].shape == x.shape


def test_compute_co2_regression_cli(tmp_path):
    from click.testing import CliRunner
    from compute_co2_regression import main

    rng = np.random.default_rng(0)
    intercept = 280 + rng.random((3, 4))
    slope = rng.random((3, 4))
    ds = _get_synthetic_dataset(intercept, slope, 360.0, spatial_co2=False)
    input_zarr = str(tmp_path / "data.zarr")
    ds.to_zarr(input_zarr)
    output_path = str(tmp_path / "reference.nc")
    result = CliRunner().invoke(
        main,
        [
            input_zarr,
            output_path,
            "--name",
            "T",
            "--co2-name",
            CO2_NAME,
            "--start-date",
            "1990-01-01",
            "--end-date",
            "1999-12-31",
            "--co2-ref",
            "360.0",
        ],
    )
    assert result.exit_code == 0, result.output
    out = xr.open_dataset(output_path)
    np.testing.assert_allclose(out["T_intercept"].values, intercept, rtol=1e-5)
    assert out.attrs["input_path"] == input_zarr
