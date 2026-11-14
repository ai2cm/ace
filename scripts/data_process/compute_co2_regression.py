"""Fit annual-mean fields against ln(CO2) at each grid point.

Writes the reference file read by the CO2 regression relaxation of the ACE
atmosphere corrector (``fme.core.corrector.co2_relaxation``): for each requested
field ``name``, the variables ``{name}_intercept`` and ``{name}_slope`` of the
ordinary-least-squares fit, over the annual means in the time window,

    field = intercept + slope * ln(co2 / co2_ref),

with ``co2_name`` and ``co2_ref`` stored as attributes. By default ``co2_ref`` is
the mean CO2 over the window, so the intercept is the climatology at that
concentration.

The window should span whole calendar years, since a partial year's mean is
biased by the seasonal cycle.
"""

import logging

import click
import numpy as np
import xarray as xr

TIME_DIM = "time"
YEAR_DIM = "year"


def annual_means(da: xr.DataArray) -> xr.DataArray:
    """Annual means of ``da``, computed one year at a time so that only one
    year of data is in memory at once, whether or not ``da`` is dask-backed.
    """
    years = np.unique(da[TIME_DIM].dt.year.values)
    means = []
    for year in years:
        selected = da.sel({TIME_DIM: da[TIME_DIM].dt.year == year})
        means.append(selected.mean(TIME_DIM).load())
    return xr.concat(means, dim=YEAR_DIM).assign_coords({YEAR_DIM: years})


def fit_against_log_co2(
    field: xr.DataArray, log_co2_ratio: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Per-gridpoint OLS fit of ``field`` against ``log_co2_ratio`` along the
    year dimension. Returns the (intercept, slope)."""
    x_anomaly = log_co2_ratio - log_co2_ratio.mean(YEAR_DIM)
    y_mean = field.mean(YEAR_DIM)
    slope = (x_anomaly * (field - y_mean)).sum(YEAR_DIM) / (x_anomaly**2).sum(YEAR_DIM)
    intercept = y_mean - slope * log_co2_ratio.mean(YEAR_DIM)
    return intercept, slope


def compute_co2_regression(
    ds: xr.Dataset,
    names: list[str],
    co2_name: str,
    start_date: str,
    end_date: str,
    co2_ref: float | None = None,
) -> xr.Dataset:
    """Fit each of ``names`` in ``ds`` against ln(co2 / co2_ref).

    Args:
        ds: Dataset with a time dimension, the fields and the CO2 variable.
        names: Names of the fields to fit.
        co2_name: Name of the CO2 variable, which may be a time series or vary
            in space.
        start_date: Start of the time window (inclusive).
        end_date: End of the time window (inclusive).
        co2_ref: Reference CO2 concentration. Defaults to the mean CO2 over the
            window.
    """
    # select variables and time before any arithmetic, so only the selected
    # data is ever read
    ds = ds[names + [co2_name]].sel({TIME_DIM: slice(start_date, end_date)})
    if ds.sizes[TIME_DIM] == 0:
        raise ValueError(f"No data between {start_date} and {end_date}.")
    co2 = annual_means(ds[co2_name])
    if co2.sizes[YEAR_DIM] < 2:
        raise ValueError("At least two years are needed to fit a slope.")
    if co2_ref is None:
        co2_ref = float(co2.mean())
    log_co2_ratio = np.log(co2 / co2_ref)
    out = xr.Dataset()
    for name in names:
        logging.info(f"Fitting {name}")
        intercept, slope = fit_against_log_co2(annual_means(ds[name]), log_co2_ratio)
        out[f"{name}_intercept"] = intercept.astype(np.float32)
        out[f"{name}_slope"] = slope.astype(np.float32)
    out.attrs["co2_name"] = co2_name
    out.attrs["co2_ref"] = co2_ref
    out.attrs["start_date"] = start_date
    out.attrs["end_date"] = end_date
    out.attrs["n_years"] = int(co2.sizes[YEAR_DIM])
    return out


@click.command()
@click.argument("input_zarr")
@click.argument("output_path")
@click.option(
    "--name", "names", multiple=True, required=True, help="Field to fit (repeatable)."
)
@click.option("--co2-name", required=True, help="Name of the CO2 variable.")
@click.option("--start-date", default="1990-01-01", show_default=True)
@click.option("--end-date", default="2019-12-31", show_default=True)
@click.option(
    "--co2-ref",
    type=float,
    default=None,
    help="Reference CO2 concentration. Defaults to the mean CO2 over the window.",
)
def main(input_zarr, output_path, names, co2_name, start_date, end_date, co2_ref):
    """Fit annual means of fields in INPUT_ZARR against ln(CO2) and write the
    intercepts and slopes to the netCDF file OUTPUT_PATH."""
    logging.basicConfig(level=logging.INFO)
    ds = xr.open_zarr(input_zarr)
    out = compute_co2_regression(
        ds,
        names=list(names),
        co2_name=co2_name,
        start_date=start_date,
        end_date=end_date,
        co2_ref=co2_ref,
    )
    out.attrs["input_path"] = input_zarr
    out.attrs["history"] = (
        "Created by scripts/data_process/compute_co2_regression.py from "
        f"{input_zarr}, {start_date} to {end_date}."
    )
    out.to_netcdf(output_path)


if __name__ == "__main__":
    main()
