import os

import click
import numpy as np
import pandas as pd
import xarray as xr


@click.command()
@click.argument("input_zarr")
@click.argument("output_directory")
@click.option("--start-date", help="For subsetting, e.g. '2016-01-01'")
@click.option("--end-date", help="For subsetting, e.g. '2016-12-31'")
@click.option("--nc-format", default="NETCDF4", help="netCDF file format")
@click.option(
    "--prepend-nans",
    is_flag=True,
    help="Prepend NaNs to first timestep. Used for baseline "
    "which is missing initial condition.",
)
def main(input_zarr, output_directory, start_date, end_date, nc_format, prepend_nans):
    """Save data at INPUT_ZARR to monthly netcdf files in OUTPUT_DIRECTORY.
    It is assumed that OUTPUT_DIRECTORY does not exist."""
    os.makedirs(output_directory, exist_ok=True)
    ds = xr.open_zarr(input_zarr)
    if prepend_nans:
        # prepend NaNs to first timestep
        ds = prepend_nans_to_dataset(ds)
    ds = ds.sel(time=slice(start_date, end_date))
    monthly_ds = ds.resample(time="MS")
    for label, data in monthly_ds:
        if isinstance(label, np.datetime64):
            # np.datetime64 times do not have a strftime method,
            # so convert to pd.Timestamp
            label = pd.Timestamp(label)
        print(f"Processing month {label}")
        filename = os.path.join(output_directory, label.strftime("%Y%m%d%H") + ".nc")
        # use these options to enable opening data with netCDF4.MFDataset
        data.to_netcdf(filename, unlimited_dims=["time"], format=nc_format)


def prepend_nans_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Prepend NaNs to time dimension of an xarray dataset."""
    time_dependent_vars = [v for v in ds if "time" in ds[v].dims]
    time_dependent_ds = ds[time_dependent_vars]
    prepend_step = xr.full_like(time_dependent_ds.isel(time=0), np.nan)
    delta_t = ds["time"].values[1] - ds["time"].values[0]
    prepend_step["time"] = ds["time"].values[0] - delta_t
    return xr.concat([prepend_step, ds], dim="time").transpose(*ds.dims)


def test_prepend_nans():
    ds = xr.tutorial.open_dataset("air_temperature")
    ds_prepended = prepend_nans_to_dataset(ds)
    assert ds_prepended.sizes["time"] == ds.sizes["time"] + 1
    assert np.isnan(ds_prepended.isel(time=0)["air"].values).all()


if __name__ == "__main__":
    main()
