import click
import os
import xarray as xr


@click.command()
@click.argument("input_zarr")
@click.argument("output_directory")
@click.option("--start-date", help="For subsetting, e.g. '2016-01-01'")
@click.option("--end-date", help="For subsetting, e.g. '2016-12-31'")
def main(input_zarr, output_directory, start_date, end_date):
    """Save data at INPUT_ZARR to monthly netcdf files in OUTPUT_DIRECTORY.
    It is assumed that OUTPUT_DIRECTORY does not exist."""
    os.makedirs(output_directory)
    ds = xr.open_zarr(input_zarr)
    ds = ds.sel(time=slice(start_date, end_date))
    monthly_ds = ds.resample(time="MS")
    for label, data in monthly_ds:
        print(f"Processing month {label}")
        filename = os.path.join(output_directory, label.strftime("%Y%m%d%H") + ".nc")
        # use these options to enable opening data with netCDF4.MFDataset
        data.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")


if __name__ == "__main__":
    main()
