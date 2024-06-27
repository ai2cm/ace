# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import os

import click
import xarray as xr
from dask.diagnostics import ProgressBar

# these are auxiliary variables that exist in dataset for convenience, e.g. to do
# masking or to more easily compute vertical integrals. But they are not inputs
# or outputs to the ML model, so we don't need normalization constants for them.
DROP_VARIABLES = (
    [
        "land_sea_mask",
        "pressure_thickness_of_atmospheric_layer_0",
        "pressure_thickness_of_atmospheric_layer_1",
        "pressure_thickness_of_atmospheric_layer_2",
        "pressure_thickness_of_atmospheric_layer_3",
        "pressure_thickness_of_atmospheric_layer_4",
        "pressure_thickness_of_atmospheric_layer_5",
        "pressure_thickness_of_atmospheric_layer_6",
        "pressure_thickness_of_atmospheric_layer_7",
    ]
    + [f"ak_{i}" for i in range(9)]
    + [f"bk_{i}" for i in range(9)]
)

DIMS = {
    "FV3GFS": ["time", "grid_xt", "grid_yt"],
    "E3SMV2": ["time", "lat", "lon"],
    "ERA5": ["time", "latitude", "longitude"],
}


def add_history_attr(ds, input_zarr, start_date, end_date):
    ds.attrs["history"] = (
        "Created by full-model/fv3gfs_data_process/get_stats.py. INPUT_ZARR:"
        f" {input_zarr}, START_DATE: {start_date}, END_DATE: {end_date}."
    )


@click.command()
@click.argument("input_zarr")
@click.argument("output_directory")
@click.option("--start-date", help="For subsetting, e.g. '2016-01-01'")
@click.option("--end-date", help="For subsetting, e.g. '2016-12-31'")
@click.option(
    "--data-type",
    default="FV3GFS",
    type=click.Choice(list(DIMS.keys())),
    help="Dataset type, used to determine some naming conventions.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="If set, print some statistics instead of writing normalization coefficients.",
)
def main(input_zarr, output_directory, start_date, end_date, data_type, debug):
    """Using data at INPUT_ZARR, compute statistics data and save to OUTPUT_DIRECTORY.
    It is assumed that OUTPUT_DIRECTORY does not exist."""
    xr.set_options(keep_attrs=True, display_max_rows=100)
    if not debug:
        os.makedirs(output_directory)
    ds = xr.open_zarr(input_zarr)
    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(start_date, end_date))

    dims = DIMS[data_type]

    centering = ds.mean(dim=dims)
    scaling_full_field = ds.std(dim=dims)
    scaling_residual = ds.diff("time").std(dim=dims)
    time_means = ds.mean(dim="time")

    for dataset in [
        centering,
        scaling_full_field,
        scaling_residual,
        time_means,
    ]:
        add_history_attr(dataset, input_zarr, start_date, end_date)

    if debug:
        normed_data = (ds - centering) / scaling_full_field
        print("Printing average of normed data:")
        print(normed_data.mean(dim=dims).compute())
        print("Printing standard deviation of normed data:")
        print(normed_data.std(dim=dims).compute())
        print("Printing standard deviation computed over all variables:")
        all_var_stddev = normed_data.to_array().std(dim=["variable"] + dims)
        print(all_var_stddev.values)
    else:
        with ProgressBar():
            centering.to_netcdf(os.path.join(output_directory, "centering.nc"))
        with ProgressBar():
            scaling_full_field.to_netcdf(
                os.path.join(output_directory, "scaling-full-field.nc")
            )
        with ProgressBar():
            scaling_residual.to_netcdf(
                os.path.join(output_directory, "scaling-residual.nc")
            )
        with ProgressBar():
            time_means.to_netcdf(os.path.join(output_directory, "time-mean.nc"))


if __name__ == "__main__":
    main()
