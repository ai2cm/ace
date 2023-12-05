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

# these variables either have no time dimension, or we expect them to vary
# only slightly in time (e.g. ocean_fraction and sea_ice_fraction). So we
# specify them here to indicate that their standard deviations should always be
# computed using the "full-field" approach.
TIME_INVARIANT_VARIABLES = {
    "FV3GFS": ("HGTsfc", "land_fraction", "ocean_fraction", "sea_ice_fraction"),
    "E3SMV2": ("PHIS", "OCNFRAC", "ICEFRAC", "LANDFRAC"),
}

DIMS = {
    "FV3GFS": ["time", "grid_xt", "grid_yt"],
    "E3SMV2": ["time", "lat", "lon"],
}


def add_history_attr(ds, input_zarr, start_date, end_date, full_field):
    scaling_str = "full-field" if full_field else "residual"
    ds.attrs["history"] = (
        "Created by full-model/fv3gfs_data_process/get_stats.py. INPUT_ZARR:"
        f" {input_zarr}, START_DATE: {start_date}, END_DATE: {end_date}, using "
        f" {scaling_str} scaling."
    )


def compute_residual_scaling(
    ds, centering, scaling, dims=["time", "grid_xt", "grid_yt"]
):
    """For some variables (e.g. surface pressure), normalization based on the
    'full field' leads to residuals that are very small and therefore don't
    contribute much to loss function. Here we rescale based on the residual
    standard deviation. See
    https://github.com/ai2cm/explore/blob/master/oliwm/2023-04-16-fme-analysis/2023-04-27-compute-new-scaling-dataset.ipynb  # noqa: E501
    for more details.
    """
    norm = (ds - centering) / scaling
    norm_residual = norm.diff("time")
    norm_residual_stddev = norm_residual.std(dim=dims)
    # We want to multiply the global_stds by the above norm_residual_stddev variable
    # so that the residuals are more evenly weighted. However, we don't want to
    # make the standard deviations of the normalized inputs/outputs very different from
    # 1. Therefore we rescale the norm_residual_stddev variable so that its geometric
    # mean is 1. Choice of using geometric mean is somewhat arbitrary.
    as_variable_array = norm_residual_stddev.to_array()
    n_variables = as_variable_array.sizes["variable"]
    geometric_mean = as_variable_array.prod(dim="variable") ** (1 / n_variables)
    norm_residual_stddev /= geometric_mean
    # rescale standard deviations so that residuals are evenly weighted in loss
    return scaling * norm_residual_stddev


@click.command()
@click.argument("input_zarr")
@click.argument("output_directory")
@click.option("--start-date", help="For subsetting, e.g. '2016-01-01'")
@click.option("--end-date", help="For subsetting, e.g. '2016-12-31'")
@click.option(
    "--data-type", default="FV3GFS", help="Input data type, e.g., 'FV3GFS' or 'E3SMV2'"
)
@click.option(
    "--full-field",
    is_flag=True,
    help="If set, compute statistics with full field. If unset, use residual.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="If set, print some statistics instead of writing normalization coefficients.",
)
def main(
    input_zarr, output_directory, start_date, end_date, data_type, full_field, debug
):
    """Using data at INPUT_ZARR, compute statistics data and save to OUTPUT_DIRECTORY.
    It is assumed that OUTPUT_DIRECTORY does not exist."""
    xr.set_options(keep_attrs=True, display_max_rows=100)
    if not debug:
        os.makedirs(output_directory)
    ds = xr.open_zarr(input_zarr)
    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(start_date, end_date))

    dims = DIMS[data_type]
    time_invariant_variables = TIME_INVARIANT_VARIABLES[data_type]

    centering = ds.mean(dim=dims)
    scaling = ds.std(dim=dims)
    time_means = ds.mean(dim="time")

    if not full_field:
        ds_time_varying = ds.drop_vars(time_invariant_variables)
        centering_time_varying = centering.drop_vars(time_invariant_variables)
        scaling_time_varying = scaling.drop_vars(time_invariant_variables)
        residual_scaling = compute_residual_scaling(
            ds_time_varying, centering_time_varying, scaling_time_varying, dims
        )
        for name in residual_scaling:
            scaling[name] = residual_scaling[name]

    for dataset in [centering, scaling, time_means]:
        add_history_attr(dataset, input_zarr, start_date, end_date, full_field)

    if debug:
        normed_data = (ds - centering) / scaling
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
            scaling.to_netcdf(os.path.join(output_directory, "scaling.nc"))
        with ProgressBar():
            time_means.to_netcdf(os.path.join(output_directory, "time-mean.nc"))


if __name__ == "__main__":
    main()
