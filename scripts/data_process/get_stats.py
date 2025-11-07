# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from typing import Dict, List, Literal, Optional

import click
import dacite
import fsspec
import xarray as xr
import yaml

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
        "mask_HI",
        "mask_sea_ice_volume",
        "mask_sea_ice_fraction",
        "mask_ocean_sea_ice_fraction",
    ]
    + [f"ak_{i}" for i in range(9)]
    + [f"bk_{i}" for i in range(9)]
    + [f"idepth_{i}" for i in range(19)]
    + [f"mask_{i}" for i in range(19)]
)

DIMS = {
    "FV3GFS": ["time", "grid_xt", "grid_yt"],
    "E3SMV2": ["time", "lat", "lon"],
    "ERA5": ["time", "latitude", "longitude"],
    "CM4": ["time", "lat", "lon"],
}

ClimateDataType = Literal["FV3GFS", "E3SMV2", "ERA5", "CM4"]


def add_history_attrs(ds, input_zarr, start_date, end_date, n_samples):
    ds.attrs["history"] = (
        "Created by full-model/scripts/data_process/get_stats.py. INPUT_ZARR:"
        f" {input_zarr}, START_DATE: {start_date}, END_DATE: {end_date}."
    )
    ds.attrs["input_samples"] = n_samples


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


@dataclasses.dataclass
class StatsConfig:
    output_directory: str
    data_type: ClimateDataType
    exclude_runs: List[str] = dataclasses.field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    beaker_dataset: Optional[str] = None


def get_stats(
    config: StatsConfig,
    input_zarr: str,
    run_name: str,
    debug: bool,
):
    # Import dask-related things here to enable testing in environments without dask.
    import dask
    import distributed

    initial_time = time.time()
    client = distributed.Client(n_workers=16)

    xr.set_options(keep_attrs=True, display_max_rows=100)
    logging.info(f"Reading data from {input_zarr}")

    # Open data with roughly 128 MiB chunks via dask's automatic chunking. This
    # is useful when opening sharded zarr stores with an inner chunk size of 1,
    # which is otherwise inefficient for the type of computation done here.
    with dask.config.set({"array.chunk-size": "128MiB"}):
        ds = xr.open_zarr(input_zarr, chunks={"time": "auto"})

    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(config.start_date, config.end_date))

    dims = DIMS[config.data_type]

    # Explicitly compute the statistics here, since xarray does not support
    # writing netCDFs with the scipy engine with the distributed scheduler.
    # There is no harm to computing here versus later, since the end result is
    # not something memory intensive.
    centering = ds.mean(dim=dims).compute()
    logging.info("Computed centering")
    scaling_full_field = ds.std(dim=dims).compute()
    logging.info("Computed scaling_full_field")
    scaling_residual = ds.diff("time").std(dim=dims).compute()
    logging.info("Computed scaling_residual")
    time_means = ds.mean(dim="time").compute()
    logging.info("Computed time_means")

    for dataset in [
        centering,
        scaling_full_field,
        scaling_residual,
        time_means,
    ]:
        n_samples = len(ds.time)
        add_history_attrs(
            dataset,
            input_zarr,
            config.start_date,
            config.end_date,
            n_samples,
        )

    if debug:
        normed_data = (ds - centering) / scaling_full_field
        logging.info(f"Average of normed data: {normed_data.mean(dim=dims).compute()}")
        logging.info(
            f"Standard deviation of normed data: {normed_data.std(dim=dims).compute()}"
        )
        all_var_stddev = normed_data.to_array().std(dim=["variable"] + dims)
        logging.info(
            f"Standard deviation computed over all variables: {all_var_stddev.values}"
        )
    else:
        out_dir = config.output_directory + "/" + run_name
        if out_dir.startswith("gs:"):
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir: Optional[str] = out_dir
        else:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            local_dir = out_dir
            remote_dir = None

        centering.to_netcdf(os.path.join(local_dir, "centering.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "centering.nc"),
                remote_dir + "/centering.nc",
            )
        scaling_full_field.to_netcdf(os.path.join(local_dir, "scaling-full-field.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-full-field.nc"),
                remote_dir + "/scaling-full-field.nc",
            )
        scaling_residual.to_netcdf(os.path.join(local_dir, "scaling-residual.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "scaling-residual.nc"),
                remote_dir + "/scaling-residual.nc",
            )
        time_means.to_netcdf(os.path.join(local_dir, "time-mean.nc"))
        if remote_dir is not None:
            copy(
                os.path.join(local_dir, "time-mean.nc"),
                remote_dir + "/time-mean.nc",
            )

    total_time = time.time() - initial_time
    logging.info(f"Total time for computing stats: {total_time:0.2f} seconds.")

    client.close()
    client = None


@dataclasses.dataclass
class Config:
    runs: Dict[str, str]
    data_output_directory: str
    stats: StatsConfig


@click.command()
@click.argument("config_yaml", type=str)
@click.argument("run", type=int)
@click.option(
    "--debug",
    is_flag=True,
    help="If set, print some statistics instead of writing normalization coefficients.",
)
def main(config_yaml: str, run: int, debug: bool):
    """
    Compute statistics for the data processing pipeline.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    run -- Run index for the data processing pipeline.
    """

    logging.basicConfig(level=logging.INFO)

    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)
    run_name = list(config.runs.keys())[run]
    if run_name in config.stats.exclude_runs:
        logging.info(f"Skipping run {run_name}")
        return
    if config.data_output_directory.endswith("/"):
        config.data_output_directory = config.data_output_directory[:-1]
    input_zarr = config.data_output_directory + "/" + run_name + ".zarr"
    get_stats(
        config=config.stats,
        input_zarr=input_zarr,
        run_name=run_name,
        debug=debug,
    )


if __name__ == "__main__":
    main()
