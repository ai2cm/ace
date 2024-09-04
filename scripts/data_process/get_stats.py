# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import os
import shutil
import tempfile
from typing import Dict, List, Literal, Optional

import click
import dacite
import fsspec
import xarray as xr
import yaml
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
    "CM4": ["time", "lat", "lon"],
}


def add_history_attrs(ds, input_zarr, start_date, end_date, n_samples):
    ds.attrs["history"] = (
        "Created by full-model/fv3gfs_data_process/get_stats.py. INPUT_ZARR:"
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
    data_type: Literal["FV3GFS", "E3SMV2", "ERA5", "CM4"]
    exclude_runs: List[str] = dataclasses.field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


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
    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)
    xr.set_options(keep_attrs=True, display_max_rows=100)
    if config.data_output_directory.endswith("/"):
        config.data_output_directory = config.data_output_directory[:-1]
    if list(config.runs.keys())[run] in config.stats.exclude_runs:
        print(f"Skipping run {list(config.runs.keys())[run]}")
        return
    input_zarr = (
        config.data_output_directory + "/" + list(config.runs.keys())[run] + ".zarr"
    )
    print(f"Reading data from {input_zarr}")
    ds = xr.open_zarr(input_zarr)
    ds = ds.drop_vars(DROP_VARIABLES, errors="ignore")
    ds = ds.sel(time=slice(config.stats.start_date, config.stats.end_date))

    dims = DIMS[config.stats.data_type]

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
        n_samples = len(ds.time)
        add_history_attrs(
            dataset,
            input_zarr,
            config.stats.start_date,
            config.stats.end_date,
            n_samples,
        )

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
        out_dir = config.stats.output_directory + "/" + list(config.runs.keys())[run]
        if out_dir.startswith("gs:"):
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir: Optional[str] = out_dir
        else:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            local_dir = out_dir
            remote_dir = None

        with ProgressBar():
            centering.to_netcdf(os.path.join(local_dir, "centering.nc"))
            if remote_dir is not None:
                copy(
                    os.path.join(local_dir, "centering.nc"),
                    remote_dir + "/centering.nc",
                )
        with ProgressBar():
            scaling_full_field.to_netcdf(
                os.path.join(local_dir, "scaling-full-field.nc")
            )
            if remote_dir is not None:
                copy(
                    os.path.join(local_dir, "scaling-full-field.nc"),
                    remote_dir + "/scaling-full-field.nc",
                )
        with ProgressBar():
            scaling_residual.to_netcdf(os.path.join(local_dir, "scaling-residual.nc"))
            if remote_dir is not None:
                copy(
                    os.path.join(local_dir, "scaling-residual.nc"),
                    remote_dir + "/scaling-residual.nc",
                )
        with ProgressBar():
            time_means.to_netcdf(os.path.join(local_dir, "time-mean.nc"))
            if remote_dir is not None:
                copy(
                    os.path.join(local_dir, "time-mean.nc"),
                    remote_dir + "/time-mean.nc",
                )


if __name__ == "__main__":
    main()
