# The dependencies of this script are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

import dataclasses
import logging
import os
import shutil
import tempfile
import time
from typing import Literal, Optional

import click
import dacite
import fsspec
import xarray as xr
import yaml
from fs_utils import is_local, makedirs, path_exists

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
    "UFS_REPLAY": ["time", "lat", "lon"],
}

ClimateDataType = Literal["FV3GFS", "E3SMV2", "ERA5", "CM4", "UFS_REPLAY"]

# (latitude, longitude) dimension names of each data type's lat-lon grid, for
# the variogram scaling stat
LAT_LON_DIMS = {
    "FV3GFS": ("grid_yt", "grid_xt"),
    "E3SMV2": ("lat", "lon"),
    "ERA5": ("latitude", "longitude"),
    "CM4": ("lat", "lon"),
    "UFS_REPLAY": ("lat", "lon"),
}

VARIOGRAM_SCALING_FILENAME = "variogram-scaling.nc"


def get_variogram_edge_offsets(window_size: int) -> list[tuple[int, int]]:
    """The (di, dj) edge kinds of a window_size x window_size window: one per
    unordered pair of distinct points, with di the latitude index step and dj
    the longitude index step.

    Must match fme.core.ensemble.get_variogram_edge_offsets, which the
    variogram score loss uses to select from the scaling file (by coordinate,
    so the order need not match).
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be odd and at least 3, got {window_size}")
    halo = window_size // 2
    return [(0, dj) for dj in range(1, halo + 1)] + [
        (di, dj) for di in range(1, halo + 1) for dj in range(-halo, halo + 1)
    ]


def compute_variogram_scaling(
    ds: xr.Dataset,
    lat_dim: str,
    lon_dim: str,
    window_size: int = 5,
) -> xr.Dataset:
    """Per-variable, per-edge-kind RMS spatial increment, the scale of the
    variogram score loss.

    For each edge kind (di, dj), the increment is x[i + di, j + dj] - x[i, j],
    with longitude periodic and no pairs across the poles. The scale is
    sqrt(mean(increment ** 2)) over time and all valid point pairs, with no
    mean subtraction and no area weighting. NaN increments (e.g. masked
    points) are excluded from the mean.

    Variables without both lat_dim and lon_dim are skipped. The result is
    lazy if ds is dask-backed.

    Args:
        ds: The dataset.
        lat_dim: Name of the latitude dimension.
        lon_dim: Name of the longitude dimension.
        window_size: Odd window width in grid points; the result has an
            ``edge`` dimension of every edge kind of this window.

    Returns:
        A dataset with an ``edge`` dimension and integer ``di``/``dj``
        coordinates on it, in the variables' physical units.
    """
    offsets = get_variogram_edge_offsets(window_size)
    n_lat = ds.sizes[lat_dim]
    result = {}
    for name, da in ds.data_vars.items():
        if lat_dim not in da.dims or lon_dim not in da.dims:
            continue
        # drop coordinates so shifted slices align by position
        x = da.variable.astype("float64")
        per_edge = []
        for di, dj in offsets:
            lower = x.isel({lat_dim: slice(0, n_lat - di)})
            upper = x.isel({lat_dim: slice(di, None)}).roll({lon_dim: -dj})
            per_edge.append(((upper - lower) ** 2).mean() ** 0.5)
        result[name] = xr.Variable.concat(per_edge, dim="edge")
        result[name].attrs = dict(da.attrs)
    return xr.Dataset(
        result,
        coords={
            "di": ("edge", [di for di, _ in offsets]),
            "dj": ("edge", [dj for _, dj in offsets]),
        },
        attrs={
            "description": (
                "RMS spatial increment x[i + di, j + dj] - x[i, j] per variable "
                "and edge kind (di: latitude index step, dj: longitude index "
                "step), over time and valid points, longitude periodic, no "
                "mean subtraction or area weighting. Scale of the variogram "
                "score loss."
            ),
            "window_size": window_size,
        },
    )


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
    """
    Attributes:
        variogram_scaling: If True, also write the variogram score scaling
            (``variogram-scaling.nc``, see compute_variogram_scaling), which
            costs a pass over 12 shifted copies of every variable.
    """

    output_directory: str
    data_type: ClimateDataType
    exclude_runs: list[str] = dataclasses.field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    beaker_dataset: str | None = None
    variogram_scaling: bool = False


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Configuration for time coarsening of a dataset.

    Attributes:
        data_output_directory: Directory to save the coarsened datasets as zarr stores.
        stats_output_directory: Directory to save the stats of the coarsened datasets.
        factor: Factor by which the time dimension is coarsened.
        beaker_dataset: Name of the Beaker dataset to create from the coarsened stats.
            If None, the coarsened stats are not uploaded to Beaker.
    """

    data_output_directory: str
    stats_output_directory: str
    factor: int
    output_names: dict[str, str] = dataclasses.field(default_factory=dict)
    beaker_dataset: str | None = None


@dataclasses.dataclass
class Config:
    runs: dict[str, str]
    data_output_directory: str
    stats: StatsConfig
    time_coarsen: TimeCoarsenConfig | None = None


def _out_dir_exists(out_dir: str) -> bool:
    """Check if the stats output directory already has results."""
    return path_exists(os.path.join(out_dir, "centering.nc"))


def get_stats(
    config: StatsConfig,
    input_zarr: str,
    out_dir: str,
    debug: bool,
):
    if not debug and _out_dir_exists(out_dir):
        logging.info(f"Stats already exist at {out_dir}. Skipping.")
        return

    # Import dask-related things here to enable testing in environments without dask.
    try:
        import dask
        import distributed

        client = distributed.Client(n_workers=16)
    except ImportError as e:
        # warn and continue
        logging.warning(f"Could not import dask ({e}), chunking is disabled.")
        client = None
        dask = None

    initial_time = time.time()

    xr.set_options(keep_attrs=True, display_max_rows=100)
    logging.info(f"Reading data from {input_zarr}")

    # Open data with roughly 128 MiB chunks via dask's automatic chunking. This
    # is useful when opening sharded zarr stores with an inner chunk size of 1,
    # which is otherwise inefficient for the type of computation done here.
    if dask is not None:
        with dask.config.set({"array.chunk-size": "128MiB"}):
            ds = xr.open_zarr(input_zarr, chunks={"time": "auto"})
    else:
        ds = xr.open_zarr(input_zarr)

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
    outputs = {
        "centering.nc": centering,
        "scaling-full-field.nc": scaling_full_field,
        "scaling-residual.nc": scaling_residual,
        "time-mean.nc": time_means,
    }
    if config.variogram_scaling:
        lat_dim, lon_dim = LAT_LON_DIMS[config.data_type]
        outputs[VARIOGRAM_SCALING_FILENAME] = compute_variogram_scaling(
            ds, lat_dim=lat_dim, lon_dim=lon_dim
        ).compute()
        logging.info("Computed variogram_scaling")

    for dataset in outputs.values():
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
        if is_local(out_dir):
            makedirs(out_dir)
            local_dir = out_dir
            remote_dir: Optional[str] = None
        else:
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir = out_dir

        for filename, dataset in outputs.items():
            dataset.to_netcdf(os.path.join(local_dir, filename))
            if remote_dir is not None:
                copy(
                    os.path.join(local_dir, filename),
                    remote_dir + "/" + filename,
                )

    total_time = time.time() - initial_time
    logging.info(f"Total time for computing stats: {total_time:0.2f} seconds.")

    if client is not None:
        client.close()
    client = None


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
    out_dir = config.stats.output_directory + "/" + run_name
    get_stats(
        config=config.stats,
        input_zarr=input_zarr,
        out_dir=out_dir,
        debug=debug,
    )
    if config.time_coarsen is not None:
        unknown_keys = set(config.time_coarsen.output_names) - set(config.runs)
        if unknown_keys:
            raise ValueError(
                f"time_coarsen.output_names keys not found in runs: {unknown_keys}"
            )
        if config.time_coarsen.data_output_directory.endswith("/"):
            config.time_coarsen.data_output_directory = (
                config.time_coarsen.data_output_directory[:-1]
            )
        if config.time_coarsen.stats_output_directory.endswith("/"):
            config.time_coarsen.stats_output_directory = (
                config.time_coarsen.stats_output_directory[:-1]
            )
        output_name = config.time_coarsen.output_names.get(run_name, run_name)
        time_coarsened_zarr = (
            config.time_coarsen.data_output_directory + "/" + output_name + ".zarr"
        )
        time_coarsened_out_dir = (
            config.time_coarsen.stats_output_directory + "/" + output_name
        )
        get_stats(
            config=config.stats,
            input_zarr=time_coarsened_zarr,
            out_dir=time_coarsened_out_dir,
            debug=debug,
        )


if __name__ == "__main__":
    main()
