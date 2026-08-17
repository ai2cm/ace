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


def store_path(directory: str, run_name: str) -> str:
    """Path of the zarr store holding a run's data within an output directory."""
    return directory.rstrip("/") + "/" + run_name + ".zarr"


def stats_path(directory: str, run_name: str) -> str:
    """Path of the directory holding a run's stats within an output directory."""
    return directory.rstrip("/") + "/" + run_name


@dataclasses.dataclass
class StatsConfig:
    output_directory: str
    data_type: ClimateDataType
    exclude_runs: list[str] = dataclasses.field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    beaker_dataset: str | None = None

    def stats_directory(self, run_name: str) -> str:
        """Directory holding a run's stats."""
        return stats_path(self.output_directory, run_name)

    def includes(self, run_name: str) -> bool:
        """Whether a run contributes to the stats."""
        return run_name not in self.exclude_runs


@dataclasses.dataclass
class TimeSliceConfig:
    """
    Optional time slice to apply before coarsening.

    Attributes:
        start: Start time (ISO 8601 string, e.g. "2000-01-01T00:00:00"). Inclusive.
        stop: Stop time (ISO 8601 string). Inclusive.
    """

    start: str | None = None
    stop: str | None = None


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Configuration for time coarsening of a dataset.

    Attributes:
        data_output_directory: Directory to save the coarsened datasets as zarr stores.
        stats_output_directory: Directory to save the stats of the coarsened datasets.
        factor: Factor by which the time dimension is coarsened.
        snapshot_names: List of snapshot variable names to coarsen. These will be
            coarsened by skipping each factor times.
        window_names: List of window variable names to coarsen. These will be
            coarsened by averaging over each factor times.
        constant_prefixes: List of prefixes for constant data variables to copy without
            modification. Raises an exception if any of these have a "time" dimension.
        output_name: Dataset name to give the coarsened data, used in place of the run
            name for both the zarr store and the stats directory. Required when the run
            names are dataset names in their own right, i.e. when
            data_output_directory holds datasets rather than the runs of one dataset.
            If None, each run's coarsened data is named after the run.
        beaker_dataset: Name of the Beaker dataset to create from the coarsened stats.
            If None, the coarsened stats are not uploaded to Beaker.
        n_split: Number of partitions to split the write into when using xpartition.
            Only used when dask and xpartition are available.
        chunking: Mapping of dimension names to inner chunk sizes for the output
            zarr store. Defaults to {"time": 1}. Spatial dimensions keep their
            existing chunking.
        sharding: Mapping of dimension names to shard sizes. If None, an unsharded
            zarr store is written with chunks as specified in ``chunking``.
        input_time_slice: Optional time slice to apply before coarsening.
    """

    data_output_directory: str
    stats_output_directory: str
    factor: int
    snapshot_names: list[str]
    window_names: list[str]
    constant_prefixes: list[str]
    output_name: str | None = None
    beaker_dataset: str | None = None
    n_split: int = 1
    chunking: dict[str, int] = dataclasses.field(default_factory=lambda: {"time": 1})
    sharding: dict[str, int] | None = dataclasses.field(
        default_factory=lambda: {"time": 360}
    )
    input_time_slice: TimeSliceConfig = dataclasses.field(
        default_factory=TimeSliceConfig
    )

    def __post_init__(self):
        if self.output_name is not None and (
            "/" in self.output_name or self.output_name.endswith(".zarr")
        ):
            raise ValueError(
                "time_coarsen.output_name must be a bare dataset name, but got "
                f"{self.output_name!r}. The output directory and the '.zarr' "
                "suffix are added to it automatically."
            )

    def coarsened_name(self, run_name: str) -> str:
        """Dataset name given to a run's coarsened data."""
        return run_name if self.output_name is None else self.output_name

    def store(self, run_name: str) -> str:
        """Zarr store holding a run's coarsened data."""
        return store_path(self.data_output_directory, self.coarsened_name(run_name))

    def stats_directory(self, run_name: str) -> str:
        """Directory holding a run's coarsened stats."""
        return stats_path(self.stats_output_directory, self.coarsened_name(run_name))

    def validate_stats_location(self, stats: StatsConfig, run_names: list[str]) -> None:
        """Check the coarsened stats would not be written over the input's stats.

        get_stats.py computes the stats of the input before those of the coarsened
        data, and skips a run whose stats already exist, so a shared directory
        means the coarsened stats are silently never computed and the input's
        stats stand in for them.

        Args:
            stats: Configuration of the stats of the data being coarsened.
            run_names: Names of the runs being coarsened.
        """
        collisions = [
            run
            for run in run_names
            if self.stats_directory(run) == stats.stats_directory(run)
        ]
        if collisions:
            run = collisions[0]
            raise ValueError(
                "The time coarsened stats would be written to the same directory as "
                f"the stats of the data they were coarsened from, for {run!r}: "
                f"{self.stats_directory(run)}. Because get_stats.py skips a "
                "run whose stats already exist, the coarsened stats would never be "
                "computed. Give time_coarsen.stats_output_directory a directory of its "
                "own, or set time_coarsen.output_name."
            )

    def validate_output_location(
        self, data_output_directory: str, run_names: list[str]
    ) -> None:
        """Check the coarsened data would not be written over or inside the input.

        Args:
            data_output_directory: Directory holding the data being coarsened.
            run_names: Names of the runs being coarsened.
        """
        source = data_output_directory.rstrip("/")
        destination = self.data_output_directory.rstrip("/")
        if destination == source:
            overwritten = [run for run in run_names if self.coarsened_name(run) == run]
            if overwritten:
                raise ValueError(
                    "Time coarsening would overwrite the data it reads: the "
                    f"coarsened store for {overwritten[0]!r} is its own input "
                    f"store. Set time_coarsen.output_name to a name that is not a "
                    "run name."
                )
        elif destination.startswith(source + "/") and self.output_name is None:
            example = self.store(run_names[0]) if run_names else destination + "/<run>"
            raise ValueError(
                f"time_coarsen.data_output_directory ({self.data_output_directory}) "
                f"is inside data_output_directory ({data_output_directory}), so the "
                "coarsened stores would be named after the runs they were coarsened "
                f"from, e.g. {example}. Set time_coarsen.output_name to the coarsened "
                "dataset's name and point time_coarsen.data_output_directory at "
                f"{data_output_directory}. See "
                "https://github.com/ai2cm/ace/issues/1399."
            )


@dataclasses.dataclass
class Config:
    runs: dict[str, str]
    data_output_directory: str
    stats: StatsConfig
    time_coarsen: TimeCoarsenConfig | None = None

    def __post_init__(self):
        if self.time_coarsen is not None:
            self.time_coarsen.validate_output_location(
                self.data_output_directory, self.run_names()
            )
            self.time_coarsen.validate_stats_location(self.stats, self.run_names())

    @property
    def has_time_coarsen(self) -> bool:
        """Whether the pipeline is configured to time coarsen the data."""
        return self.time_coarsen is not None

    def run_names(self) -> list[str]:
        """Names of the configured runs, in the order they are declared."""
        return list(self.runs)

    def included_run_names(self) -> list[str]:
        """Names of the runs that contribute to the stats."""
        return [run for run in self.run_names() if self.stats.includes(run)]

    def raw_store(self, run_name: str) -> str:
        """Zarr store holding the native-resolution data for a run."""
        return store_path(self.data_output_directory, run_name)

    def raw_stats_directory(self, run_name: str) -> str:
        """Directory holding the native-resolution stats for a run."""
        return self.stats.stats_directory(run_name)

    def coarsened_store(self, run_name: str) -> str:
        """Zarr store holding the time-coarsened data for a run."""
        if self.time_coarsen is None:
            raise ValueError("No time_coarsen section is configured.")
        return self.time_coarsen.store(run_name)

    def coarsened_stats_directory(self, run_name: str) -> str:
        """Directory holding the time-coarsened stats for a run."""
        if self.time_coarsen is None:
            raise ValueError("No time_coarsen section is configured.")
        return self.time_coarsen.stats_directory(run_name)


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
        if is_local(out_dir):
            makedirs(out_dir)
            local_dir = out_dir
            remote_dir: Optional[str] = None
        else:
            temp_dir = tempfile.TemporaryDirectory()
            local_dir = temp_dir.name
            remote_dir = out_dir

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
    run_name = config.run_names()[run]
    if not config.stats.includes(run_name):
        logging.info(f"Skipping run {run_name}")
        return
    get_stats(
        config=config.stats,
        input_zarr=config.raw_store(run_name),
        out_dir=config.raw_stats_directory(run_name),
        debug=debug,
    )
    if config.has_time_coarsen:
        get_stats(
            config=config.stats,
            input_zarr=config.coarsened_store(run_name),
            out_dir=config.coarsened_stats_directory(run_name),
            debug=debug,
        )


if __name__ == "__main__":
    main()
