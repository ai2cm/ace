import dataclasses
import logging
import os
from typing import Literal

import click
import dacite
import xarray as xr
import yaml
from coupled_dataset_utils import (
    CoupledFieldNamesConfig,
    CoupledSeaIceConfig,
    CoupledSeaSurfaceConfig,
    CoupledSurfaceTemperatureConfig,
    ExtraFieldsConfig,
    compute_coupled_atmosphere,
    compute_coupled_ocean,
    compute_coupled_sea_ice,
)
from get_stats import ClimateDataType, StatsConfig, get_stats
from merge_stats import MergeStatsConfig, merge_stats
from writer_utils import OutputWriterConfig


def _get_stats(
    input_zarr_path: str,
    climate_data_type: ClimateDataType,
    debug: bool,
    subsample: bool,
) -> str:
    """Thin wrapper around get_stats.

    Note that distributed stats computation is handled directly by the
    imported get_stats util.

    Args:
        climate_data_type: Type of climate data for statistics computation.
        debug: If True, do nothing because the input_zarr_path doesn't exist.
        subsample: If True, update path to include subsample suffix.

    """
    # Update path if subsampled
    if subsample:
        input_zarr_path = input_zarr_path.replace(".zarr", "-subsample.zarr")

    # Check if stats directory already exists
    run_name = os.path.basename(input_zarr_path).replace(".zarr", "-stats")
    stats_dir = os.path.join(os.path.dirname(input_zarr_path), run_name)
    if os.path.exists(stats_dir):
        logging.info(
            f"Stats directory {stats_dir} already exists. Skipping stats computation."
        )
        return stats_dir
    if debug:
        # do nothing
        logging.info(
            f"Skipping stats computation for {input_zarr_path} because debug=True."
        )
    else:
        logging.info(f"Computing statistics")
        logging.info(f"  Output directory: {stats_dir}")
        get_stats(
            config=StatsConfig(
                output_directory=os.path.dirname(input_zarr_path),
                data_type=climate_data_type,
            ),
            input_zarr=input_zarr_path,
            run_name=run_name,
            debug=False,
        )
    return stats_dir


def _merge_stats(
    sea_ice_dir: str,
    ocean_dir: str,
    atmos_dir: str,
    uncoupled_ocean_dir: str,
    uncoupled_atmos_dir: str,
    output_directory: str,
    debug: bool,
    subsample: bool,
):
    """Merge statistics from coupled and uncoupled datasets for different
    training scenarios.

    Creates three merged statistics directories:
    1. uncoupled_atmosphere: coupled sea ice + uncoupled atmos
    2. coupled_atmosphere: coupled atmos + uncoupled atmos
    3. ocean: coupled ocean + uncoupled ocean

    Args:
        sea_ice_dir: Directory with coupled sea ice statistics.
        ocean_dir: Directory with coupled ocean statistics.
        atmos_dir: Directory with coupled atmosphere statistics.
        uncoupled_ocean_dir: Directory with uncoupled ocean statistics.
        uncoupled_atmos_dir: Directory with uncoupled atmosphere statistics.
        output_directory: Base directory where merged stats will be written.
        debug: If True, skip merging (datasets don't exist in debug mode).
        subsample: If True, include '-subsample' in output directory name.

    """
    if debug:
        logging.info("Skipping stats merging in debug mode")
        return

    if subsample:
        output_directory = output_directory.replace("-stats", "-subsample-stats")

    # uncoupled atmosphere training (coupled_sea_ice + uncoupled_atmos)
    output_dir = os.path.join(output_directory, f"uncoupled_atmosphere")
    if not os.path.exists(output_dir):
        logging.info("Merging stats for uncoupled atmosphere training")
        logging.info(f"  Input: {sea_ice_dir}")
        logging.info(f"  Input: {uncoupled_atmos_dir}")
        logging.info(f"  Output: {output_dir}")
        merge_stats(
            config=MergeStatsConfig(
                input_directories=[sea_ice_dir, uncoupled_atmos_dir],
                output_directory=output_dir,
            )
        )
    else:
        logging.info(f"Merged atmosphere stats already exist at {output_dir}, skipping")

    # uncoupled/coupled ocean training (coupled_ocean + uncoupled_ocean)
    output_dir = os.path.join(output_directory, "ocean")
    if not os.path.exists(output_dir):
        logging.info("Merging stats for uncoupled / coupled ocean training")
        logging.info(f"  Input: {ocean_dir}")
        logging.info(f"  Input: {uncoupled_ocean_dir}")
        logging.info(f"  Output: {output_dir}")
        merge_stats(
            config=MergeStatsConfig(
                input_directories=[ocean_dir, uncoupled_ocean_dir],
                output_directory=output_dir,
            )
        )
    else:
        logging.info(f"Merged ocean stats already exist at {output_dir}, skipping")

    # coupled atmosphere training (coupled_atmos + uncoupled_atmos)
    output_dir = os.path.join(output_directory, f"coupled_atmosphere")
    if not os.path.exists(output_dir):
        logging.info("Merging stats for coupled atmosphere training")
        logging.info(f"  Input: {atmos_dir}")
        logging.info(f"  Input: {uncoupled_atmos_dir}")
        logging.info(f"  Output: {output_dir}")
        merge_stats(
            config=MergeStatsConfig(
                input_directories=[atmos_dir, uncoupled_atmos_dir],
                output_directory=output_dir,
            )
        )
    else:
        logging.info(
            f"Coupled atmosphere stats already exist at {output_dir}, skipping"
        )


@dataclasses.dataclass
class CoupledInputDatasetConfig:
    """Configuration for a single input dataset.

    Parameters:
        zarr_path: Path to the zarr dataset.
        timedelta: Time resolution of the dataset (e.g., "6h", "5D").
        time_chunk_size: Chunk size for the time dimension.
        extra_fields: Optional list of variable names/prefixes
            to copy to the outputs.
    """

    zarr_path: str
    time_chunk_size: int
    extra_fields: ExtraFieldsConfig = dataclasses.field(
        default_factory=ExtraFieldsConfig
    )
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    def get_dataset(self) -> xr.Dataset:
        """Load the zarr dataset with specified time chunking and range.

        Returns:
            Dataset loaded from zarr with time selection applied.
        """
        logging.info(f"Loading dataset from {self.zarr_path}")
        ds = xr.open_zarr(self.zarr_path, chunks={"time": self.time_chunk_size})
        ds = ds.sel(time=slice(self.first_timestamp, self.last_timestamp))
        logging.info(f"  Loaded dataset with {len(ds.time)} timesteps")
        return ds

    def log_info(self):
        """Log information about the dataset configuration."""
        logging.info(f"  zarr_path: {self.zarr_path}")
        logging.info(f"  time_chunk_size: {self.time_chunk_size}")
        if self.first_timestamp or self.last_timestamp:
            logging.info(
                f"  time_range: {self.first_timestamp or 'start'} to "
                f"{self.last_timestamp or 'end'}"
            )
        if self.extra_fields.names_and_prefixes:
            logging.info("  extra_fields: " f"{self.extra_fields.names_and_prefixes}")


StatsType = Literal["centering", "scaling-full-field", "scaling-residual", "time-mean"]


@dataclasses.dataclass
class InputStatsConfig:
    """Configuration of input stats datasets.

    Parameters:
        atmosphere_dir: Directory name of atmosphere statistics.
        ocean_dir: Directory name of ocean statistics.
    """

    atmosphere_dir: str
    ocean_dir: str


@dataclasses.dataclass
class InputDatasetsConfig:
    """Configuration for all input datasets.

    Parameters:
        climate_data_type: Identifier for get_stats.
        atmosphere: Configuration for atmosphere input dataset.
        ocean: Configuration for ocean input dataset.
        sea_ice: Configuration for sea ice input dataset.
        stats: Configuration for stats datasets.
    """

    climate_data_type: ClimateDataType
    atmosphere: CoupledInputDatasetConfig
    ocean: CoupledInputDatasetConfig
    stats: InputStatsConfig
    sea_ice: CoupledInputDatasetConfig | None = None

    def log_info(self):
        """Log information about all input datasets."""
        logging.info(f"Climate data type: {self.climate_data_type}")
        logging.info("Atmosphere input dataset:")
        self.atmosphere.log_info()
        logging.info("Ocean input dataset:")
        self.ocean.log_info()
        if self.sea_ice is not None:
            logging.info("Sea ice input dataset:")
            self.sea_ice.log_info()
        else:
            logging.info("Sea ice input dataset: None")


@dataclasses.dataclass
class CoupledDatasetsConfig:
    """Configuration for coupled dataset processing.

    Parameters:
        version: Version string for the output datasets.
        family_name: Common name for the family of coupled output datasets.
        output_directory: Directory where outputs will be written.
        coupled_ts: Configuration for surface temperature in coupled atmosphere.
        coupled_sea_surface: Configuration for sea ice in coupled ocean.
        coupled_sea_ice: Configuration for for sea ice in the uncoupled atmosphere.
        input_field_names: Names of input fields.
        output_writer: Configuration for writing to output store.
    """

    version: str
    family_name: str
    output_directory: str
    coupled_ts: CoupledSurfaceTemperatureConfig
    coupled_sea_surface: CoupledSeaSurfaceConfig
    coupled_sea_ice: CoupledSeaIceConfig = dataclasses.field(
        default_factory=CoupledSeaIceConfig
    )
    input_field_names: CoupledFieldNamesConfig = dataclasses.field(
        default_factory=CoupledFieldNamesConfig
    )
    output_writer: OutputWriterConfig = dataclasses.field(
        default_factory=OutputWriterConfig
    )

    @property
    def sea_ice_output_store(self) -> str:
        """Get the output path for the coupled sea ice dataset.

        Returns:
            Absolute path to the coupled sea ice output zarr store.

        """
        return os.path.join(
            self.output_directory,
            f"{self.version}-{self.family_name}-sea_ice.zarr",
        )

    @property
    def ocean_output_store(self) -> str:
        """Get the output path for the coupled ocean dataset.

        Returns:
            Absolute path to the coupled ocean output zarr store.
        """
        return os.path.join(
            self.output_directory,
            f"{self.version}-{self.family_name}-ocean.zarr",
        )

    @property
    def atmosphere_output_store(self) -> str:
        """Get the output path for the coupled atmosphere dataset.

        The filename includes the configured CoupledSurfaceTemperatureConfig.how
        string.

        Returns:
            Absolute path to the coupled atmosphere output zarr store.

        """
        prefix = f"{self.version}-{self.family_name}-{self.coupled_ts.how}"
        return os.path.join(
            self.output_directory,
            f"{prefix}-atmosphere.zarr",
        )

    @property
    def coupled_stats_directory(self) -> str:
        """Get the output directory for the coupled stats.

        Returns:
            Absolute path to the coupled stats output directory.

        """
        return os.path.join(
            self.output_directory,
            f"{self.version}-{self.family_name}-stats",
        )

    def write_datasets_and_stats(
        self,
        input_datasets: InputDatasetsConfig,
        debug: bool,
        subsample: bool,
    ):
        logging.info("=" * 80)
        logging.info("Creating coupled atmosphere-ocean datasets")
        logging.info(f"  Atmosphere output: {self.atmosphere_output_store}")
        logging.info(f"  Ocean output: {self.ocean_output_store}")
        logging.info(f"  Sea ice output: {self.sea_ice_output_store}")
        logging.info(f"  Stats output: {self.coupled_stats_directory}")
        logging.info("=" * 80)

        # Check if outputs already exist (unless in debug mode)
        sea_ice_exists = not debug and os.path.exists(self.sea_ice_output_store)
        ocean_exists = not debug and os.path.exists(self.ocean_output_store)
        atmos_exists = not debug and os.path.exists(self.atmosphere_output_store)

        if sea_ice_exists and atmos_exists and ocean_exists:
            logging.info(f"All coupled output datasets already exist. Skipping...")
            # Still compute stats if they don't exist
            sea_ice_stats_dir = _get_stats(
                input_zarr_path=self.sea_ice_output_store,
                climate_data_type=input_datasets.climate_data_type,
                debug=debug,
                subsample=subsample,
            )
            atmos_stats_dir = _get_stats(
                input_zarr_path=self.atmosphere_output_store,
                climate_data_type=input_datasets.climate_data_type,
                debug=debug,
                subsample=subsample,
            )
            ocean_stats_dir = _get_stats(
                input_zarr_path=self.ocean_output_store,
                climate_data_type=input_datasets.climate_data_type,
                debug=debug,
                subsample=subsample,
            )
            _merge_stats(
                sea_ice_dir=sea_ice_stats_dir,
                ocean_dir=ocean_stats_dir,
                atmos_dir=atmos_stats_dir,
                uncoupled_ocean_dir=input_datasets.stats.ocean_dir,
                uncoupled_atmos_dir=input_datasets.stats.atmosphere_dir,
                output_directory=self.coupled_stats_directory,
                debug=debug,
                subsample=subsample,
            )
            return

        if sea_ice_exists:
            logging.warning(
                f"Coupled sea ice output {self.sea_ice_output_store} already exists. "
                "It will be recomputed but not overwritten."
            )
        if atmos_exists:
            logging.warning(
                f"Coupled atmosphere output {self.atmosphere_output_store} already "
                "exists. It will be recomputed but not overwritten."
            )
        if ocean_exists:
            logging.warning(
                f"Coupled ocean output {self.ocean_output_store} already exists. "
                "It will be recomputed but not overwritten."
            )

        self.output_writer.start_dask_client(debug)

        atmos = input_datasets.atmosphere.get_dataset()
        ocean = input_datasets.ocean.get_dataset()

        sea_ice: xr.Dataset | None = None
        sea_ice_extras: ExtraFieldsConfig | None = None
        if input_datasets.sea_ice is not None:
            sea_ice = input_datasets.sea_ice.get_dataset()
            sea_ice_extras = input_datasets.sea_ice.extra_fields
        else:
            logging.info(
                "Sea ice input dataset not configured. Using atmosphere input dataset "
                "as source for sea ice fields."
            )

        logging.info("=" * 80)
        logging.info("Computing coupled sea ice fields")
        logging.info("=" * 80)
        atmos_extras = input_datasets.atmosphere.extra_fields
        coupled_sea_ice = compute_coupled_sea_ice(
            ocean=ocean,
            atmos=atmos,
            sea_ice=sea_ice,
            config=self.coupled_sea_ice,
            input_field_names=self.input_field_names,
            atmos_extras=atmos_extras,
            sea_ice_extras=sea_ice_extras,
        )

        logging.info("=" * 80)
        logging.info("Computing coupled ocean fields")
        logging.info("=" * 80)
        coupled_ocean = compute_coupled_ocean(
            ocean=ocean,
            atmos=atmos,
            # drop extra atmos fields from sea ice dataset
            coupled_sea_ice=atmos_extras.drop_extra_data_vars(coupled_sea_ice),
            config=self.coupled_sea_surface,
            input_field_names=self.input_field_names,
            extras=input_datasets.ocean.extra_fields,
        )

        logging.info("=" * 80)
        logging.info("Computing coupled atmosphere fields")
        logging.info("=" * 80)
        coupled_atmos = compute_coupled_atmosphere(
            atmos=atmos,
            ocean=ocean,
            coupled_ocean=coupled_ocean,
            config=self.coupled_ts,
            input_field_names=self.input_field_names,
            extras=atmos_extras,
        )

        atmos_output_store = self.atmosphere_output_store
        ocean_output_store = self.ocean_output_store
        sea_ice_output_store = self.sea_ice_output_store

        if subsample:
            tdim = self.input_field_names.time_dim
            logging.info(f"Subsampling coupled ocean to 73 timesteps")
            coupled_ocean = coupled_ocean.isel({tdim: slice(None, 73)})
            logging.info(f"Subsampling coupled atmosphere to 1460 timesteps")
            coupled_atmos = coupled_atmos.isel({tdim: slice(None, 365 * 4)})
            logging.info(f"Subsampling coupled sea ice to 1460 timesteps")
            coupled_sea_ice = coupled_sea_ice.isel({tdim: slice(None, 365 * 4)})
            ocean_output_store = ocean_output_store.replace(".zarr", "-subsample.zarr")
            atmos_output_store = atmos_output_store.replace(".zarr", "-subsample.zarr")
            sea_ice_output_store = sea_ice_output_store.replace(
                ".zarr", "-subsample.zarr"
            )

        if debug:
            with xr.set_options(display_max_rows=500):
                logging.info("Debug mode: printing coupled ocean dataset")
                print(coupled_ocean)
                logging.info("Debug mode: printing coupled atmosphere dataset")
                print(coupled_atmos)
                logging.info("Debug mode: printing coupled sea ice dataset")
                print(coupled_sea_ice)
        else:
            if not atmos_exists:
                self.output_writer.write(coupled_atmos, atmos_output_store)
            if not ocean_exists:
                self.output_writer.write(coupled_ocean, ocean_output_store)
            if not sea_ice_exists:
                self.output_writer.write(coupled_sea_ice, sea_ice_output_store)

        self.output_writer.close_dask_client()

        atmos_stats_dir = _get_stats(
            input_zarr_path=self.atmosphere_output_store,
            climate_data_type=input_datasets.climate_data_type,
            debug=debug,
            subsample=subsample,
        )
        ocean_stats_dir = _get_stats(
            input_zarr_path=self.ocean_output_store,
            climate_data_type=input_datasets.climate_data_type,
            debug=debug,
            subsample=subsample,
        )
        sea_ice_stats_dir = _get_stats(
            input_zarr_path=self.sea_ice_output_store,
            climate_data_type=input_datasets.climate_data_type,
            debug=debug,
            subsample=subsample,
        )
        _merge_stats(
            sea_ice_dir=sea_ice_stats_dir,
            ocean_dir=ocean_stats_dir,
            atmos_dir=atmos_stats_dir,
            uncoupled_ocean_dir=input_datasets.stats.ocean_dir,
            uncoupled_atmos_dir=input_datasets.stats.atmosphere_dir,
            output_directory=self.coupled_stats_directory,
            debug=debug,
            subsample=subsample,
        )


@dataclasses.dataclass
class CreateCoupledDatasetsConfig:
    """Top-level configuration for creating coupled datasets.

    This configuration orchestrates the creation of coupled datasets from
    separate atmosphere, ocean, and sea ice datasets, including window
    averaging, coupling logic, and statistics computation.

    Parameters:
        coupled_datasets: Configuration for coupled dataset processing.
        input_datasets: Configuration of input data stores and corresponding stats
            directories.
    """

    coupled_datasets: CoupledDatasetsConfig
    input_datasets: InputDatasetsConfig

    def __post_init__(self):
        """Initialize and print dataset information after dataclass construction."""
        self.input_datasets.log_info()

    def write_coupled_datasets(self, debug: bool, subsample: bool):
        """Create coupled atmosphere and ocean datasets.

        Args:
            debug: If True, run in debug mode (print instead of write).
            subsample: If True, subsample to one year of data.
        """
        logging.info("")
        logging.info("=" * 80)
        logging.info("Creating coupled datasets")
        logging.info("=" * 80)
        self.coupled_datasets.write_datasets_and_stats(
            input_datasets=self.input_datasets,
            debug=debug,
            subsample=subsample,
        )
        logging.info("Completed coupled datasets creation")

    @classmethod
    def from_file(cls, path: str) -> "CreateCoupledDatasetsConfig":
        """Load configuration from a YAML file.

        Parameters:
            path: Path to the YAML configuration file.

        Returns:
            CreateCoupledDatasetsConfig instance.
        """
        with open(path, "r") as file:
            data = yaml.safe_load(file)

        return dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )


@click.command()
@click.option("--yaml", help="Path to dataset configuration YAML file.")
@click.option(
    "--debug",
    is_flag=True,
    help="Print metadata instead of writing output.",
)
@click.option(
    "--subsample", is_flag=True, help="Subsample one year of the data before writing."
)
def main(
    yaml: str,
    debug: bool,
    subsample: bool,
):
    """Create coupled atmosphere-ocean datasets from configuration file.

    This script orchestrates the creation of window-averaged and coupled datasets
    for training coupled climate emulators. It processes atmosphere, ocean, and
    optionally sea ice data, applies coupling logic, and computes statistics.

    Args:
        yaml: Path to YAML configuration file defining datasets and processing.
        debug: If True, print dataset info instead of writing to disk.
        subsample: If True, process only one year of data for testing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("=" * 80)
    logging.info("Starting coupled datasets creation pipeline")
    logging.info(f"  Configuration file: {yaml}")
    logging.info(f"  Debug mode: {debug}")
    logging.info(f"  Subsample mode: {subsample}")
    logging.info("=" * 80)

    logging.info("Loading configuration from YAML file")
    config = CreateCoupledDatasetsConfig.from_file(yaml)

    config.write_coupled_datasets(debug=debug, subsample=subsample)

    logging.info("")
    logging.info("=" * 80)
    logging.info("Pipeline completed successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
