import dataclasses
import logging
import os
from typing import Literal

import click
import dacite
import xarray as xr
import yaml
from combine_stats import combine_stats
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
    start_date: str | None = None,
    end_date: str | None = None,
    debug: bool = False,
) -> str:
    """Thin wrapper around get_stats.

    Note that distributed stats computation is handled directly by the
    imported get_stats util.

    Args:
        input_zarr_path: Path to the input zarr dataset.
        climate_data_type: Type of climate data for statistics computation.
        start_date: StatsConfig start date.
        end_date: StatsConfig end date.
        debug: If True, do nothing because the input_zarr_path doesn't exist.

    """
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
                start_date=start_date,
                end_date=end_date,
            ),
            input_zarr=input_zarr_path,
            run_name=run_name,
            debug=False,
        )
    return stats_dir


def _merge_stats(
    sea_ice_dir: str,
    ocean_dir: str | None,
    atmos_dir: str | None,
    uncoupled_ocean_dir: str | None,
    uncoupled_atmos_dir: str,
    output_directory: str,
    debug: bool,
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

    """
    if debug:
        logging.info("Skipping stats merging in debug mode")
        return

    # uncoupled atmosphere training (coupled_sea_ice + uncoupled_atmos)
    output_dir = os.path.join(output_directory, "uncoupled_atmosphere")
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

    def _merge_stats_helper(
        input_dir: str | None,
        uncoupled_dir: str | None,
        output_subdir: str,
        label: str,
    ):
        if input_dir is None:
            return

        output_dir = os.path.join(output_directory, output_subdir)
        if os.path.exists(output_dir):
            logging.info(
                f"Merged {label} stats already exist at {output_dir}, skipping"
            )
            return

        if uncoupled_dir is None:
            raise ValueError(
                f"Cannot merge coupled {label} stats because stats directory for "
                f"{label} input dataset not configured."
            )

        logging.info(f"Merging stats for {label} training")
        logging.info(f"  Input: {input_dir}")
        logging.info(f"  Input: {uncoupled_dir}")
        logging.info(f"  Output: {output_dir}")
        merge_stats(
            config=MergeStatsConfig(
                input_directories=[input_dir, uncoupled_dir],
                output_directory=output_dir,
            )
        )

    # uncoupled/coupled ocean training (coupled_ocean + uncoupled_ocean)
    _merge_stats_helper(
        input_dir=ocean_dir,
        uncoupled_dir=uncoupled_ocean_dir,
        output_subdir="ocean",
        label="ocean",
    )

    # coupled atmosphere training (coupled_atmos + uncoupled_atmos)
    _merge_stats_helper(
        input_dir=atmos_dir,
        uncoupled_dir=uncoupled_atmos_dir,
        output_subdir="coupled_atmosphere",
        label="coupled atmosphere",
    )


def _combine_ensemble_stats(
    ensemble_stats_dir: str,
    run_names: list[str],
    debug: bool,
    history: str | None = None,
):
    """Combine statistics across ensemble runs for each stat category.

    Args:
        ensemble_stats_dir: Base directory containing per-run stats subdirectories.
        run_names: List of run names corresponding to subdirectories in
            ensemble_stats_dir.
        debug: If True, skip combining (datasets don't exist in debug mode).
        history: Optional history string to add to output datasets.
    """
    if debug:
        logging.info("Skipping ensemble stats combining in debug mode")
        return

    categories = ["uncoupled_atmosphere", "coupled_atmosphere", "ocean"]
    output_directory = os.path.join(ensemble_stats_dir, "combined")

    for category in categories:
        # Build stats_roots for this category, checking which runs have it
        stats_roots = []
        for run_name in run_names:
            category_dir = os.path.join(ensemble_stats_dir, run_name, category)
            if os.path.exists(category_dir):
                # combine_stats uses string concatenation (root + filename),
                # so paths must end with "/"
                stats_roots.append(category_dir + "/")

        # If we have any runs with this category, combine them
        if stats_roots:
            logging.info(f"Combining {len(stats_roots)} ensemble stats for {category}")
            combine_stats(
                stats_roots=stats_roots,
                output_directory=output_directory,
                subdirectory=category,
                history=history,
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

    def log_info(self, label: Literal["sea ice", "ocean", "atmosphere"]):
        """Log information about the dataset configuration."""
        logging.info(f"CoupledInputDatsetConfig: {label} dataset")
        logging.info(f"  zarr_path: {self.zarr_path}")
        logging.info(f"  time_chunk_size: {self.time_chunk_size}")
        if self.first_timestamp or self.last_timestamp:
            logging.info(
                f"  time_range: {self.first_timestamp or 'start'} to "
                f"{self.last_timestamp or 'end'}"
            )
        if self.extra_fields.names_and_prefixes:
            logging.info("  extra_fields: " f"{self.extra_fields.names_and_prefixes}")


@dataclasses.dataclass
class NullCoupledInputDatasetConfig:
    """Null version of CoupledInputDatasetConfig."""

    extra_fields = None

    def get_dataset(self) -> None:
        return None

    def log_info(self, label: Literal["sea ice", "ocean"]):
        """Log information about the dataset configuration."""
        logging.info(f"The {label} dataset was not configured")


StatsType = Literal["centering", "scaling-full-field", "scaling-residual", "time-mean"]


@dataclasses.dataclass
class InputStatsConfig:
    """Configuration of input stats datasets.

    Parameters:
        atmosphere_dir: Directory name of atmosphere statistics.
        ocean_dir: Directory name of ocean statistics.
    """

    atmosphere_dir: str
    ocean_dir: str | None = None


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
    stats: InputStatsConfig
    atmosphere: CoupledInputDatasetConfig
    ocean: CoupledInputDatasetConfig | NullCoupledInputDatasetConfig = (
        dataclasses.field(default_factory=NullCoupledInputDatasetConfig)
    )
    sea_ice: CoupledInputDatasetConfig | NullCoupledInputDatasetConfig = (
        dataclasses.field(default_factory=NullCoupledInputDatasetConfig)
    )

    def log_info(self):
        """Log information about all input datasets."""
        logging.info(f"Climate data type: {self.climate_data_type}")
        self.atmosphere.log_info("atmosphere")
        self.ocean.log_info("ocean")
        self.sea_ice.log_info("sea ice")


@dataclasses.dataclass
class EnsembleRunConfig:
    """Configuration for a single run in an ensemble (without shared fields)."""

    atmosphere: CoupledInputDatasetConfig
    ocean: CoupledInputDatasetConfig | NullCoupledInputDatasetConfig = (
        dataclasses.field(default_factory=NullCoupledInputDatasetConfig)
    )
    sea_ice: CoupledInputDatasetConfig | NullCoupledInputDatasetConfig = (
        dataclasses.field(default_factory=NullCoupledInputDatasetConfig)
    )


@dataclasses.dataclass
class InputEnsembleConfig:
    runs: dict[str, EnsembleRunConfig]
    stats: InputStatsConfig
    climate_data_type: ClimateDataType

    def log_info(self):
        """Log information about all input datasets."""
        logging.info(f"Climate data type: {self.climate_data_type}")
        logging.info(f"Ensemble with {len(self.runs)} runs:")
        for run_name, run_config in self.runs.items():
            logging.info(f"  {run_name}:")
            logging.info(f"    atmosphere: {run_config.atmosphere.zarr_path}")
            if not isinstance(run_config.ocean, NullCoupledInputDatasetConfig):
                logging.info(f"    ocean: {run_config.ocean.zarr_path}")
            if not isinstance(run_config.sea_ice, NullCoupledInputDatasetConfig):
                logging.info(f"    sea_ice: {run_config.sea_ice.zarr_path}")


@dataclasses.dataclass
class CoupledStatsConfig:
    start_date: str | None = None
    end_date: str | None = None


@dataclasses.dataclass
class CoupledDatasetsConfig:
    """Configuration for coupled dataset processing.

    Parameters:
        coupled_ts: Configuration for surface temperature in coupled atmosphere.
        coupled_sea_surface: Configuration for sea ice in coupled ocean.
        coupled_sea_ice: Configuration for for sea ice in the uncoupled atmosphere.
        input_field_names: Names of input fields.
        output_writer: Configuration for writing to output store.
    """

    coupled_sea_ice: CoupledSeaIceConfig = dataclasses.field(
        default_factory=CoupledSeaIceConfig
    )
    coupled_ts: CoupledSurfaceTemperatureConfig | None = None
    coupled_sea_surface: CoupledSeaSurfaceConfig | None = None
    input_field_names: CoupledFieldNamesConfig = dataclasses.field(
        default_factory=CoupledFieldNamesConfig
    )
    output_writer: OutputWriterConfig = dataclasses.field(
        default_factory=OutputWriterConfig
    )

    def validate(self, input_datasets: InputDatasetsConfig):
        if self.coupled_sea_surface is None:
            if self.coupled_ts is not None:
                raise ValueError(
                    "coupled_ts configured but coupled_sea_surface config is None. "
                    "The coupled ocean dataset must be created in order to compute "
                    "the coupled atmosphere dataset. Please remove coupled_ts or add "
                    "coupled_sea_surface and configure the ocean input dataset."
                )
        elif isinstance(input_datasets.ocean, NullCoupledInputDatasetConfig):
            raise ValueError(
                "Ocean input dataset not configured but coupled_sea_surface "
                "config is not None."
            )

    def _get_and_merge_stats(
        self,
        input_datasets: InputDatasetsConfig,
        compute_ocean: bool,
        compute_atmos: bool,
        sea_ice_output_store: str,
        ocean_output_store: str,
        atmosphere_output_store: str,
        coupled_stats_directory: str,
        stats_config: CoupledStatsConfig,
        debug: bool,
    ):
        sea_ice_stats_dir = _get_stats(
            input_zarr_path=sea_ice_output_store,
            climate_data_type=input_datasets.climate_data_type,
            start_date=stats_config.start_date,
            end_date=stats_config.end_date,
            debug=debug,
        )

        atmos_stats_dir: str | None = None
        if compute_atmos:
            atmos_stats_dir = _get_stats(
                input_zarr_path=atmosphere_output_store,
                climate_data_type=input_datasets.climate_data_type,
                start_date=stats_config.start_date,
                end_date=stats_config.end_date,
                debug=debug,
            )

        ocean_stats_dir: str | None = None
        if compute_ocean:
            ocean_stats_dir = _get_stats(
                input_zarr_path=ocean_output_store,
                climate_data_type=input_datasets.climate_data_type,
                start_date=stats_config.start_date,
                end_date=stats_config.end_date,
                debug=debug,
            )

        _merge_stats(
            sea_ice_dir=sea_ice_stats_dir,
            ocean_dir=ocean_stats_dir,
            atmos_dir=atmos_stats_dir,
            uncoupled_ocean_dir=input_datasets.stats.ocean_dir,
            uncoupled_atmos_dir=input_datasets.stats.atmosphere_dir,
            output_directory=coupled_stats_directory,
            debug=debug,
        )

    def write_datasets_and_stats(
        self,
        input_datasets: InputDatasetsConfig,
        sea_ice_output_store: str,
        ocean_output_store: str,
        atmosphere_output_store: str,
        coupled_stats_directory: str,
        stats_config: CoupledStatsConfig,
        debug: bool,
        subsample: bool,
    ):
        self.validate(input_datasets)

        logging.info("=" * 80)
        logging.info("Creating coupled atmosphere-ocean datasets")
        logging.info(f"  Atmosphere output: {atmosphere_output_store}")
        logging.info(f"  Ocean output: {ocean_output_store}")
        logging.info(f"  Sea ice output: {sea_ice_output_store}")
        logging.info(f"  Stats output: {coupled_stats_directory}")
        logging.info("=" * 80)

        # Check if outputs already exist (unless in debug mode)
        sea_ice_exists = not debug and os.path.exists(sea_ice_output_store)
        ocean_exists = not debug and os.path.exists(ocean_output_store)
        atmos_exists = not debug and os.path.exists(atmosphere_output_store)

        if sea_ice_exists:
            logging.warning(
                f"Coupled sea ice output {sea_ice_output_store} already exists. "
                "It will be recomputed but not overwritten."
            )

        compute_ocean = self.coupled_sea_surface is not None
        compute_atmos = self.coupled_ts is not None

        if compute_ocean and ocean_exists:
            logging.warning(
                f"Coupled ocean output {ocean_output_store} already exists. "
                "It will be recomputed but not overwritten."
            )

        if compute_atmos and atmos_exists:
            logging.warning(
                f"Coupled atmosphere output {atmosphere_output_store} already "
                "exists. It will be recomputed but not overwritten."
            )

        all_exist = sea_ice_exists
        if compute_ocean:
            all_exist = all_exist and ocean_exists
        if compute_atmos:
            all_exist = all_exist and atmos_exists

        if all_exist and not debug:
            logging.info(f"All coupled output datasets already exist. Skipping...")
            # Still compute stats if they don't exist
            self._get_and_merge_stats(
                input_datasets,
                compute_ocean=compute_ocean,
                compute_atmos=compute_atmos,
                sea_ice_output_store=sea_ice_output_store,
                ocean_output_store=ocean_output_store,
                atmosphere_output_store=atmosphere_output_store,
                coupled_stats_directory=coupled_stats_directory,
                stats_config=stats_config,
                debug=debug,
            )
            return

        self.output_writer.start_dask_client(debug)

        atmos = input_datasets.atmosphere.get_dataset()
        ocean = input_datasets.ocean.get_dataset()
        sea_ice = input_datasets.sea_ice.get_dataset()

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
            sea_ice_extras=input_datasets.sea_ice.extra_fields,
        )

        coupled_ocean: xr.Dataset | None = None
        coupled_atmos: xr.Dataset | None = None

        if compute_ocean:
            assert self.coupled_sea_surface is not None
            logging.info("=" * 80)
            logging.info("Computing coupled ocean fields")
            logging.info("=" * 80)
            # Validation ensures ocean is not None if coupled_sea_surface is set
            assert ocean is not None
            coupled_ocean = compute_coupled_ocean(
                ocean=ocean,
                atmos=atmos,
                # drop extra atmos fields from sea ice dataset
                coupled_sea_ice=atmos_extras.drop_extra_data_vars(coupled_sea_ice),
                config=self.coupled_sea_surface,
                input_field_names=self.input_field_names,
                extras=input_datasets.ocean.extra_fields,
            )

        if compute_atmos:
            assert self.coupled_ts is not None
            logging.info("=" * 80)
            logging.info("Computing coupled atmosphere fields")
            logging.info("=" * 80)
            # Validation ensures ocean/coupled_ocean are set if coupled_ts is set
            assert ocean is not None
            assert coupled_ocean is not None
            coupled_atmos = compute_coupled_atmosphere(
                atmos=atmos,
                ocean=ocean,
                coupled_ocean=coupled_ocean,
                config=self.coupled_ts,
                input_field_names=self.input_field_names,
                extras=atmos_extras,
            )

        if subsample:
            tdim = self.input_field_names.time_dim
            if coupled_ocean is not None:
                logging.info(f"Subsampling coupled ocean to 73 timesteps")
                coupled_ocean = coupled_ocean.isel({tdim: slice(None, 73)})
            if coupled_atmos is not None:
                logging.info(f"Subsampling coupled atmosphere to 1460 timesteps")
                coupled_atmos = coupled_atmos.isel({tdim: slice(None, 365 * 4)})
            logging.info(f"Subsampling coupled sea ice to 1460 timesteps")
            coupled_sea_ice = coupled_sea_ice.isel({tdim: slice(None, 365 * 4)})

        if debug:
            with xr.set_options(display_max_rows=500):
                if coupled_ocean is not None:
                    logging.info("Debug mode: printing coupled ocean dataset")
                    print(coupled_ocean)
                if coupled_atmos is not None:
                    logging.info("Debug mode: printing coupled atmosphere dataset")
                    print(coupled_atmos)
                logging.info("Debug mode: printing coupled sea ice dataset")
                print(coupled_sea_ice)
        else:
            if not sea_ice_exists:
                self.output_writer.write(coupled_sea_ice, sea_ice_output_store)

            if not atmos_exists and coupled_atmos is not None:
                self.output_writer.write(coupled_atmos, atmosphere_output_store)

            if not ocean_exists and coupled_ocean is not None:
                self.output_writer.write(coupled_ocean, ocean_output_store)

        self.output_writer.close_dask_client()

        self._get_and_merge_stats(
            input_datasets,
            compute_ocean=compute_ocean,
            compute_atmos=compute_atmos,
            sea_ice_output_store=sea_ice_output_store,
            ocean_output_store=ocean_output_store,
            atmosphere_output_store=atmosphere_output_store,
            coupled_stats_directory=coupled_stats_directory,
            stats_config=stats_config,
            debug=debug,
        )


@dataclasses.dataclass
class CreateCoupledDatasetsConfig:
    """Top-level configuration for creating coupled datasets.

    This configuration orchestrates the creation of coupled datasets from
    separate atmosphere, ocean, and sea ice datasets, including window
    averaging, coupling logic, and statistics computation.

    Parameters:
        version: Version string for the output datasets.
        family_name: Common name for the family of coupled output datasets.
        output_directory: Directory where outputs will be written.
        coupled_datasets: Configuration for coupled dataset processing.
        input_datasets: Configuration of input data stores and corresponding stats
            directories.
        stats: Configuration of output dataset stats start and end dates.
    """

    version: str
    family_name: str
    output_directory: str
    coupled_datasets: CoupledDatasetsConfig
    input_datasets: InputDatasetsConfig | InputEnsembleConfig
    stats: CoupledStatsConfig = dataclasses.field(default_factory=CoupledStatsConfig)
    _history: str | None = None  # added by from_file

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

        Returns:
            Absolute path to the coupled atmosphere output zarr store.

        """
        return os.path.join(
            self.output_directory,
            f"{self.version}-{self.family_name}-atmosphere.zarr",
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

        def _subsample_path(path: str) -> str:
            """Add '-subsample' suffix to zarr or stats paths if subsample=True."""
            if not subsample:
                return path
            if path.endswith(".zarr"):
                return path.replace(".zarr", "-subsample.zarr")
            elif "-stats" in path:
                return path.replace("-stats", "-subsample-stats")
            return path

        if isinstance(self.input_datasets, InputDatasetsConfig):
            # Single dataset: use property-based output paths
            self.coupled_datasets.write_datasets_and_stats(
                input_datasets=self.input_datasets,
                sea_ice_output_store=_subsample_path(self.sea_ice_output_store),
                ocean_output_store=_subsample_path(self.ocean_output_store),
                atmosphere_output_store=_subsample_path(self.atmosphere_output_store),
                coupled_stats_directory=_subsample_path(self.coupled_stats_directory),
                stats_config=self.stats,
                debug=debug,
                subsample=subsample,
            )
        elif isinstance(self.input_datasets, InputEnsembleConfig):
            # Ensemble: loop over runs with run-specific output paths
            ensemble_dir = os.path.join(
                self.output_directory, f"{self.version}-{self.family_name}"
            )
            ensemble_stats_dir = os.path.join(
                self.output_directory, f"{self.version}-{self.family_name}-stats"
            )
            if subsample:
                ensemble_dir = ensemble_dir + "-subsample"
                ensemble_stats_dir = ensemble_stats_dir.replace(
                    "-stats", "-subsample-stats"
                )
            for run_name, run_config in self.input_datasets.runs.items():
                logging.info("")
                logging.info("=" * 80)
                logging.info(f"Processing ensemble run: {run_name}")
                logging.info("=" * 80)

                # Construct full InputDatasetsConfig for this run
                run_input_datasets = InputDatasetsConfig(
                    climate_data_type=self.input_datasets.climate_data_type,
                    stats=self.input_datasets.stats,
                    atmosphere=run_config.atmosphere,
                    ocean=run_config.ocean,
                    sea_ice=run_config.sea_ice,
                )

                # Generate run-specific output paths
                sea_ice_output_store = os.path.join(
                    ensemble_dir, f"{run_name}-sea_ice.zarr"
                )
                ocean_output_store = os.path.join(
                    ensemble_dir, f"{run_name}-ocean.zarr"
                )
                atmosphere_output_store = os.path.join(
                    ensemble_dir, f"{run_name}-atmosphere.zarr"
                )
                coupled_stats_directory = os.path.join(ensemble_stats_dir, run_name)

                self.coupled_datasets.write_datasets_and_stats(
                    input_datasets=run_input_datasets,
                    sea_ice_output_store=sea_ice_output_store,
                    ocean_output_store=ocean_output_store,
                    atmosphere_output_store=atmosphere_output_store,
                    coupled_stats_directory=coupled_stats_directory,
                    stats_config=self.stats,
                    debug=debug,
                    subsample=subsample,
                )

            logging.info("Combining ensemble stats...")
            _combine_ensemble_stats(
                ensemble_stats_dir,
                run_names=list(self.input_datasets.runs.keys()),
                debug=debug,
                history=self._history,
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

        config = dacite.from_dict(
            data_class=cls, data=data, config=dacite.Config(cast=[tuple], strict=True)
        )
        config._history = (
            "Created by full-model/data_process/create_coupled_dataset.py from "
            f"configuration file {path}."
        )
        return config


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
