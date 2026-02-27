"""
Process raw zarr datasets from multiple resolutions for downscaling experiments.

This script consolidates variables from raw SHiELD simulation outputs at three
resolutions (100km, 25km, 3km) into standardized zarr datasets. It handles:
- Loading variables from single or multiple source zarr files per resolution
- Renaming variables and dimensions to standardized names
- Filtering to specific variables of interest
- Writing output with source path tracking as attributes

The script is designed for flexibility in dataset selection and variable filtering,
with configuration currently hardcoded for the SHiELD AMIP plus-4K simulations.

Usage examples:
    # Process all resolutions with all configured variables
    python process_from_raw_zarrs.py /output/path

    # Process only 25km and 3km resolutions
    python process_from_raw_zarrs.py /output/path --datasets 25km 3km

    # Add specific variables from all resolutions
    python process_from_raw_zarrs.py /output/path --variables PRATEsfc HGTsfc --zarr-mode a

    # Dry run to preview actions
    python process_from_raw_zarrs.py /output/path --dry-run
"""

import argparse
import dataclasses
from typing import Protocol

import dask.distributed as dd
import xarray as xr
from obstore.store import GCSStore
from zarr.storage import ObjectStore


class DatasetLoader(Protocol):
    """Protocol for dataset loaders that can load datasets and track source paths.

    Implementations should handle loading variables from one or more source files
    and provide source path information for each variable.
    """

    name: str

    def load_dataset(self) -> xr.Dataset:
        """Load and return the dataset with all configured variables."""
        ...

    def get_source_path(self, variable_name: str) -> str:
        """Return the source path (zarr file) for a given variable name."""
        ...


def _validate_variables_exist(
    ds: xr.Dataset, variable_names: list[str], dataset_name: str, path: str
) -> None:
    """Validate that all requested variables exist in the dataset.

    Args:
        ds: The opened xarray Dataset
        variable_names: List of variable names to validate
        dataset_name: Name of the dataset (for error messages)
        path: Path to the source file (for error messages)

    Raises:
        ValueError: If any variables are missing from the dataset
    """
    missing_vars = [var for var in variable_names if var not in ds.data_vars]
    if missing_vars:
        raise ValueError(
            f"Variables {missing_vars} not found in dataset '{dataset_name}' at {path}. "
            f"Available variables in source: {list(ds.data_vars)}"
        )


def _open_gcs_obstore(path: str) -> xr.Dataset:
    # extract bucket and prefix from "gs://bucket/prefix" path
    protocol, _, bucket, *prefix_parts = path.split("/")
    prefix = "/".join(prefix_parts)
    # open GCS store and dataset
    gcs_store = GCSStore(bucket, prefix=prefix)
    store = ObjectStore(gcs_store, read_only=True)
    ds = xr.open_dataset(store, engine="zarr", chunks="auto")
    return ds


def _maybe_rechunk_with_shard(ds: xr.Dataset) -> xr.Dataset:
    key = list(ds.data_vars.keys())[0]
    encoding = ds[key].encoding
    if "shards" in encoding and encoding["shards"] is not None:
        chunk_leading = encoding["chunks"][0]
        shard_leading = encoding["shards"][0]
        if chunk_leading != shard_leading:
            print(
                f"Rechunking dataset to match shard size for variable '{key}': "
                f"chunk size {chunk_leading} -> {shard_leading}"
            )
            ds = ds.chunk({ds[key].dims[0]: shard_leading})
    return ds


def _open_zarr_store(path: str) -> xr.Dataset:
    if path.startswith("gs://"):
        ds = _open_gcs_obstore(path)
    else:
        raise NotImplementedError(
            f"Unsupported path protocol in {path}. Only 'gs://' is supported."
        )

    ds = _maybe_rechunk_with_shard(ds)
    return ds


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for loading a dataset from a single zarr file.

    Use this when all variables are stored in a single source zarr file.

    Attributes:
        name: Identifier for this dataset (e.g., "100km")
        path: Full path to the source zarr file
        variable_names: List of variable names to extract from the source
        rename_map: Mapping of source names to target names for variables and dimensions
    """

    name: str
    path: str
    variable_names: list[str]
    rename_map: dict[str, str]

    def load_dataset(self) -> xr.Dataset:
        """Load the dataset, selecting and renaming specified variables."""
        ds = _open_zarr_store(self.path)

        # Validate that all configured variables exist in the source
        _validate_variables_exist(ds, self.variable_names, self.name, self.path)

        ds = ds[self.variable_names]
        # Add source path tracking to each variable's attributes
        for variable in ds.data_vars:
            ds[variable].attrs["source_path"] = self.path

        ds = ds.rename(self.rename_map)
        return ds


@dataclasses.dataclass
class DatasetPerVariableConfig:
    """Configuration for loading a dataset from multiple zarr files.

    Use this when variables are split across multiple source zarr files. This class
    loads each file separately, selects the specified variables, and merges them.

    Attributes:
        name: Identifier for this dataset (e.g., "25km")
        base_path: Base directory containing all source zarr files
        filename_map: Mapping from logical names (keys) to actual zarr filenames
        per_file_variables: Mapping from logical names to lists of variables in
            each file
        rename_map: Mapping of source names to target names for variables and
            dimensions
        extra_paths: Optional mapping of variable names to full paths, used to
            override the base_path + filename_map for specific variables (e.g., land 
            fraction)
    """

    name: str
    base_path: str
    filename_map: dict[str, str]
    per_file_variables: dict[str, list[str]]
    rename_map: dict[str, str]
    extra_paths: dict[str, str] | None = None

    def __post_init__(self):
        """Validate that no variable appears in multiple files."""
        all_variables = []
        for variable_list in self.per_file_variables.values():
            all_variables.extend(variable_list)
        if len(all_variables) != len(set(all_variables)):
            non_unique = set(
                [var for var in all_variables if all_variables.count(var) > 1]
            )
            raise ValueError(
                "Please ensure each variable is only listed once across all "
                f"files. Non-unique variables: {non_unique}"
            )

    def load_dataset(self) -> xr.Dataset:
        """Load variables from multiple files and merge into a single dataset."""
        datasets = []
        for filename_key, variable_names in self.per_file_variables.items():
            path = f"{self.base_path}/{self.filename_map[filename_key]}"
            ds = _open_zarr_store(path)

            # Validate that all configured variables exist in this source file
            _validate_variables_exist(ds, variable_names, self.name, path)

            ds = ds[variable_names]

            # Add source path tracking to each variable's attributes
            for variable in ds.data_vars:
                ds[variable].attrs["source_path"] = path

            ds = ds.rename(self.rename_map)
            datasets.append(ds)

        if self.extra_paths is not None:
            for variable, extra_path in self.extra_paths.items():
                ds_extra = _open_zarr_store(extra_path)

                # Validate that the variable exists in the extra path
                _validate_variables_exist(
                    ds_extra, [variable], self.name, extra_path
                )

                ds_extra = ds_extra[[variable]]
                ds_extra[variable].attrs["source_path"] = extra_path
                ds_extra = ds_extra.rename(self.rename_map)
                datasets.append(ds_extra)
        
        combined_ds = xr.merge(datasets)
        return combined_ds


# =============================================================================
# Dataset Configuration
# =============================================================================
# Configuration is hardcoded here for the SHiELD AMIP plus-4K simulations.
# If more datasets are added, consider moving to a YAML config file.

# Logical groupings for different types of fields in the source data
FLUX_FIELDS = "fluxes"
INSTANTANEOUS_PHYSICS_FIELDS = "inst_phys"
STATIC_FIELDS = "static"
PRESSURE_FIELDS = "pressures"
LAND_FRAC_25KM = "land_frac_25km"
LAND_FRAC_3KM = "land_frac_3km"

# Mapping from logical field groups to actual zarr filenames
RAW_FILENAMES = {
    FLUX_FIELDS: "fluxes_2d.zarr",
    INSTANTANEOUS_PHYSICS_FIELDS: "instantaneous_physics_fields.zarr",
    STATIC_FIELDS: "static.zarr",
    PRESSURE_FIELDS: "instantaneous_surface_and_sea_level_pressure.zarr",
}

LAND_FRAC_25KM_PATH = "gs://vcm-ml-intermediate/2025-11-19-X-SHiELD-AMIP-downscaling-land-fraction/25km/land_fraction.zarr"  # noqa
LAND_FRAC_3KM_PATH = "gs://vcm-ml-intermediate/2025-11-19-X-SHiELD-AMIP-downscaling-land-fraction/3km/land_fraction.zarr"  # noqa

# Variables to extract from each file group (for 25km and 3km resolutions)
# Note: These use standardized variable names that already exist in the source files
FILENAME_KEY_TO_VARIABLES = {
    FLUX_FIELDS: ["PRATEsfc"],
    INSTANTANEOUS_PHYSICS_FIELDS: [
        "air_temperature_at_two_meters",
        "eastward_wind_at_ten_meters",
        "northward_wind_at_ten_meters",
    ],
    STATIC_FIELDS: [
        "HGTsfc",  # Note: land_fraction not available in 25km/3km data
    ],
    PRESSURE_FIELDS: ["PRESsfc", "PRMSL"],
}

# Source data paths for SHiELD AMIP plus-4K simulations
COARSE_RES_PATH = (  # ~100km resolution
    "gs://vcm-ml-intermediate/2026-02-12-X-SHiELD-AMIP-plus-4K-1deg-8layer-1yr.zarr"
)
MED_RES_BASE_PATH = (  # ~25km resolution
    "gs://vcm-ml-raw-flexible-retention/2026-02-09-X-SHiELD-AMIP-plus-4K-downscaling/"
    "regridded-zarrs/gaussian_grid_180_by_360_refined_to_720_by_1440/plus-4K"
)
FINE_RES_BASE_PATH = (  # ~3km resolution
    "gs://vcm-ml-raw-flexible-retention/2026-02-09-X-SHiELD-AMIP-plus-4K-downscaling/"
    "regridded-zarrs/gaussian_grid_180_by_360_refined_to_5760_by_11520/plus-4K"
)


def main(
    output_path: str,
    dataset_loaders: list[DatasetLoader],
    zarr_mode: str,
    variables_to_process: list[str] | None,
    nprocesses: int,
    dry_run: bool = False,
):
    """Process datasets and write to output zarr files.

    For each dataset loader:
    1. Load the dataset with configured variables
    2. Add source_path attributes to track data provenance
    3. Filter to requested variables if specified
    4. Write to output (or print plan if dry_run)

    Args:
        output_path: Directory where output zarr files will be written
        dataset_loaders: List of configured dataset loaders (one per resolution)
        zarr_mode: Mode for xarray.Dataset.to_zarr ('w', 'w-', or 'a')
        variables_to_process: Optional list of variables to filter to (None = all)
        nprocesses: Number of Dask worker processes to use
        dry_run: If True, print actions without writing any data
    """
    with dd.Client(n_workers=nprocesses):
        for loader in dataset_loaders:
            print(f"Processing dataset: {loader.name}")

            # Load dataset with configured variables
            ds = loader.load_dataset()
            print(
                f"Opened dataset of {len(ds.data_vars)} variables: {list(ds.data_vars)}"
            )

            # Filter to requested variables if specified
            if variables_to_process is not None:
                _validate_variables_exist(
                    ds, variables_to_process, loader.name, "merged dataset"
                )
                ds = ds[variables_to_process]
                print(f"Filtered to {variables_to_process} requested variables")

            # Write output or print dry-run message
            output_file = f"{output_path}/{loader.name}.zarr"
            if not dry_run:
                print(f"  Writing to {output_file} with mode '{zarr_mode}'")
                ds.to_zarr(output_file, mode=zarr_mode)
                print(f"  ✓ Successfully wrote {loader.name}")
            else:
                print(f"  [DRY RUN] Dataset chunks: {dict(ds.chunks)}")
                print(
                    f"  [DRY RUN] Would write {len(ds.data_vars)} variables to "
                    f"{output_file} with mode '{zarr_mode}'"
                )


if __name__ == "__main__":
    # Parse command-line arguments
    choices = ("100km", "25km", "3km", "all")
    parser = argparse.ArgumentParser(
        description="Process raw zarr datasets for downscaling."
    )
    parser.add_argument(
        "output_path", help="Directory where processed zarr files will be written."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=choices,
        default=["all"],
        help=(
            "Which datasets to process. Choose from '100km', '25km', '3km', or 'all'."
            " Default is 'all'."
        ),
    )
    # https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html
    parser.add_argument(
        "--zarr-mode",
        choices=("w", "w-", "a"),
        default="w-",
        help=(
            "Mode to use when writing zarr datasets. See xarray.Dataset.to_zarr "
            "documentation for details. Default is 'w-' (write, fail if exists)."
        ),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help=(
            "List of variables (from the source naming) to process. If not set, "
            "all variables in config for the selected dataset(s) will be processed."
        ),
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=8,
        help="Number of processes to use for dask operations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "If set, will print out the actions that would be taken without "
            "actually processing or writing any datasets."
        ),
    )
    args = parser.parse_args()

    # Configure dataset loaders for each resolution
    # COARSE Is missing PRMSL
    coarse_config = DatasetConfig(
        name="100km",
        path=COARSE_RES_PATH,
        variable_names=[
            "PRATEsfc",
            "PRESsfc",
            "TMP2m",
            "UGRD10m",
            "VGRD10m",
            "HGTsfc",
            "land_fraction",
        ],
        rename_map={
            "TMP2m": "air_temperature_at_two_meters",
            "UGRD10m": "eastward_wind_at_ten_meters",
            "VGRD10m": "northward_wind_at_ten_meters",
            "grid_xt": "latitude",
            "grid_yt": "longitude",
        },
    )

    mid_res_config = DatasetPerVariableConfig(
        name="25km",
        base_path=MED_RES_BASE_PATH,
        filename_map=RAW_FILENAMES,
        per_file_variables=FILENAME_KEY_TO_VARIABLES,
        rename_map={"grid_xt": "latitude", "grid_yt": "longitude"},
        extra_paths={"land_fraction": LAND_FRAC_25KM_PATH},
    )

    fine_res_config = DatasetPerVariableConfig(
        name="3km",
        base_path=FINE_RES_BASE_PATH,
        filename_map=RAW_FILENAMES,
        per_file_variables=FILENAME_KEY_TO_VARIABLES,
        rename_map={"grid_xt": "latitude", "grid_yt": "longitude"},
        extra_paths={"land_fraction": LAND_FRAC_3KM_PATH},
    )

    # Select which datasets to process based on command-line arguments
    selected_loaders: list[DatasetLoader]
    if "all" in args.datasets:
        selected_loaders = [coarse_config, mid_res_config, fine_res_config]
    else:
        selected_loaders = []
        if "100km" in args.datasets:
            selected_loaders.append(coarse_config)
        if "25km" in args.datasets:
            selected_loaders.append(mid_res_config)
        if "3km" in args.datasets:
            selected_loaders.append(fine_res_config)

    # Run the processing pipeline
    main(
        output_path=args.output_path,
        dataset_loaders=selected_loaders,
        zarr_mode=args.zarr_mode,
        variables_to_process=args.variables,
        nprocesses=args.nprocesses,
        dry_run=args.dry_run,
    )
