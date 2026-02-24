import glob
import logging
import os
import re
from dataclasses import dataclass, field

import click
import dacite
import xarray as xr
import yaml
from create_coupled_datasets import (
    CreateCoupledDatasetsConfig,
    InputDatasetsConfig,
    NullCoupledInputDatasetConfig,
)


@dataclass
class TimeSelectionConfig:
    """Time selection: either a single timestamp or a range."""

    timestamp: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    def validate(self):
        if self.timestamp is not None:
            if self.start_time is not None or self.end_time is not None:
                raise ValueError(
                    "Use either 'timestamp' or 'start_time'/'end_time', not both."
                )
        elif self.start_time is None or self.end_time is None:
            raise ValueError(
                "Provide either 'timestamp' or both 'start_time' and 'end_time'."
            )


@dataclass
class CreateCoupledICConfig:
    """Configuration for creating coupled initial condition NetCDF files.

    Parameters:
        coupled_config_path: Path to the YAML used for create_coupled_datasets
            (used to get output paths and original zarr paths).
        coupled_ocean_zarr: Optional path (or glob) to coupled ocean zarr.
            Defaults to {output_directory}/{version}-{family_name}-ocean.zarr.
        coupled_atmosphere_zarr: Optional path (or glob) to coupled atmosphere
            zarr. Defaults to path with how from coupled config:
            {output_directory}/{version}-{family_name}-{how}-atmosphere.zarr.
        original_ocean_zarr: Optional path to original ocean zarr. Defaults to
            input_datasets.ocean.zarr_path from the coupled config.
        original_atmosphere_zarr: Optional path to original atmosphere zarr.
            Defaults to input_datasets.atmosphere.zarr_path from the coupled config.
        time: Time selection (single timestamp or start_time/end_time range).
        output_directory: Directory where NetCDF files will be written.
        output_prefix: Prefix for output files: {prefix}_ocean_ic.nc and
            {prefix}_atmosphere_ic.nc. Defaults to "ic".
        use_coupled: If True (default), merge coupled zarr with original for
            ocean and atmosphere. If False, use only original ocean and
            atmosphere data (no merging).
    """

    coupled_config_path: str
    coupled_ocean_zarr: str | None = None
    coupled_atmosphere_zarr: str | None = None
    original_ocean_zarr: str | None = None
    original_atmosphere_zarr: str | None = None
    time: TimeSelectionConfig = field(default_factory=TimeSelectionConfig)
    output_directory: str = "."
    output_prefix: str = "ic"
    use_coupled: bool = True

    def resolve_paths(self, coupled_config: CreateCoupledDatasetsConfig) -> None:
        """Resolve zarr paths from coupled config when not overridden."""
        input_datasets = coupled_config.input_datasets
        if not isinstance(input_datasets, InputDatasetsConfig):
            raise ValueError(
                "create_coupled_ic expects a single-dataset coupled config "
                "(InputDatasetsConfig); ensemble configs are not supported. "
                "Use a YAML with input_datasets containing atmosphere/ocean, not runs."
            )
        # Original paths are always needed (for merge or as sole source)
        if self.original_ocean_zarr is None:
            ocean_cfg = input_datasets.ocean
            if isinstance(ocean_cfg, NullCoupledInputDatasetConfig):
                raise ValueError(
                    "original_ocean_zarr not set and could not be inferred from "
                    "coupled config (no ocean input dataset)."
                )
            self.original_ocean_zarr = ocean_cfg.zarr_path
        if self.original_atmosphere_zarr is None:
            self.original_atmosphere_zarr = input_datasets.atmosphere.zarr_path

        if not self.use_coupled:
            return

        if self.coupled_ocean_zarr is None:
            self.coupled_ocean_zarr = coupled_config.ocean_output_store
        elif "*" in self.coupled_ocean_zarr:
            matches = glob.glob(self.coupled_ocean_zarr)
            if len(matches) != 1:
                raise ValueError(
                    f"Glob '{self.coupled_ocean_zarr}' must match exactly one path, "
                    f"got {len(matches)}: {matches}"
                )
            self.coupled_ocean_zarr = matches[0]

        if self.coupled_atmosphere_zarr is None:
            coupled_ts = getattr(coupled_config.coupled_datasets, "coupled_ts", None)
            if coupled_ts is not None and getattr(coupled_ts, "how", None):
                how = coupled_ts.how
                path_with_how = os.path.join(
                    coupled_config.output_directory,
                    f"{coupled_config.version}-{coupled_config.family_name}-{how}-atmosphere.zarr",
                )
                if os.path.exists(path_with_how):
                    self.coupled_atmosphere_zarr = path_with_how
                else:
                    self.coupled_atmosphere_zarr = (
                        coupled_config.atmosphere_output_store
                    )
            else:
                self.coupled_atmosphere_zarr = coupled_config.atmosphere_output_store
        elif "*" in self.coupled_atmosphere_zarr:
            matches = glob.glob(self.coupled_atmosphere_zarr)
            if len(matches) != 1:
                raise ValueError(
                    f"Glob '{self.coupled_atmosphere_zarr}' must match exactly one "
                    f"path, got {len(matches)}: {matches}"
                )
            self.coupled_atmosphere_zarr = matches[0]


def _merge_with_original(coupled_ds: xr.Dataset, original_ds: xr.Dataset) -> xr.Dataset:
    """Merge coupled dataset with original; coupled overwrites overlapping vars."""
    overlap = [v for v in coupled_ds.data_vars if v in original_ds.data_vars]
    original_dropped = original_ds.drop_vars(overlap, errors="ignore")
    return xr.merge([original_dropped, coupled_ds], compat="override", join="override")


def _load_original_only(
    original_zarr: str,
    time_config: TimeSelectionConfig,
    label: str,
) -> xr.Dataset:
    """Load original zarr only and select time (no merging with coupled)."""
    logging.info(f"Loading original {label} only (use_coupled=false): {original_zarr}")
    ds = xr.open_zarr(original_zarr)
    out = _select_time(ds, time_config)
    logging.info(
        f"Original {label}: {len(out.data_vars)} vars, \
            time size {out.sizes.get('time', 1)}"
    )
    return out


def _parse_timestamp_to_cftime(timestamp_str: str, time_coord: xr.DataArray):
    """Parse an ISO-style timestamp string to cftime matching the dataset's time.

    Args:
        timestamp_str: e.g. "0200-01-01T00:00:00"
        time_coord: time coordinate from the dataset (used for cftime type/calendar).

    Returns:
        cftime.datetime instance compatible with time_coord.
    """
    match = re.match(
        r"^(-?\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$", timestamp_str.strip()
    )
    if not match:
        raise ValueError(
            f"Timestamp must be ISO format YYYY-MM-DDTHH:MM:SS, got: {timestamp_str!r}"
        )
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour, minute, second = int(match.group(4)), int(match.group(5)), int(match.group(6))
    values = time_coord.values
    if values.size == 0:
        raise ValueError("Dataset time coordinate is empty")
    cftime_type = type(values.flat[0])
    return cftime_type(year, month, day, hour, minute, second)


def _select_time(ds: xr.Dataset, time_config: TimeSelectionConfig) -> xr.Dataset:
    """Select time slice from dataset. Handles cftime time coordinates."""
    if "time" not in ds.dims:
        return ds
    time_coord = ds.time
    if time_config.timestamp is not None:
        t = _parse_timestamp_to_cftime(time_config.timestamp, time_coord)
        if t not in time_coord.values:
            raise ValueError(
                f"Time {time_config.timestamp!r} not found in dataset. "
                f"Use an exact timestamp present in the data."
            )
        return ds.sel(time=t)
    assert time_config.start_time is not None and time_config.end_time is not None
    start = _parse_timestamp_to_cftime(time_config.start_time, time_coord)
    end = _parse_timestamp_to_cftime(time_config.end_time, time_coord)
    return ds.sel(time=slice(start, end))


def _load_and_merge(
    coupled_zarr: str,
    original_zarr: str,
    time_config: TimeSelectionConfig,
    label: str,
) -> xr.Dataset:
    logging.info(f"Loading coupled {label}: {coupled_zarr}")
    coupled = xr.open_zarr(coupled_zarr)
    logging.info(f"Loading original {label}: {original_zarr}")
    original = xr.open_zarr(original_zarr)
    merged = _merge_with_original(coupled, original)
    out = _select_time(merged, time_config)
    logging.info(
        f"Merged {label}: {len(out.data_vars)} vars, \
            time size {out.sizes.get('time', 1)}"
    )
    return out


def run(config: CreateCoupledICConfig) -> None:
    """Create ocean and atmosphere IC NetCDF files from config."""
    config.time.validate()
    logging.info("Loading coupled dataset config: %s", config.coupled_config_path)
    coupled_config = CreateCoupledDatasetsConfig.from_file(config.coupled_config_path)
    config.resolve_paths(coupled_config)

    # Paths are resolved by resolve_paths (or set in config)
    original_ocean = config.original_ocean_zarr
    original_atmosphere = config.original_atmosphere_zarr
    assert original_ocean is not None
    assert original_atmosphere is not None

    os.makedirs(config.output_directory, exist_ok=True)

    if config.use_coupled:
        coupled_ocean = config.coupled_ocean_zarr
        coupled_atmosphere = config.coupled_atmosphere_zarr
        assert coupled_ocean is not None
        assert coupled_atmosphere is not None
        ocean_ic = _load_and_merge(
            coupled_ocean,
            original_ocean,
            config.time,
            "ocean",
        )
    else:
        ocean_ic = _load_original_only(
            original_ocean,
            config.time,
            "ocean",
        )
    ocean_path = os.path.join(
        config.output_directory, f"{config.output_prefix}_ocean_ic.nc"
    )
    logging.info("Writing ocean IC: %s", ocean_path)
    ocean_ic.to_netcdf(ocean_path)

    if config.use_coupled:
        coupled_atmosphere = config.coupled_atmosphere_zarr
        assert coupled_atmosphere is not None
        atmosphere_ic = _load_and_merge(
            coupled_atmosphere,
            original_atmosphere,
            config.time,
            "atmosphere",
        )
    else:
        atmosphere_ic = _load_original_only(
            original_atmosphere,
            config.time,
            "atmosphere",
        )
    atmos_path = os.path.join(
        config.output_directory, f"{config.output_prefix}_atmosphere_ic.nc"
    )
    logging.info("Writing atmosphere IC: %s", atmos_path)
    atmosphere_ic.to_netcdf(atmos_path)

    logging.info("Done. Ocean IC: %s  Atmosphere IC: %s", ocean_path, atmos_path)


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML config for creating coupled ICs.",
)
def main(config_path: str) -> None:
    """Create initial condition NetCDF files from coupled datasets.

    Reads the coupled-dataset config to locate coupled and original zarr paths,
    merges ocean and atmosphere with their originals, applies the time selection
    from the config, and writes NetCDF files.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    config = dacite.from_dict(
        CreateCoupledICConfig,
        data,
        config=dacite.Config(cast=[tuple], strict=True),
    )
    run(config)


if __name__ == "__main__":
    main()
