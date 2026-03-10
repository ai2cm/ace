import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Mapping

import dacite
import xarray as xr
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_stats import StatsConfig

try:
    import dask  # noqa: F401
    import xpartition  # noqa: F401

    _HAS_XPARTITION = True
except ImportError:
    _HAS_XPARTITION = False


@dataclasses.dataclass
class TimeCoarsenConfig:
    """
    Configuration for time coarsening of a dataset.

    Attributes:
        factor: Factor by which to coarsen the time dimension.
        data_output_directory: Directory to save the coarsened datasets as zarr stores.
        stats_output_directory: Directory to save the stats of the coarsened datasets.
        snapshot_names: List of snapshot variable names to coarsen. These will be
            coarsened by skipping each factor times.
        window_names: List of window variable names to coarsen. These will be
            coarsened by averaging over each factor times.
        constant_prefixes: List of prefixes for constant data variables to copy without
            modification. Raises an exception if any of these have a "time" dimension.
        n_split: Number of partitions to split the write into when using xpartition.
            Only used when dask and xpartition are available.
        chunking: Mapping of dimension names to inner chunk sizes for the output
            zarr store. Defaults to {"time": 1, "lat": -1, "lon": -1}.
        sharding: Mapping of dimension names to shard sizes. If None, an unsharded
            zarr store is written with chunks as specified in ``chunking``.
    """

    factor: int
    data_output_directory: str
    stats_output_directory: str
    snapshot_names: list[str]
    window_names: list[str]
    constant_prefixes: list[str]
    n_split: int = 1
    chunking: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"time": 1, "lat": -1, "lon": -1}
    )
    sharding: dict[str, int] | None = dataclasses.field(
        default_factory=lambda: {"time": 360, "lat": -1, "lon": -1}
    )


@dataclasses.dataclass
class Config:
    runs: Mapping[str, str]
    data_output_directory: str
    stats: StatsConfig
    time_coarsen: TimeCoarsenConfig


def main(config: Config, run: int, dry_run: bool = False):
    logging.basicConfig(level=logging.INFO)
    run_name = list(config.runs.keys())[run]
    if config.data_output_directory.endswith("/"):
        config.data_output_directory = config.data_output_directory[:-1]
    input_zarr = config.data_output_directory + "/" + run_name + ".zarr"
    output_zarr = config.time_coarsen.data_output_directory + "/" + run_name + ".zarr"
    process_path_pair(
        input_path=input_zarr,
        output_path=output_zarr,
        config=config.time_coarsen,
        dry_run=dry_run,
    )


def _set_attributes(
    ds: xr.Dataset, original_attributes: dict, config: TimeCoarsenConfig
) -> None:
    """Set coarsening metadata and history on the dataset attributes in-place."""
    attributes = original_attributes.copy()
    attributes["snapshot_names"] = json.dumps(config.snapshot_names)
    attributes["window_names"] = json.dumps(config.window_names)
    attributes["constant_prefixes"] = json.dumps(config.constant_prefixes)
    attributes["coarsen_factor"] = config.factor
    history_entry = (
        f"Dataset coarsened by a factor of {config.factor} "
        "by scripts/data_process/time_coarsen.py."
    )
    if "history" in attributes:
        attributes["history"] = attributes["history"] + " " + history_entry
    else:
        attributes["history"] = history_entry
    ds.attrs = attributes


def coarsen(ds: xr.Dataset, config: TimeCoarsenConfig) -> xr.Dataset:
    """Apply time coarsening to a dataset.

    Works with both eager (numpy) and lazy (dask) arrays.
    """
    constant_names = [
        name
        for name in ds.data_vars
        if any(name.startswith(prefix) for prefix in config.constant_prefixes)
    ]
    if set(constant_names).intersection(set(config.snapshot_names)):
        raise ValueError(
            "Constant names overlap with snapshot names: "
            f"{set(constant_names).intersection(set(config.snapshot_names))}"
        )
    if set(constant_names).intersection(set(config.window_names)):
        raise ValueError(
            "Constant names overlap with window names: "
            f"{set(constant_names).intersection(set(config.window_names))}"
        )
    for name in constant_names:
        if "time" in ds[name].dims:
            raise ValueError(
                f"Constant data variable {name} has a 'time' dimension, "
                "which is not allowed."
            )
    ds_constants = ds[constant_names]
    ds_snapshot = ds[config.snapshot_names].isel(
        time=slice(config.factor - 1, None, config.factor)
    )
    ds_window = (
        ds[config.window_names]
        .coarsen(time=config.factor, boundary="trim")
        .mean()
        .drop("time")
    )  # use time of snapshots
    ds_coarsened = xr.merge([ds_snapshot, ds_window, ds_constants])
    _set_attributes(ds_coarsened, ds.attrs, config)
    return ds_coarsened


def _write_eager(ds: xr.Dataset, path: str) -> None:
    """Write dataset eagerly using xarray's to_zarr."""
    ds.to_zarr(path, mode="w", zarr_version=3)


def _write_xpartition(ds: xr.Dataset, path: str, config: TimeCoarsenConfig) -> None:
    """Write dataset lazily using xpartition for chunked parallel writes."""
    import xpartition  # noqa: F401

    if config.sharding is None:
        outer_chunks = config.chunking
    else:
        outer_chunks = config.sharding

    ds = ds.chunk(outer_chunks)

    if config.sharding is None:
        inner_chunks = None
    else:
        inner_chunks = config.chunking

    ds.partition.initialize_store(path, inner_chunks=inner_chunks)
    for i in range(config.n_split):
        segment_number = f"{i + 1} / {config.n_split}"
        logging.info(f"Writing segment {segment_number}")
        segment_time = time.time()
        ds.partition.write(
            path,
            config.n_split,
            ["time"],
            i,
            collect_variable_writes=True,
        )
        segment_time = time.time() - segment_time
        logging.info(f"Segment {segment_number} time: {segment_time:0.2f} seconds")


def process_path_pair(
    input_path: str, output_path: str, config: TimeCoarsenConfig, dry_run: bool
):
    logging.info(f"Processing input: {input_path} to output: {output_path}")
    if os.path.exists(output_path):
        logging.warning(f"Output path {output_path} already exists. Skipping.")
        return
    if _HAS_XPARTITION:
        ds = xr.open_zarr(input_path)
    else:
        ds = xr.open_dataset(input_path)
    ds_coarsened = coarsen(ds, config)
    if not dry_run:
        if _HAS_XPARTITION:
            _write_xpartition(ds_coarsened, output_path, config)
        else:
            _write_eager(ds_coarsened, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Coarsen Script")
    parser.add_argument("config_yaml", type=str, help="Path to configuration yaml file")
    parser.add_argument("run", type=int, help="Run number")
    parser.add_argument(
        "--dry-run", action="store_true", help="If set, do not write output zarr files."
    )
    args = parser.parse_args()
    with open(args.config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dacite.from_dict(data_class=Config, data=config_dict)
    main(config, run=args.run, dry_run=args.dry_run)
