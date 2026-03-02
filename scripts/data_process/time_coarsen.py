import argparse
import dataclasses
import logging
import os
import sys
from typing import Mapping

import dacite
import xarray as xr
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_stats import StatsConfig


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
    """

    factor: int
    data_output_directory: str
    stats_output_directory: str
    snapshot_names: list[str]
    window_names: list[str]
    constant_prefixes: list[str]


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
    if run_name in config.stats.exclude_runs:
        logging.info(f"Skipping run {run_name}")
        return


def write_zarr(
    ds: xr.Dataset, original_attributes: dict | None, history_entry: str, path: str
) -> None:
    if len(ds.attrs) > 0 and original_attributes is not None:
        raise ValueError(
            "Dataset already has attributes, cannot overwrite original "
            "attributes. Instead copy over any original attributes you want "
            "onto ds.attrs, and then pass original_attributes=None."
        )
    if original_attributes is not None:
        ds.attrs = original_attributes
    attributes = ds.attrs
    if "history" in attributes:
        attributes["history"] = attributes["history"] + " " + history_entry
    else:
        attributes["history"] = history_entry
    ds.to_zarr(path, mode="w", zarr_version=3)


def process_path_pair(
    input_path: str, output_path: str, config: TimeCoarsenConfig, dry_run: bool
):
    logging.info(f"Processing input: {input_path} to output: {output_path}")
    if os.path.exists(output_path):
        logging.warning(f"Output path {output_path} already exists. Skipping.")
        return
    ds = xr.open_dataset(input_path)
    constant_names = [
        name
        for name in ds.data_vars
        if any(name.startswith(prefix) for prefix in config.constant_prefixes)
    ]
    assert set(constant_names).intersection(set(config.snapshot_names)) == set()
    assert set(constant_names).intersection(set(config.window_names)) == set()
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
    attributes = ds.attrs.copy()
    attributes["snapshot_names"] = config.snapshot_names
    attributes["window_names"] = config.window_names
    attributes["constant_prefixes"] = config.constant_prefixes
    attributes["coarsen_factor"] = config.factor
    history_entry = (
        f"Dataset coarsened by a factor of {config.factor} "
        "by scripts/data_process/time_coarsen.py."
    )
    ds_coarsened.attrs = None  # clear attributes, we'll add them back in write_zarr
    if not dry_run:
        write_zarr(
            ds_coarsened,
            original_attributes=attributes,
            history_entry=history_entry,
            path=output_path,
        )


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
