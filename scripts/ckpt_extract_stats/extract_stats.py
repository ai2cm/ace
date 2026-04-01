"""Extract normalization statistics from an ACE Trainer checkpoint as netCDF files.

Handles both modern and legacy checkpoint formats via StepperConfig.from_stepper_state.
Produces files named after the normalization config keys:
  - network-means.nc, network-stds.nc (always present)
  - residual-means.nc, residual-stds.nc (if residual normalization is present)
  - loss-means.nc, loss-stds.nc (if loss normalization is present)
"""

import logging
import os
import pathlib
from typing import Any

import click
import xarray as xr

logger = logging.getLogger(__name__)

NORMALIZATION_KEYS = ("network", "residual", "loss")


def _find_normalization(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively search a stepper config dict for the normalization config.

    The normalization config is identified as a dict containing a "network" key
    whose value is a dict with "means" and "stds" keys.
    """
    if (
        "normalization" in config
        and isinstance(config["normalization"], dict)
        and "network" in config["normalization"]
    ):
        network = config["normalization"]["network"]
        if isinstance(network, dict) and "means" in network and "stds" in network:
            return config["normalization"]

    for value in config.values():
        if isinstance(value, dict):
            try:
                return _find_normalization(value)
            except ValueError:
                continue

    raise ValueError(
        "Could not find normalization config with network means/stds "
        f"in config keys: {list(config.keys())}"
    )


def _dict_to_dataset(data: dict[str, float]) -> xr.Dataset:
    """Convert a {variable_name: scalar_value} dict to an xarray Dataset."""
    return xr.Dataset({name: xr.DataArray(value) for name, value in data.items()})


def extract_stats(checkpoint_path: str | pathlib.Path) -> dict[str, xr.Dataset]:
    """Extract normalization stats from a checkpoint as xarray Datasets.

    Uses StepperConfig.from_stepper_state to parse both legacy and modern
    checkpoint formats without building the full stepper (which would require
    distributed context, GPU, etc.).

    Returns:
        Dict mapping filename to Dataset, e.g.
        {"network-means.nc": Dataset, "network-stds.nc": Dataset, ...}
    """
    import dataclasses

    import torch

    from fme.ace.stepper.single_module import StepperConfig

    checkpoint = torch.load(
        str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False
    )
    stepper_state = checkpoint["stepper"]
    config = StepperConfig.from_stepper_state(stepper_state)
    config_dict = dataclasses.asdict(config)
    normalization = _find_normalization(config_dict)

    result: dict[str, xr.Dataset] = {}
    for key in NORMALIZATION_KEYS:
        norm = normalization.get(key)
        if norm is None:
            continue
        means = norm.get("means", {})
        stds = norm.get("stds", {})
        if means:
            filename = f"{key}-means.nc"
            result[filename] = _dict_to_dataset(means)
            logger.info(f"Extracted {filename} with {len(means)} variables")
        if stds:
            filename = f"{key}-stds.nc"
            result[filename] = _dict_to_dataset(stds)
            logger.info(f"Extracted {filename} with {len(stds)} variables")

    return result


def write_stats(stats: dict[str, xr.Dataset], output_dir: str | pathlib.Path) -> None:
    """Write extracted stats datasets to netCDF files in output_dir."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, ds in stats.items():
        path = output_dir / filename
        ds.to_netcdf(path)
        logger.info(f"Wrote {path}")


def upload_to_beaker(
    output_dir: str | pathlib.Path,
    dataset_name: str,
    description: str = "",
) -> None:
    """Upload the stats directory to a Beaker dataset."""
    import beaker as beaker_module
    from beaker import Beaker

    beaker_client = Beaker.from_env()
    try:
        beaker_client.dataset.get(dataset_name)
        logger.info(f"Beaker dataset '{dataset_name}' already exists. Skipping upload.")
        return
    except beaker_module.exceptions.BeakerDatasetNotFound:
        pass

    beaker_client.dataset.create(
        dataset_name,
        str(output_dir),
        workspace="ai2/ace",
        description=description,
    )
    logger.info(f"Uploaded stats to Beaker dataset '{dataset_name}'")


@click.command()
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Directory to write the extracted netCDF stats files.",
)
@click.option(
    "--beaker-dataset",
    default=None,
    type=str,
    help="If provided, upload extracted stats to this Beaker dataset name.",
)
def main(checkpoint_path: str, output_dir: str, beaker_dataset: str | None):
    """Extract normalization statistics from a Trainer checkpoint.

    CHECKPOINT_PATH is the path to a .tar checkpoint file.
    """
    logging.basicConfig(level=logging.INFO)

    stats = extract_stats(checkpoint_path)
    write_stats(stats, output_dir)

    filenames = ", ".join(stats.keys())
    logger.info(f"Extracted stats files: {filenames}")

    if beaker_dataset is not None:
        description = (
            f"Normalization stats extracted from checkpoint "
            f"{os.path.basename(checkpoint_path)}. "
            f"Files: {filenames}."
        )
        upload_to_beaker(output_dir, beaker_dataset, description=description)


if __name__ == "__main__":
    main()
