import argparse
import dataclasses

import dacite
import xarray as xr
import yaml


@dataclasses.dataclass
class PathPair:
    input: str
    output: str


@dataclasses.dataclass
class Config:
    """
    Configuration for time coarsening of a dataset.

    Attributes:
        input_path: Path to the input dataset.
        output_path: Path to save the coarsened dataset as a zarr store.
        coarsen_factor: Factor by which to coarsen the time dimension.
        snapshot_names: List of snapshot variable names to coarsen. These will be
            coarsened by skipping each coarsen_factor times.
        window_names: List of window variable names to coarsen. These will be
            coarsened by averaging over each coarsen_factor times.
        constant_prefixes: List of prefixes for constant data variables to copy without
            modification. Raises an exception if any of these have a "time" dimension.
    """

    paths: list[PathPair]
    coarsen_factor: int
    snapshot_names: list[str]
    window_names: list[str]
    constant_prefixes: list[str] = dataclasses.field(
        default_factory=lambda: ["ak_", "bk_"]
    )


def main(config: Config):
    for paths in config.paths:
        process_path_pair(paths, config)


def process_path_pair(pair: PathPair, config: Config):
    ds = xr.open_dataset(pair.input)
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
        time=slice(config.coarsen_factor - 1, None, config.coarsen_factor)
    )
    ds_window = (
        ds[config.window_names]
        .coarsen(time=config.coarsen_factor, boundary="trim")
        .mean()
        .drop("time")
    )  # use time of snapshots
    # ds_coarsened = ds_snapshot#xr.merge([ds_snapshot, ds_window, ds_coords])
    ds_coarsened = xr.merge([ds_snapshot, ds_window, ds_constants])
    ds_coarsened.to_zarr(pair.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Coarsen Script")
    parser.add_argument(
        "config_yaml", type=str, required=True, help="Path to configuration yaml file"
    )
    args = parser.parse_args()
    with open(args.config_yaml, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=Config, data=config_dict, config=dacite.Config(strict=True)
    )
    main(config)
