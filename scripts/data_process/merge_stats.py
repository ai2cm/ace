import argparse
import dataclasses
import logging
import os
import tempfile

import dacite
import fsspec
import xarray as xr
import yaml
from get_stats import copy

STATS_NC_FILE_NAMES = [
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
]


@dataclasses.dataclass
class RenameStatConfig:
    """
    Parameters:
        data_var: The variable that provides the statistic.
        new_name: The new name for the statistic.
        drop: Whether or not to drop data_var.
    """

    data_var: str
    new_name: str
    drop: bool = False

    def rename_var(self, stats: xr.Dataset) -> xr.Dataset:
        print(f"Renaming {self.data_var} to {self.new_name}")
        stats[self.new_name] = stats[self.data_var].copy()
        if self.drop:
            return stats.drop_vars([self.data_var])
        return stats


@dataclasses.dataclass
class MergeStatsConfig:
    input_directories: list[str]
    output_directory: str
    rename: list[RenameStatConfig] = dataclasses.field(default_factory=list)
    latitude_dim: str = "lat"
    longitude_dim: str = "lon"
    exclude_names: list[str] = dataclasses.field(default_factory=list)

    def open_input_datasets(self, fname: str):
        datasets = []
        for path in self.input_directories:
            with fsspec.open(os.path.join(path, fname)) as file:
                ds = xr.open_dataset(file, decode_timedelta=False).load()
            datasets.append(ds)
        return datasets


def merge_stats(config: MergeStatsConfig):
    """Merge statistics from multiple input directories into a single output directory.

    Args:
        config: MergeStatsConfig object containing input directories, output directory,
                and optional rename configurations.
    """
    stats_dir = config.output_directory

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    for fname in STATS_NC_FILE_NAMES:
        logging.info(f"Combining {fname} stats datasets")
        datasets = config.open_input_datasets(fname)
        combined_stats = {}
        overlapping_names = set()
        for i, ds in enumerate(datasets):
            latdim = config.latitude_dim
            londim = config.longitude_dim
            if config.latitude_dim in ds.dims and i > 0:
                ds = ds.assign_coords({latdim: datasets[0][latdim]})
            if config.longitude_dim in ds.dims and i > 0:
                ds = ds.assign_coords({londim: datasets[0][londim]})
            for name in ds.data_vars:
                if name in combined_stats:
                    overlapping_names.add(name)
                    continue
                combined_stats[name] = ds[name]
                combined_stats[name].attrs["source"] = ds.encoding["source"]
        if len(overlapping_names) == 0:
            logging.info("No overlapping names found")
        else:
            logging.warning("Overlapping stats found in input directories")
            for name in overlapping_names:
                source = combined_stats[name].attrs["source"]
                logging.warning(f"{name} chosen from {source}")
        stats = xr.Dataset(combined_stats)
        for rename_config in config.rename:
            stats = rename_config.rename_var(stats)
        stats.attrs["input_samples"] = datasets[0].attrs["input_samples"]
        logging.info(
            "Using first stats dataset's 'input_samples' attribute of "
            f"{stats.attrs['input_samples']} for the merged stats."
        )
        print(stats.load())
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, fname)
            stats.to_netcdf(local_path)
            destination_path = os.path.join(stats_dir, fname)
            copy(local_path, destination_path)


def main():
    """Main CLI entry point for merging statistics."""
    parser = argparse.ArgumentParser(
        description="Merge statistics from multiple input directories."
    )
    parser.add_argument("yaml_path", type=str, help="Path to the config file.")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    config = dacite.from_dict(
        data_class=MergeStatsConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )

    merge_stats(config)


if __name__ == "__main__":
    main()
