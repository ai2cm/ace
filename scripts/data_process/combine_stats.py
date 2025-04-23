import dataclasses
import shutil
import tempfile
from typing import Dict, List

import click
import dacite
import fsspec
import xarray as xr
import yaml


def copy(source: str, destination: str):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)


@dataclasses.dataclass
class StatsConfig:
    output_directory: str
    exclude_runs: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Config:
    runs: Dict[str, str]
    stats: StatsConfig


def add_history_attrs(ds, config_filename: str, stats_output_dir: str):
    ds.attrs["history"] = (
        "Created by full-model/fv3gfs_data_process/combine_stats.py from "
        f"configuration file {config_filename} using inputs at {stats_output_dir}."
    )


def open_datasets(roots: List[str], filename: str) -> List[xr.Dataset]:
    datasets = []
    for root in roots:
        with fsspec.open(root + filename) as file:
            ds = xr.open_dataset(file, decode_timedelta=False).load()
        datasets.append(ds)
    return datasets


@click.command()
@click.argument("config_yaml", type=str)
def main(config_yaml: str):
    """
    Combine statistics for the data processing pipeline.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    """
    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)
    xr.set_options(keep_attrs=True)

    stats_roots = [
        config.stats.output_directory + "/" + run + "/"
        for run in config.runs.keys()
        if run not in config.stats.exclude_runs
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in (
            "centering.nc",
            "scaling-residual.nc",
            "time-mean.nc",
        ):
            datasets = open_datasets(stats_roots, filename)
            if len(datasets) > 1:
                samples = xr.DataArray(
                    [float(ds.attrs["input_samples"]) for ds in datasets], dims=["run"]
                )
                combined = xr.concat(datasets, dim="run")
                if filename.startswith("scaling"):
                    # standard deviations are averaged as variances
                    average = (combined**2).weighted(samples).mean(dim="run") ** 0.5
                else:
                    average = combined.weighted(samples).mean(dim="run")
            else:
                average = datasets[0]

            add_history_attrs(average, config_yaml, config.stats.output_directory)
            average.to_netcdf(tmpdir + "/" + filename)
            copy(
                tmpdir + "/" + filename,
                config.stats.output_directory + "/combined/" + filename,
            )
        # for scaling-full-field.nc, we also need to account for the means.
        full_field_filename = "scaling-full-field.nc"
        centering_filename = "centering.nc"
        full_field_datasets = open_datasets(stats_roots, full_field_filename)
        if len(full_field_datasets) > 1:
            samples = xr.DataArray(
                [float(ds.attrs["input_samples"]) for ds in full_field_datasets],
                dims=["run"],
            )
            centering_datasets = open_datasets(stats_roots, centering_filename)
            average = get_combined_stats(
                full_field_datasets, centering_datasets, samples
            )
        else:
            average = full_field_datasets[0]

        add_history_attrs(average, config_yaml, config.stats.output_directory)
        average.to_netcdf(tmpdir + "/" + full_field_filename)
        copy(
            tmpdir + "/" + full_field_filename,
            config.stats.output_directory + "/combined/" + full_field_filename,
        )


def get_combined_stats(full_field_datasets, centering_datasets, samples):
    combined_scaling = xr.concat(full_field_datasets, dim="run")
    combined_centering = xr.concat(centering_datasets, dim="run")
    average_centering = combined_centering.weighted(samples).mean(dim="run")
    centering_variance = (combined_centering - average_centering) ** 2
    total_variance = centering_variance + combined_scaling**2
    average = total_variance.weighted(samples).mean(dim="run") ** 0.5
    return average


if __name__ == "__main__":
    main()
