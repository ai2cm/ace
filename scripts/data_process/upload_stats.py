import dataclasses
import shutil
import tempfile
from typing import Dict, List, Optional

import click
import dacite
import fsspec
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
    beaker_dataset: str
    exclude_runs: List[str] = dataclasses.field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclasses.dataclass
class Config:
    runs: Dict[str, str]
    data_output_directory: str
    stats: StatsConfig


@click.command()
@click.argument("config_yaml", type=str)
def main(config_yaml: str):
    """
    Combine statistics for the data processing pipeline.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    """
    # imported here so we don't need to install beaker for the tests
    from beaker import Beaker

    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)

    stats_combined_dir = config.stats.output_directory + "/combined/"
    beaker = Beaker.from_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in (
            "centering.nc",
            "scaling-full-field.nc",
            "scaling-residual.nc",
            "time-mean.nc",
        ):
            copy(stats_combined_dir + filename, tmpdir + "/" + filename)
        runs = [run for run in config.runs if run not in config.stats.exclude_runs]
        run_names = ", ".join(runs)
        if config.stats.start_date is None:
            start = "start of run"
        else:
            start = config.stats.start_date
        if config.stats.end_date is None:
            end = "end of run"
        else:
            end = config.stats.end_date
        beaker.dataset.create(
            config.stats.beaker_dataset,
            tmpdir,
            workspace="ai2/ace",
            description=(
                "Coefficients for normalization for data "
                f"{config.data_output_directory} runs {run_names}. "
                f"Computed from {start} to {end}."
            ),
        )


if __name__ == "__main__":
    main()
