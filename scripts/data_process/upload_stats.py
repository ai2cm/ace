import dataclasses
import logging
import os
import shutil
import sys
import tempfile

import click
import dacite
import fsspec
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_stats import StatsConfig, TimeCoarsenConfig

STATS_FILENAMES = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)


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
class Config:
    runs: dict[str, str]
    data_output_directory: str
    stats: StatsConfig
    time_coarsen: TimeCoarsenConfig | None = None

    def __post_init__(self):
        if self.stats.beaker_dataset is None and (
            self.time_coarsen is None or self.time_coarsen.beaker_dataset is None
        ):
            raise ValueError(
                "No Beaker dataset to upload. Set stats.beaker_dataset, or set "
                "time_coarsen.beaker_dataset to upload only the time-coarsened stats."
            )


@dataclasses.dataclass
class UploadSpec:
    """
    A single Beaker dataset to create from a directory of combined stats.

    Attributes:
        beaker_dataset: Name of the Beaker dataset to create.
        combined_directory: Directory holding the combined stats netCDFs.
        description: Description to attach to the Beaker dataset.
    """

    beaker_dataset: str
    combined_directory: str
    description: str


def _describe(
    config: Config, data_directory: str, coarsen_factor: int | None = None
) -> str:
    runs = [run for run in config.runs if run not in config.stats.exclude_runs]
    run_names = ", ".join(runs)
    start = config.stats.start_date or "start of run"
    end = config.stats.end_date or "end of run"
    description = (
        f"Coefficients for normalization for data {data_directory} "
        f"runs {run_names}. Computed from {start} to {end}."
    )
    if coarsen_factor is not None:
        description += f" Time coarsened by a factor of {coarsen_factor}."
    return description


def _upload_specs(config: Config) -> list[UploadSpec]:
    specs = []
    if config.stats.beaker_dataset is None:
        logging.warning(
            "No stats.beaker_dataset configured; stats at "
            f"{config.stats.output_directory} will not be uploaded."
        )
    else:
        specs.append(
            UploadSpec(
                beaker_dataset=config.stats.beaker_dataset,
                combined_directory=config.stats.output_directory + "/combined/",
                description=_describe(config, config.data_output_directory),
            )
        )
    if config.time_coarsen is not None:
        if config.time_coarsen.beaker_dataset is None:
            logging.warning(
                "No time_coarsen.beaker_dataset configured; time coarsened stats at "
                f"{config.time_coarsen.stats_output_directory} will not be uploaded."
            )
        else:
            specs.append(
                UploadSpec(
                    beaker_dataset=config.time_coarsen.beaker_dataset,
                    combined_directory=(
                        config.time_coarsen.stats_output_directory + "/combined/"
                    ),
                    description=_describe(
                        config,
                        config.time_coarsen.data_output_directory,
                        coarsen_factor=config.time_coarsen.factor,
                    ),
                )
            )
    return specs


def _upload(beaker_client, spec: UploadSpec):
    import beaker as beaker_module

    try:
        beaker_client.dataset.get(spec.beaker_dataset)
        logging.info(
            f"Beaker dataset '{spec.beaker_dataset}' already exists. Skipping."
        )
        return
    except beaker_module.exceptions.BeakerDatasetNotFound:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in STATS_FILENAMES:
            copy(spec.combined_directory + filename, tmpdir + "/" + filename)
        beaker_client.dataset.create(
            spec.beaker_dataset,
            tmpdir,
            workspace="ai2/ace",
            description=spec.description,
        )


@click.command()
@click.argument("config_yaml", type=str)
def main(config_yaml: str):
    """
    Upload normalization statistics for the data processing pipeline to Beaker.

    Arguments:
    config_yaml -- Path to the configuration file for the data processing pipeline.
    """
    logging.basicConfig(level=logging.INFO)

    with open(config_yaml, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)
    config = dacite.from_dict(data_class=Config, data=config_data)

    specs = _upload_specs(config)

    # imported here so we don't need to install beaker for the tests
    from beaker import Beaker

    beaker_client = Beaker.from_env()
    for spec in specs:
        _upload(beaker_client, spec)


if __name__ == "__main__":
    main()
