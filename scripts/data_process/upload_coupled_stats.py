import dataclasses
import logging
import os
import sys
import tempfile

import click

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from create_coupled_datasets import STATS_CATEGORIES, CreateCoupledDatasetsConfig
from fs_utils import path_exists
from upload_stats import STATS_FILENAMES, copy


@dataclasses.dataclass
class UploadSpec:
    """
    The Beaker dataset to create from a coupled config's merged stats.

    Attributes:
        beaker_dataset: Name of the Beaker dataset to create.
        merged_stats_directory: Directory holding one merged stats subdirectory
            per category.
        description: Description to attach to the Beaker dataset.
    """

    beaker_dataset: str
    merged_stats_directory: str
    description: str


def _describe(config: CreateCoupledDatasetsConfig) -> str:
    start = config.stats.start_date or "start of run"
    end = config.stats.end_date or "end of run"
    return (
        f"Coefficients for normalization for coupled datasets "
        f"{config.version}-{config.family_name} in {config.output_directory}. "
        f"Computed from {start} to {end}."
    )


def _upload_spec(config: CreateCoupledDatasetsConfig) -> UploadSpec | None:
    if config.stats.beaker_dataset is None:
        logging.warning(
            "No stats.beaker_dataset configured; stats at "
            f"{config.merged_stats_directory} will not be uploaded."
        )
        return None
    return UploadSpec(
        beaker_dataset=config.stats.beaker_dataset,
        merged_stats_directory=config.merged_stats_directory,
        description=_describe(config),
    )


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

    categories = [
        category
        for category in STATS_CATEGORIES
        if path_exists(os.path.join(spec.merged_stats_directory, category))
    ]
    if not categories:
        raise FileNotFoundError(
            f"No merged stats subdirectory of {STATS_CATEGORIES} found in "
            f"{spec.merged_stats_directory}."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        for category in categories:
            os.makedirs(os.path.join(tmpdir, category))
            for filename in STATS_FILENAMES:
                copy(
                    os.path.join(spec.merged_stats_directory, category, filename),
                    os.path.join(tmpdir, category, filename),
                )
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
    Upload the merged coupled stats written by create_coupled_datasets.py to
    Beaker, one subdirectory per category.

    Arguments:
    config_yaml -- Path to the create_coupled_datasets.py configuration file.
    """
    logging.basicConfig(level=logging.INFO)

    config = CreateCoupledDatasetsConfig.from_file(config_yaml)
    spec = _upload_spec(config)
    if spec is None:
        return

    # imported here so we don't need to install beaker for the tests
    from beaker import Beaker

    beaker_client = Beaker.from_env()
    _upload(beaker_client, spec)


if __name__ == "__main__":
    main()
