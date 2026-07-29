"""The example configs' data blocks must parse.

``fme/translate/examples/`` holds the intended config surface for the whole PR
series, reviewed before the code existed. These tests hold this PR's blocks —
``train_data:`` and ``validation_data:`` — to what those examples say.
"""

import pathlib

import dacite
import pytest
import yaml

from fme.translate.data.config import TranslateDataLoaderConfig

EXAMPLES = pathlib.Path(__file__).parents[1] / "examples"


def _load(example: str, block: str) -> TranslateDataLoaderConfig:
    with open(EXAMPLES / example) as f:
        config = yaml.safe_load(f)
    return dacite.from_dict(
        data_class=TranslateDataLoaderConfig,
        data=config[block],
        config=dacite.Config(strict=True),
    )


@pytest.mark.parametrize(
    "example, block",
    [
        (example, block)
        for example in ["multi-resolution-latent.yaml", "transfer-learning.yaml"]
        for block in ["train_data", "validation_data"]
    ],
)
def test_example_data_block_parses(example, block):
    config = _load(example, block)
    assert len(config.streams) >= 2
    assert config.batch_size == 16


def test_multi_resolution_streams_are_explicitly_named():
    config = _load("multi-resolution-latent.yaml", "train_data")
    assert list(config.stream_configs) == ["era5_1deg", "era5_2deg", "era5_4deg"]
    assert [stream.domain for stream in config.streams] == [
        "atmos_1deg",
        "atmos_2deg",
        "atmos_4deg",
    ]


def test_transfer_learning_stream_names_default_to_their_domains():
    config = _load("transfer-learning.yaml", "train_data")
    assert all(stream.name is None for stream in config.streams)
    assert list(config.stream_configs) == ["era5", "shield_c96"]
