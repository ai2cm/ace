import pathlib

import dacite
import yaml

from fme.ace.inference.evaluator import InferenceEvaluatorConfig
from fme.ace.train.train_config import TrainConfig

EXAMPLES_DIRECTORY = pathlib.PurePath(__file__).parent


def test_train_config_is_valid():
    with open(EXAMPLES_DIRECTORY / "ace-train-config.yaml") as f:
        data = yaml.safe_load(f)

    dacite.from_dict(
        data_class=TrainConfig, data=data, config=dacite.Config(strict=True)
    )


def test_evaluator_config_is_valid():
    with open(EXAMPLES_DIRECTORY / "ace-evaluator-config.yaml") as f:
        data = yaml.safe_load(f)
    dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
