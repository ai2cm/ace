import argparse
from typing import TypeVar

import dacite
import yaml

from .evaluator import EvaluatorConfig
from .inference import InferenceConfig
from .predict import DownscalerConfig
from .train import TrainerConfig

T = TypeVar("T")

CONFIG_CHOICES = ["train", "inference", "evaluator", "predict"]

CONFIG_CLASSES: dict[str, type] = {
    "train": TrainerConfig,
    "inference": InferenceConfig,
    "evaluator": EvaluatorConfig,
    "predict": DownscalerConfig,
}


def validate_config(config_dict: dict, data_class: type[T]) -> T:
    return dacite.from_dict(
        data_class=data_class,
        data=config_dict,
        config=dacite.Config(strict=True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a downscaling entrypoint config from a YAML file."
    )
    parser.add_argument(
        "config_type",
        type=str,
        help=(
            "Kind of configuration to validate. " f"Expected one of: {CONFIG_CHOICES}."
        ),
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the configuration file (YAML).",
    )
    args = parser.parse_args()

    if args.config_type not in CONFIG_CLASSES:
        raise ValueError(
            f"Unrecognized config type: '{args.config_type}'. "
            f"Expected one of: {CONFIG_CHOICES}."
        )

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
        validate_config(config_dict, CONFIG_CLASSES[args.config_type])
        print("Configuration is valid.")
