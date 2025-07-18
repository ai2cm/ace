import argparse
import logging

import dacite
import dacite.exceptions
import yaml

from fme.ace.inference.evaluator import InferenceEvaluatorConfig
from fme.ace.inference.inference import InferenceConfig
from fme.ace.stepper import SingleModuleStepperConfig
from fme.ace.train.train_config import TrainConfig
from fme.core.config import update_dict_with_dotlist

CONFIG_CHOICES = ["train", "inference", "evaluator"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the train or inference config file."
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help=(
            "Deprecated, use --config_type evaluator to validate an evaluator config."
        ),
    )
    parser.add_argument(
        "--config_type",
        type=str,
        choices=CONFIG_CHOICES,
        default="train",
        help=("Indicates the kind of configuration being validated."),
    )
    parser.add_argument(
        "--override",
        nargs="*",
        help="A dotlist of key=value pairs to override the config. "
        "For example, --override a.b=1 c=2, where a dot indicates nesting.",
    )
    args = parser.parse_args()

    if args.inference:
        logging.warning(
            "The --inference flag is deprecated. "
            "Use --config_type evaluator to validate an evaluator config."
        )
        config_type = "evaluator"
    else:
        config_type = args.config_type

    with open(args.path) as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    config_data = update_dict_with_dotlist(config_data, args.override)
    if config_type == "evaluator":
        dacite.from_dict(
            data_class=InferenceEvaluatorConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
    elif config_type == "inference":
        dacite.from_dict(
            data_class=InferenceConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
    elif config_type == "train":
        try:
            dacite.from_dict(
                data_class=TrainConfig,
                data=config_data,
                config=dacite.Config(strict=True),
            )
        except dacite.exceptions.UnionMatchError as err:
            if "checkpoint_path" not in config_data["stepper"]:
                dacite.from_dict(
                    data_class=SingleModuleStepperConfig,
                    data=config_data["stepper"],
                    config=dacite.Config(strict=True),
                )
            # if there was no issue for SingleModuleStepperConfig, raise original error
            raise err
    else:
        raise ValueError(
            f"Invalid config type: {config_type}, expected one of {CONFIG_CHOICES}"
        )
