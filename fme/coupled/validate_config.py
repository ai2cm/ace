import argparse

import dacite
import dacite.exceptions
import yaml

from fme.core.config import update_dict_with_dotlist
from fme.coupled.inference.evaluator import InferenceEvaluatorConfig
from fme.coupled.inference.inference import InferenceConfig
from fme.coupled.train.train_config import TrainConfig

CONFIG_CHOICES = ["train", "inference", "evaluator"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the coupled train or inference config file."
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

    with open(args.path) as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    config_data = update_dict_with_dotlist(config_data, args.override)
    try:
        if args.config_type == "evaluator":
            dacite.from_dict(
                data_class=InferenceEvaluatorConfig,
                data=config_data,
                config=dacite.Config(strict=True),
            )
        elif args.config_type == "inference":
            dacite.from_dict(
                data_class=InferenceConfig,
                data=config_data,
                config=dacite.Config(strict=True),
            )
        elif args.config_type == "train":
            dacite.from_dict(
                data_class=TrainConfig,
                data=config_data,
                config=dacite.Config(strict=True),
            )
    except Exception as err:
        raise ValueError(
            f"Building the coupled {args.config_type} config from {args.path} failed"
        ) from err
