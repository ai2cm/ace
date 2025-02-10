import argparse

import dacite
import dacite.exceptions
import yaml

from fme.core.config import update_dict_with_dotlist
from fme.diffusion.train_config import TrainConfig

CONFIG_CHOICES = ["train"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the train or inference config file."
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

    config_type = args.config_type

    with open(args.path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    config_data = update_dict_with_dotlist(config_data, args.override)

    if config_type == "train":
        dacite.from_dict(
            data_class=TrainConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
    else:
        raise ValueError(
            f"Invalid config type: {config_type}, expected one of {CONFIG_CHOICES}"
        )
