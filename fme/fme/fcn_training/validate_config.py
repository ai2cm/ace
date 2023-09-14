import argparse

import dacite
import yaml

from fme.fcn_training.inference.inference import InferenceConfig
from fme.fcn_training.train_config import TrainConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the train or inference config file."
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help=(
            "Indicates the path is an inference config, "
            "otherwise assume training config."
        ),
    )
    args = parser.parse_args()

    with open(args.path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    if args.inference:
        dacite.from_dict(
            data_class=InferenceConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
    else:
        dacite.from_dict(
            data_class=TrainConfig,
            data=config_data,
            config=dacite.Config(strict=True),
        )
