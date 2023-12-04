import argparse

import dacite
import yaml

from fme.fcn_training.inference.inference import InferenceConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the train or inference config file."
    )
    args = parser.parse_args()

    with open(args.path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.CLoader)

    dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
