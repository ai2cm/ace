import argparse

import dacite
import dacite.exceptions
import yaml

from fme.ace.inference.inference import InferenceConfig
from fme.ace.train_config import TrainConfig
from fme.core.stepper import SingleModuleStepperConfig

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
