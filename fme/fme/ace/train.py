import argparse
import os

import dacite
import yaml

from fme.ace._train import run_train_from_config
from fme.ace.train_config import TrainConfig


def main(yaml_config: str):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir, exist_ok=True)
    with open(os.path.join(train_config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    run_train_from_config(train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(yaml_config=args.yaml_config)
