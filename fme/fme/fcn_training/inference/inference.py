import dataclasses
import os
import time
from typing import Union
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.dicts import to_flat_dict
from fme.fcn_training.utils.data_loader_multifiles import get_data_loader
from fme.fcn_training.utils.data_loader_params import DataLoaderParams
import argparse

import torch
import logging
from fme.fcn_training.utils import logging_utils

from fme.core import SingleModuleStepper
from fme.core.wandb import WandB

import fme
from fme.fcn_training.train_config import LoggingConfig
from fme.fcn_training.inference.data_writer import DataWriter, NullDataWriter
import dacite
import yaml

wandb = WandB.get_instance()


def load_stepper(checkpoint_file: str) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(
        checkpoint["stepper"], load_optimizer=False
    )
    return stepper


@dataclasses.dataclass
class InferenceConfig:
    """
    Attributes:
        experiment_dir: Directory to save results to
        n_forward_steps: Number of steps to run the model forward for
        checkpoint_path: Path to stepper checkpoint to load
        logging: configuration for logging
        validation_data: configuration for validation data
        log_video: Whether to log videos of the predictions
        save_prediction_files: Whether to save the predictions as a netcdf file
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    validation_data: DataLoaderParams
    log_video: bool = True
    save_prediction_files: bool = True

    def __post_init__(self):
        if self.save_prediction_files and not self.log_video:
            raise ValueError(
                "save_prediction_files is only valid when log_video is True, "
                "otherwise there are no predictions to save"
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self):
        self.logging.configure_wandb(
            config=to_flat_dict(dataclasses.asdict(self)), resume=False
        )

    def load_stepper(self) -> SingleModuleStepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path)


def get_n_samples(data_loader):
    n_samples = 0
    for data in data_loader:
        n_samples += next(data.values().__iter__()).shape[0]
    return n_samples


def main(
    yaml_config: str,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
        with open(os.path.join(data["experiment_dir"], "config.yaml"), "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    config.configure_logging(log_filename="inference_out.log")
    config.configure_wandb()

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging_utils.log_beaker_url()

    stepper = config.load_stepper()
    logging.info("Loading inference data")
    data_requirements = stepper.get_data_requirements(
        n_forward_steps=config.n_forward_steps
    )
    valid_data_loader, valid_dataset = get_data_loader(
        config.validation_data,
        requirements=data_requirements,
        train=False,
    )

    aggregator = InferenceAggregator(
        valid_dataset.area_weights.to(fme.get_device()),
        record_step_20=config.n_forward_steps >= 20,
        log_video=config.log_video,
    )
    n_samples = get_n_samples(valid_data_loader)
    if config.save_prediction_files:
        writer: Union[DataWriter, NullDataWriter] = DataWriter(
            filename=os.path.join(
                config.experiment_dir,
                "autoregressive_predictions.nc",
            ),
            n_samples=n_samples,
            metadata=valid_dataset.metadata,
        )
    else:
        writer = NullDataWriter()
    with torch.no_grad():
        i_sample = 0
        i_time = 0
        for data in valid_data_loader:
            _, gen_data, _, _ = stepper.run_on_batch(
                data,
                train=False,
                n_forward_steps=config.n_forward_steps,
                aggregator=aggregator,
            )
            writer.append_batch(
                target=data,
                prediction=gen_data,
                start_timestep=i_time,
                start_sample=i_sample,
            )
            i_sample += next(data.values().__iter__()).shape[0]
    step_logs = aggregator.get_inference_logs(label="inference")
    for log in step_logs:
        wandb.log(log)
        # wandb.log cannot be called more than "a few times per second"
        time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
