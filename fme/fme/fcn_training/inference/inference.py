import dataclasses
import os
import time
from typing import Optional, Union
import argparse

import torch
import logging
import dacite
import yaml

import fme
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.dicts import to_flat_dict
from fme.fcn_training.utils import logging_utils
from fme.fcn_training.utils.data_loader_multifiles import get_data_loader
from fme.fcn_training.utils.data_loader_params import DataLoaderParams
from fme.fcn_training.train_config import LoggingConfig
from fme.core import SingleModuleStepper
from fme.core.wandb import WandB
from fme.fcn_training.inference.data_writer import DataWriter, NullDataWriter
from fme.fcn_training.inference.loop import run_inference

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
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for. Must be divisble
            by forward_steps_in_memory.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: configuration for logging.
        validation_data: configuration for validation data.
        log_video: Whether to log videos of the predictions.
        save_prediction_files: Whether to save the predictions as a netcdf file.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    validation_data: DataLoaderParams
    log_video: bool = True
    save_prediction_files: bool = True
    forward_steps_in_memory: int = 1

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
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

    def _get_data_loader(window_time_slice: Optional[slice] = None):
        """
        Helper function to keep the data loader configuration static,
        ensuring we get the same batches each time a data loader is
        retrieved, other than the choice of window_time_slice.
        """
        return get_data_loader(
            config.validation_data,
            requirements=data_requirements,
            train=False,
            window_time_slice=window_time_slice,
        )

    valid_data_loader, valid_dataset = _get_data_loader()

    output_netcdf_filename = os.path.join(
        config.experiment_dir,
        "autoregressive_predictions.nc",
    )
    aggregator = InferenceAggregator(
        valid_dataset.area_weights.to(fme.get_device()),
        record_step_20=config.n_forward_steps >= 20,
        log_video=config.log_video,
        n_timesteps=config.n_forward_steps + 1,
    )
    n_samples = get_n_samples(valid_data_loader)
    if config.save_prediction_files:
        writer: Union[DataWriter, NullDataWriter] = DataWriter(
            filename=output_netcdf_filename,
            n_samples=n_samples,
            metadata=valid_dataset.metadata,
        )
    else:
        writer = NullDataWriter()

    def data_loader_factory(window_time_slice: Optional[slice] = None):
        return _get_data_loader(window_time_slice=window_time_slice)[0]

    run_inference(
        aggregator=aggregator,
        writer=writer,
        stepper=stepper,
        data_loader_factory=data_loader_factory,
        n_forward_steps=config.n_forward_steps,
        forward_steps_in_memory=config.forward_steps_in_memory,
    )

    step_logs = aggregator.get_inference_logs(label="inference")
    for log in step_logs:
        wandb.log(log)
        # wandb.log cannot be called more than "a few times per second"
        time.sleep(0.3)
    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
