import argparse
import dataclasses
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import dacite
import torch
import yaml

import fme
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference.main import InferenceAggregator
from fme.core.data_loading.data_typing import GriddedData, SigmaCoordinates
from fme.core.data_loading.getters import get_inference_data
from fme.core.data_loading.inference import InferenceDataLoaderParams
from fme.core.dicts import to_flat_dict
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.wandb import WandB
from fme.fcn_training.inference.data_writer import DataWriter, DataWriterConfig
from fme.fcn_training.inference.loop import run_dataset_inference, run_inference
from fme.fcn_training.train_config import LoggingConfig
from fme.fcn_training.utils import gcs_utils, logging_utils


def _load_stepper_config(checkpoint_file: str) -> SingleModuleStepperConfig:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    return SingleModuleStepperConfig.from_state(checkpoint["stepper"]["config"])


def _load_stepper(
    checkpoint_file: str,
    area: torch.Tensor,
    sigma_coordinates: SigmaCoordinates,
) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(
        checkpoint["stepper"], area=area, sigma_coordinates=sigma_coordinates
    )
    return stepper


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Attributes:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for. Must be divisble
            by forward_steps_in_memory.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: configuration for logging.
        validation_loader: Configuration for validation data.
        prediction_loader: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_extended_video_netcdfs: Whether to log videos of the predictions with
            statistical metrics as netcdf files.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
            time dimension.
        save_prediction_files: Whether to save the predictions as a netcdf file.
        save_raw_prediction_names: Names of variables to save in the predictions
             netcdf file. Ignored if save_prediction_files is False.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    validation_loader: InferenceDataLoaderParams
    prediction_loader: Optional[InferenceDataLoaderParams] = None
    log_video: bool = True
    log_extended_video: bool = False
    log_extended_video_netcdfs: Optional[bool] = None
    log_zonal_mean_images: bool = True
    save_prediction_files: Optional[bool] = None
    save_raw_prediction_names: Optional[Sequence[str]] = None
    forward_steps_in_memory: int = 1
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )
        deprecated_writer_attrs = {
            k: getattr(self, k)
            for k in [
                "log_extended_video_netcdfs",
                "save_prediction_files",
                "save_raw_prediction_names",
            ]
            if getattr(self, k) is not None
        }
        for k, v in deprecated_writer_attrs.items():
            warnings.warn(
                f"Inference configuration attribute `{k}` is deprecated. "
                f"Using its value `{v}`, but please use attribute `data_writer` "
                "instead."
            )
            setattr(self.data_writer, k, v)
        if (self.data_writer.time_coarsen is not None) and (
            self.forward_steps_in_memory % self.data_writer.time_coarsen.coarsen_factor
            != 0
        ):
            raise ValueError(
                "forward_steps_in_memory must be divisible by "
                f"time_coarsen.coarsen_factor. Got {self.forward_steps_in_memory} "
                f"and {self.data_writer.time_coarsen.coarsen_factor}."
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, env_vars: Optional[dict] = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        if "environment" in config:
            logging.warning("Not recording env vars since 'environment' is in config.")
        elif env_vars is not None:
            config["environment"] = env_vars
        self.logging.configure_wandb(config=config, resume=False, **kwargs)

    def clean_wandb(self):
        self.logging.clean_wandb(self.experiment_dir)

    def configure_gcs(self):
        self.logging.configure_gcs()

    def load_stepper(
        self, area: torch.Tensor, sigma_coordinates: SigmaCoordinates
    ) -> SingleModuleStepper:
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
            sigma_coordinates: The sigma coordinates of the model.
        """
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return _load_stepper(
            self.checkpoint_path,
            area=area,
            sigma_coordinates=sigma_coordinates,
        )

    def load_stepper_config(self) -> SingleModuleStepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return _load_stepper_config(self.checkpoint_path)

    def get_data_writer(self, data: GriddedData) -> DataWriter:
        return self.data_writer.build(
            experiment_dir=self.experiment_dir,
            n_samples=self.validation_loader.n_samples,
            n_timesteps=self.n_forward_steps + 1,
            metadata=data.metadata,
            coords=data.coords,
        )


def main(
    yaml_config: str,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)
    gcs_utils.authenticate()

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()

    start_time = time.time()
    stepper_config = config.load_stepper_config()
    logging.info("Loading inference data")
    data_requirements = stepper_config.get_data_requirements(
        n_forward_steps=config.n_forward_steps
    )

    data = get_inference_data(
        config.validation_loader,
        config.forward_steps_in_memory,
        data_requirements,
    )

    stepper = config.load_stepper(
        data.area_weights.to(fme.get_device()),
        sigma_coordinates=data.sigma_coordinates.to(fme.get_device()),
    )

    aggregator = InferenceAggregator(
        data.area_weights.to(fme.get_device()),
        sigma_coordinates=data.sigma_coordinates,
        record_step_20=config.n_forward_steps >= 20,
        log_video=config.log_video,
        enable_extended_videos=config.log_extended_video,
        log_zonal_mean_images=config.log_zonal_mean_images,
        n_timesteps=config.n_forward_steps + 1,
        metadata=data.metadata,
    )
    writer = config.get_data_writer(data)

    logging.info("Starting inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            config.forward_steps_in_memory,
            data_requirements,
        )

        timers = run_dataset_inference(
            aggregator=aggregator,
            normalizer=stepper.normalizer,
            prediction_data=prediction_data,
            target_data=data,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
            writer=writer,
        )
    else:
        timers = run_inference(
            aggregator=aggregator,
            writer=writer,
            stepper=stepper,
            data=data,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
        )

    duration = time.time() - start_time
    total_steps = config.n_forward_steps * config.validation_loader.n_samples
    total_steps_per_second = total_steps / duration
    logging.info(f"Inference duration: {duration:.2f} seconds")
    logging.info(f"Total steps per second: {total_steps_per_second:.2f} steps/second")

    step_logs = aggregator.get_inference_logs(label="inference")
    wandb = WandB.get_instance()
    if wandb.enabled:
        logging.info("Starting logging of metrics to wandb")
        duration_logs = {
            "duration_seconds": duration,
            "total_steps_per_second": total_steps_per_second,
        }
        wandb.log({**timers, **duration_logs}, step=0)
        for i, log in enumerate(step_logs):
            wandb.log(log, step=i)
            # wandb.log cannot be called more than "a few times per second"
            time.sleep(0.3)
    writer.flush()

    logging.info("Writing reduced metrics to disk in netcdf format.")
    for name, ds in aggregator.get_datasets(("time_mean", "zonal_mean")).items():
        coords = {k: v for k, v in data.coords.items() if k in ds.dims}
        ds = ds.assign_coords(coords)
        ds.to_netcdf(Path(config.experiment_dir) / f"{name}_diagnostics.nc")

    config.clean_wandb()

    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
