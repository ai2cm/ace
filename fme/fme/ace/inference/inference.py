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
from fme.ace.inference.data_writer import DataWriter, DataWriterConfig
from fme.ace.inference.loop import run_dataset_inference, run_inference
from fme.ace.train_config import LoggingConfig
from fme.ace.utils import logging_utils
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference import InferenceAggregatorConfig
from fme.core.data_loading.data_typing import GriddedData, SigmaCoordinates
from fme.core.data_loading.getters import get_inference_data
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.dicts import to_flat_dict
from fme.core.ocean import OceanConfig
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.wandb import WandB


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
        loader: Configuration for data to be used as initial conditions, forcing, and
            target in inference.
        prediction_loader: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference aggregator.
        ocean: Ocean configuration for running inference with a
            different one than what is used in training.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    loader: InferenceDataLoaderConfig
    prediction_loader: Optional[InferenceDataLoaderConfig] = None
    log_video: Optional[bool] = None
    log_extended_video: Optional[bool] = None
    log_zonal_mean_images: Optional[bool] = None
    forward_steps_in_memory: int = 1
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    monthly_reference_data: Optional[str] = None
    aggregator: InferenceAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceAggregatorConfig()
    )
    ocean: Optional[OceanConfig] = None

    def __post_init__(self):
        if self.n_forward_steps % self.forward_steps_in_memory != 0:
            raise ValueError(
                "n_forward_steps must be divisible by steps_in_memory, "
                f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
            )
        deprecated_aggregator_attrs = {
            k: getattr(self, k)
            for k in [
                "log_video",
                "log_extended_video",
                "log_zonal_mean_images",
                "monthly_reference_data",
            ]
            if getattr(self, k) is not None
        }
        for k, v in deprecated_aggregator_attrs.items():
            warnings.warn(
                f"Inference configuration attribute `{k}` is deprecated. "
                f"Using its value `{v}`, but please use attribute `aggregator` "
                "instead."
            )
            setattr(self.aggregator, k, v)
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
        stepper = _load_stepper(
            self.checkpoint_path,
            area=area,
            sigma_coordinates=sigma_coordinates,
        )
        if self.ocean is not None:
            logging.info(
                "Overriding training ocean configuration with the inference "
                "ocean config."
            )
            stepper.ocean = self.ocean
        return stepper

    def load_stepper_config(self) -> SingleModuleStepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return _load_stepper_config(self.checkpoint_path)

    def get_data_writer(
        self, data: GriddedData, prognostic_names: Sequence[str]
    ) -> DataWriter:
        return self.data_writer.build(
            experiment_dir=self.experiment_dir,
            n_samples=self.loader.n_samples,
            n_timesteps=self.n_forward_steps + 1,
            prognostic_names=prognostic_names,
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

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()

    start_time = time.time()
    stepper_config = config.load_stepper_config()
    logging.info("Loading inference data")
    data_requirements = stepper_config.get_data_requirements(
        n_forward_steps=config.n_forward_steps
    )

    data = get_inference_data(
        config.loader,
        config.forward_steps_in_memory,
        data_requirements,
    )

    stepper = config.load_stepper(
        data.area_weights.to(fme.get_device()),
        sigma_coordinates=data.sigma_coordinates.to(fme.get_device()),
    )

    aggregator_config: InferenceAggregatorConfig = config.aggregator
    aggregator = aggregator_config.build(
        area_weights=data.area_weights.to(fme.get_device()),
        sigma_coordinates=data.sigma_coordinates,
        record_step_20=config.n_forward_steps >= 20,
        n_timesteps=config.n_forward_steps + 1,
        metadata=data.metadata,
    )

    writer = config.get_data_writer(data, stepper.prognostic_names)

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

    final_flush_start_time = time.time()
    logging.info("Starting final flush of data writer")
    writer.flush()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    for name, ds in aggregator.get_datasets(
        ("time_mean", "zonal_mean", "histogram")
    ).items():
        coords = {k: v for k, v in data.coords.items() if k in ds.dims}
        ds = ds.assign_coords(coords)
        ds.to_netcdf(Path(config.experiment_dir) / f"{name}_diagnostics.nc")

    final_flush_duration = time.time() - final_flush_start_time
    logging.info(f"Final writer flush duration: {final_flush_duration:.2f} seconds")
    timers["final_writer_flush"] = final_flush_duration

    duration = time.time() - start_time
    total_steps = config.n_forward_steps * config.loader.n_samples
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
            # Sleep to avoid overloading wandb API
            time.sleep(0.01)

    config.clean_wandb()

    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
