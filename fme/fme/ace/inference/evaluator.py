import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import dacite
import torch
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.loop import run_dataset_comparison, run_inference_evaluator
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference import InferenceEvaluatorAggregatorConfig
from fme.core.data_loading.data_typing import GriddedData, SigmaCoordinates
from fme.core.data_loading.getters import get_inference_data
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.dicts import to_flat_dict
from fme.core.logging_utils import LoggingConfig
from fme.core.ocean import OceanConfig
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.wandb import WandB


def load_stepper_config(checkpoint_file: str) -> SingleModuleStepperConfig:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    return SingleModuleStepperConfig.from_state(checkpoint["stepper"]["config"])


def load_stepper(
    checkpoint_file: str,
    area: torch.Tensor,
    sigma_coordinates: SigmaCoordinates,
    ocean_config: Optional[OceanConfig] = None,
) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(
        checkpoint["stepper"], area=area, sigma_coordinates=sigma_coordinates
    )
    if ocean_config is not None:
        logging.info(
            "Overriding training ocean configuration with the inference ocean config."
        )
        new_ocean = ocean_config.build(
            stepper.in_packer.names, stepper.out_packer.names, stepper.timestep
        )
        stepper.ocean = new_ocean
    return stepper


def validate_time_coarsen_config(
    config: TimeCoarsenConfig, forward_steps_in_memory: int, n_forward_steps: int
):
    coarsen_factor = config.coarsen_factor
    if forward_steps_in_memory % coarsen_factor != 0:
        raise ValueError(
            "forward_steps_in_memory must be divisible by "
            f"time_coarsen.coarsen_factor. Got {forward_steps_in_memory} "
            f"and {coarsen_factor}."
        )
    if n_forward_steps % coarsen_factor != 0:
        raise ValueError(
            "n_forward_steps must be divisible by "
            f"time_coarsen.coarsen_factor. Got {n_forward_steps} "
            f"and {coarsen_factor}."
        )


@dataclasses.dataclass
class InferenceEvaluatorConfig:
    """
    Configuration for running inference including comparison to reference data.

    Attributes:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for.
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
        aggregator: Configuration for inference evaluator aggregator.
        ocean: Ocean configuration for running inference with a
            different one than what is used in training.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    loader: InferenceDataLoaderConfig
    prediction_loader: Optional[InferenceDataLoaderConfig] = None
    forward_steps_in_memory: int = 1
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    aggregator: InferenceEvaluatorAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceEvaluatorAggregatorConfig()
    )
    ocean: Optional[OceanConfig] = None

    def __post_init__(self):
        if self.data_writer.time_coarsen is not None:
            validate_time_coarsen_config(
                self.data_writer.time_coarsen,
                self.forward_steps_in_memory,
                self.n_forward_steps,
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, env_vars: Optional[dict] = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resume=False, **kwargs
        )

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
        stepper = load_stepper(
            self.checkpoint_path,
            area=area,
            sigma_coordinates=sigma_coordinates,
            ocean_config=self.ocean,
        )
        return stepper

    def load_stepper_config(self) -> SingleModuleStepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path)

    def get_data_writer(
        self, data: GriddedData, prognostic_names: Sequence[str]
    ) -> PairedDataWriter:
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            n_samples=self.loader.n_samples,
            n_timesteps=self.n_forward_steps,
            timestep=data.timestep,
            prognostic_names=prognostic_names,
            metadata=data.metadata,
            coords=data.coords,
        )


def main(yaml_config: str):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return run_evaluator_from_config(config)


def run_evaluator_from_config(config: InferenceEvaluatorConfig):
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging.info(f"Current device is {fme.get_device()}")

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
    if stepper.timestep != data.timestep:
        raise ValueError(
            f"Timestep of the loaded stepper, {stepper.timestep}, does not "
            f"match that of the forcing data, {data.timestep}."
        )

    aggregator_config: InferenceEvaluatorAggregatorConfig = config.aggregator
    for batch in data.loader:
        initial_times = batch.times.isel(time=0)
        break
    aggregator = aggregator_config.build(
        area_weights=data.area_weights.to(fme.get_device()),
        sigma_coordinates=data.sigma_coordinates,
        timestep=data.timestep,
        record_step_20=config.n_forward_steps >= 20,
        n_timesteps=config.n_forward_steps + 1,
        metadata=data.metadata,
        data_grid=data.grid,
        initial_times=initial_times,
    )

    writer = config.get_data_writer(data, stepper.prognostic_names)

    logging.info("Starting inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            config.forward_steps_in_memory,
            data_requirements,
        )

        timers = run_dataset_comparison(
            aggregator=aggregator,
            normalizer=stepper.normalizer,
            prediction_data=prediction_data,
            target_data=data,
            writer=writer,
        )
    else:
        timers = run_inference_evaluator(
            aggregator=aggregator,
            writer=writer,
            stepper=stepper,
            data=data,
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
            wandb.log(log, step=i, sleep=0.01)

    config.clean_wandb()

    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
