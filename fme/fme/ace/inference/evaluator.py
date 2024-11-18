import argparse
import dataclasses
import logging
import os
from typing import Optional, Sequence

import dacite
import torch
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.loop import (
    DeriverABC,
    get_record_to_wandb,
    run_dataset_comparison,
    run_inference_evaluator,
    write_reduced_metrics,
)
from fme.ace.inference.timing import GlobalTimer
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference import InferenceEvaluatorAggregatorConfig
from fme.core.data_loading.batch_data import BatchData, GriddedData
from fme.core.data_loading.getters import get_inference_data
from fme.core.data_loading.inference import InferenceDataLoaderConfig
from fme.core.dicts import to_flat_dict
from fme.core.logging_utils import LoggingConfig
from fme.core.ocean import OceanConfig
from fme.core.stepper import SingleModuleStepperConfig


def load_stepper_config(
    checkpoint_file: str, ocean_config: Optional[OceanConfig]
) -> SingleModuleStepperConfig:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    config = SingleModuleStepperConfig.from_state(checkpoint["stepper"]["config"])
    if ocean_config is not None:
        logging.info(
            "Overriding training ocean configuration with the inference ocean config."
        )
        config.ocean = ocean_config
    return config


def load_stepper(
    checkpoint_file: str,
    ocean_config: Optional[OceanConfig] = None,
) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(checkpoint["stepper"])
    if ocean_config is not None:
        logging.info(
            "Overriding training ocean configuration with the inference ocean config."
        )
        new_ocean = ocean_config.build(
            stepper.in_names, stepper.out_names, stepper.timestep
        )
        stepper.replace_ocean(new_ocean)
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

    def load_stepper(self) -> SingleModuleStepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        stepper = load_stepper(self.checkpoint_path, ocean_config=self.ocean)
        return stepper

    def load_stepper_config(self) -> SingleModuleStepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path, ocean_config=self.ocean)

    def get_data_writer(
        self, data: GriddedData, prognostic_names: Sequence[str]
    ) -> PairedDataWriter:
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            n_initial_conditions=self.loader.n_initial_conditions,
            n_timesteps=self.n_forward_steps,
            timestep=data.timestep,
            prognostic_names=prognostic_names,
            variable_metadata=data.variable_metadata,
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
    with GlobalTimer():
        return run_evaluator_from_config(config)


class _Deriver(DeriverABC):
    """
    DeriverABC implementation for dataset comparison.
    """

    def __init__(self, n_ic_timesteps: int):
        self._n_ic_timesteps = n_ic_timesteps

    @property
    def n_ic_timesteps(self) -> int:
        return self._n_ic_timesteps

    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData:
        if compute_derived_variables:
            timer = GlobalTimer.get_instance()
            with timer.context("compute_derived_variables"):
                data = data.compute_derived_variables(data)
        return data.remove_initial_condition(self._n_ic_timesteps)


def run_evaluator_from_config(config: InferenceEvaluatorConfig):
    timer = GlobalTimer.get_instance()
    timer.start("inference")
    timer.start("initialization")

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging.info(f"Current device is {fme.get_device()}")

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

    stepper = config.load_stepper()
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
        sigma_coordinates=data.sigma_coordinates,
        horizontal_coordinates=data.horizontal_coordinates,
        timestep=data.timestep,
        record_step_20=config.n_forward_steps >= 20,
        n_timesteps=config.n_forward_steps + 1,
        variable_metadata=data.variable_metadata,
        initial_times=initial_times,
        channel_mean_names=stepper.out_names,
    )

    writer = config.get_data_writer(data, stepper.prognostic_names)

    timer.stop("initialization")
    logging.info("Starting inference")
    record_logs = get_record_to_wandb(label="inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            config.forward_steps_in_memory,
            data_requirements,
        )
        deriver = _Deriver(n_ic_timesteps=stepper_config.n_ic_timesteps)
        run_dataset_comparison(
            aggregator=aggregator,
            normalizer=stepper.normalizer,
            prediction_data=prediction_data,
            target_data=data,
            deriver=deriver,
            writer=writer,
            record_logs=record_logs,
        )
    else:
        run_inference_evaluator(
            aggregator=aggregator,
            writer=writer,
            stepper=stepper,
            data=data,
            record_logs=record_logs,
        )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.flush()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    write_reduced_metrics(
        aggregator,
        data.coords,
        config.experiment_dir,
        excluded=[
            "video",
        ],
    )
    timer.stop("final_writer_flush")

    timer.stop("inference")
    total_steps = config.n_forward_steps * config.loader.n_initial_conditions
    inference_duration = timer.get_duration("inference")
    wandb_logging_duration = timer.get_duration("wandb_logging")
    total_steps_per_second = total_steps / (inference_duration - wandb_logging_duration)
    timer.log_durations()
    logging.info(
        "Total steps per second (ignoring wandb logging): "
        f"{total_steps_per_second:.2f} steps/second"
    )

    summary_logs = {
        "total_steps_per_second": total_steps_per_second,
        **timer.get_durations(),
        **aggregator.get_summary_logs(),
    }
    record_logs([summary_logs])
    config.clean_wandb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
