import dataclasses
import logging
import os
from typing import Callable, Optional, Sequence

import dacite
import torch

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator.inference import InferenceEvaluatorAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.getters import get_inference_data
from fme.ace.data_loading.gridded_data import InferenceGriddedData
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.ace.inference.loop import DeriverABC, run_dataset_comparison
from fme.ace.stepper import (
    SingleModuleStepperConfig,
    Stepper,
    StepperOverrideConfig,
    load_stepper,
    load_stepper_config,
)
from fme.core.cli import prepare_config, prepare_directory
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping


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

    Parameters:
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
        stepper_override: Configuration for overriding select stepper configuration
            options at inference time (optional).
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
    stepper_override: Optional[StepperOverrideConfig] = None

    def __post_init__(self):
        if self.data_writer.time_coarsen is not None:
            validate_time_coarsen_config(
                self.data_writer.time_coarsen,
                self.forward_steps_in_memory,
                self.n_forward_steps,
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(
        self, env_vars: Optional[dict] = None, resumable: bool = False, **kwargs
    ):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def load_stepper(self) -> Stepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path, self.stepper_override)

    def load_stepper_config(self) -> SingleModuleStepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path, self.stepper_override)

    def get_data_writer(self, data: InferenceGriddedData) -> PairedDataWriter:
        variable_metadata = get_derived_variable_metadata() | data.variable_metadata
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            n_initial_conditions=self.loader.n_initial_conditions,
            n_timesteps=self.n_forward_steps,
            timestep=data.timestep,
            variable_metadata=variable_metadata,
            coords=data.coords,
        )


def main(yaml_config: str, override_dotlist: Optional[Sequence[str]] = None):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.experiment_dir, config_data)
    with GlobalTimer():
        return run_evaluator_from_config(config)


class _Deriver(DeriverABC):
    """
    DeriverABC implementation for dataset comparison.
    """

    def __init__(
        self,
        n_ic_timesteps: int,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
    ):
        self._n_ic_timesteps = n_ic_timesteps
        self._derive_func = derive_func

    @property
    def n_ic_timesteps(self) -> int:
        return self._n_ic_timesteps

    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData:
        if compute_derived_variables:
            timer = GlobalTimer.get_instance()
            with timer.context("compute_derived_variables"):
                data = data.compute_derived_variables(
                    derive_func=self._derive_func,
                    forcing_data=data,
                )
        return data.remove_initial_condition(self._n_ic_timesteps)


def run_evaluator_from_config(config: InferenceEvaluatorConfig):
    timer = GlobalTimer.get_instance()
    timer.start_outer("inference")
    timer.start("initialization")

    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    logging.info(f"Current device is {fme.get_device()}")

    stepper_config = config.load_stepper_config()
    logging.info("Loading inference data")
    window_requirements = stepper_config.get_evaluation_window_data_requirements(
        n_forward_steps=config.forward_steps_in_memory
    )
    initial_condition_requirements = (
        stepper_config.get_prognostic_state_data_requirements()
    )
    data = get_inference_data(
        config=config.loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
    )

    stepper = config.load_stepper()
    stepper.set_eval()
    stepper.validate_inference_data(data)

    aggregator_config: InferenceEvaluatorAggregatorConfig = config.aggregator
    for batch in data.loader:
        initial_time = batch.time.isel(time=0)
        break
    variable_metadata = get_derived_variable_metadata() | data.variable_metadata
    aggregator = aggregator_config.build(
        horizontal_coordinates=data.horizontal_coordinates,
        timestep=data.timestep,
        record_step_20=config.n_forward_steps >= 20,
        n_timesteps=config.n_forward_steps + stepper_config.n_ic_timesteps,
        variable_metadata=variable_metadata,
        initial_time=initial_time,
        channel_mean_names=stepper.loss_names,
        normalize=stepper.normalizer.normalize,
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(data)

    timer.stop()
    logging.info("Starting inference")
    record_logs = get_record_to_wandb(label="inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            total_forward_steps=config.n_forward_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition_requirements,
        )
        deriver = _Deriver(
            n_ic_timesteps=stepper_config.n_ic_timesteps,
            derive_func=stepper.derive_func,
        )
        run_dataset_comparison(
            aggregator=aggregator,
            prediction_data=prediction_data,
            target_data=data,
            deriver=deriver,
            writer=writer,
            record_logs=record_logs,
        )
    else:
        run_inference(
            predict=stepper.predict_paired,
            data=data,
            aggregator=aggregator,
            writer=writer,
            record_logs=record_logs,
        )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.flush()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop()

    timer.stop_outer("inference")
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
