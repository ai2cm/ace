import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence

import dacite
import torch

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.inference.evaluator import validate_time_coarsen_config
from fme.ace.stepper import load_stepper as load_single_stepper
from fme.ace.stepper import load_stepper_config as load_single_stepper_config
from fme.core.cli import prepare_config, prepare_directory
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.coupled.aggregator import InferenceEvaluatorAggregatorConfig
from fme.coupled.data_loading.getters import get_inference_data
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.inference.data_writer import (
    CoupledDataWriterConfig,
    CoupledPairedDataWriter,
    DatasetMetadata,
)
from fme.coupled.stepper import (
    ComponentConfig,
    CoupledOceanFractionConfig,
    CoupledStepper,
    CoupledStepperConfig,
    load_coupled_stepper,
)


@dataclasses.dataclass
class StandaloneComponentConfig:
    """
    Configuration specifying the path to one of the components (ocean or
    atmosphere) within a CoupledStepper. Intended for inference with separate
    pretrained Stepper training checkpoints.

    """

    timedelta: str
    path: str


@dataclasses.dataclass
class StandaloneComponentCheckpointsConfig:
    """
    Configuration for creating a CoupledStepper from two separate Stepper
    checkpoints, for standalone inference.

    Parameters:
        ocean: The ocean component configuration.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include 'ocean'.
        sst_name: Name of the sea surface temperature field in the ocean data.

    """

    ocean: StandaloneComponentConfig
    atmosphere: StandaloneComponentConfig
    sst_name: str = "sst"
    ocean_fraction_prediction: CoupledOceanFractionConfig | None = None

    def load_stepper_config(self) -> CoupledStepperConfig:
        return CoupledStepperConfig(
            ocean=ComponentConfig(
                timedelta=self.ocean.timedelta,
                stepper=load_single_stepper_config(self.ocean.path),
            ),
            atmosphere=ComponentConfig(
                timedelta=self.atmosphere.timedelta,
                stepper=load_single_stepper_config(self.atmosphere.path),
            ),
            sst_name=self.sst_name,
            ocean_fraction_prediction=self.ocean_fraction_prediction,
        )

    def load_stepper(self) -> CoupledStepper:
        ocean = load_single_stepper(self.ocean.path)
        atmosphere = load_single_stepper(self.atmosphere.path)
        dataset_info = CoupledDatasetInfo(
            ocean=ocean.training_dataset_info,
            atmosphere=atmosphere.training_dataset_info,
        )

        return CoupledStepper(
            config=self.load_stepper_config(),
            ocean=ocean,
            atmosphere=atmosphere,
            dataset_info=dataset_info,
        )


def load_stepper_config(
    checkpoint_path: str | pathlib.Path | StandaloneComponentCheckpointsConfig,
) -> CoupledStepperConfig:
    """Load a coupled stepper configuration.

    Args:
        checkpoint_path: The path to the serialized CoupledStepper checkpoint, or a
            StandaloneComponentCheckpointsConfig.

    Returns:
        The CoupledStepperConfig from the serialized checkpoint or constructed from the
        standalone ocean and atmosphere checkpoints.
    """
    if isinstance(checkpoint_path, StandaloneComponentCheckpointsConfig):
        logging.info(
            f"Loading ocean model checkpoint from {checkpoint_path.ocean.path}"
        )
        logging.info(
            "Loading atmosphere model checkpoint from "
            f"{checkpoint_path.atmosphere.path}"
        )
        return checkpoint_path.load_stepper_config()

    logging.info(f"Loading trained coupled model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=fme.get_device(), weights_only=False
    )
    config = CoupledStepperConfig.from_state(checkpoint["stepper"]["config"])

    return config


def load_stepper(
    checkpoint_path: str | pathlib.Path | StandaloneComponentCheckpointsConfig,
) -> CoupledStepper:
    """Load a coupled stepper.

    Args:
        checkpoint_path: The path to the serialized CoupledStepper checkpoint, or a
            StandaloneComponentCheckpointsConfig.

    Returns:
        The CoupledStepper serialized in the checkpoint or constructed from the
        standalone ocean and atmosphere checkpoints.
    """
    if isinstance(checkpoint_path, StandaloneComponentCheckpointsConfig):
        logging.info(
            f"Loading ocean model checkpoint from {checkpoint_path.ocean.path}"
        )
        logging.info(
            "Loading atmosphere model checkpoint from "
            f"{checkpoint_path.atmosphere.path}"
        )
        return checkpoint_path.load_stepper()

    return load_coupled_stepper(checkpoint_path)


@dataclasses.dataclass
class InferenceEvaluatorConfig:
    """
    Configuration for running inference including comparison to reference data.

    Parameters:
        experiment_dir: Directory to save results to.
        n_coupled_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to a CoupledStepper training checkpoint to load, or a
            mapping to two separate Stepper training checkpoints.
        logging: configuration for logging.
        loader: Configuration for data to be used as initial conditions, forcing, and
            target in inference.
        coupled_steps_in_memory: Number of coupled steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference evaluator aggregator.
    """

    experiment_dir: str
    n_coupled_steps: int
    checkpoint_path: str | StandaloneComponentCheckpointsConfig
    logging: LoggingConfig
    loader: InferenceDataLoaderConfig
    coupled_steps_in_memory: int = 1
    data_writer: CoupledDataWriterConfig = dataclasses.field(
        default_factory=lambda: CoupledDataWriterConfig()
    )
    aggregator: InferenceEvaluatorAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceEvaluatorAggregatorConfig()
    )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(
        self, env_vars: dict | None = None, resumable: bool = False, **kwargs
    ):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def load_stepper(self) -> CoupledStepper:
        return load_stepper(self.checkpoint_path)

    def load_stepper_config(self) -> CoupledStepperConfig:
        return load_stepper_config(self.checkpoint_path)

    def get_data_writer(
        self,
        data: InferenceGriddedData,
    ) -> CoupledPairedDataWriter:
        if self.data_writer.ocean.time_coarsen is not None:
            try:
                validate_time_coarsen_config(
                    self.data_writer.ocean.time_coarsen,
                    self.coupled_steps_in_memory,
                    self.n_coupled_steps,
                )
            except ValueError as err:
                raise ValueError(
                    f"Ocean time_coarsen config invalid with error: {str(err)}"
                )
        if self.data_writer.atmosphere.time_coarsen is not None:
            try:
                validate_time_coarsen_config(
                    self.data_writer.atmosphere.time_coarsen,
                    self.coupled_steps_in_memory * data.n_inner_steps,
                    self.n_coupled_steps * data.n_inner_steps,
                )
            except ValueError as err:
                raise ValueError(
                    f"Atmosphere time_coarsen config invalid with error: {str(err)}"
                )

        variable_metadata = get_derived_variable_metadata() | data.variable_metadata
        dataset_metadata = DatasetMetadata.from_env()
        coupled_dataset_metadata = {
            "ocean": dataset_metadata,
            "atmosphere": dataset_metadata,
        }
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            n_initial_conditions=self.loader.n_initial_conditions,
            n_timesteps_ocean=self.n_coupled_steps,
            n_timesteps_atmosphere=self.n_coupled_steps * data.n_inner_steps,
            ocean_timestep=data.ocean_timestep,
            atmosphere_timestep=data.atmosphere_timestep,
            variable_metadata=variable_metadata,
            coords=data.coords,
            dataset_metadata=coupled_dataset_metadata,
        )


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.experiment_dir, config_data)
    with GlobalTimer(), torch.no_grad():
        return run_evaluator_from_config(config)


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
        n_coupled_steps=config.coupled_steps_in_memory
    )
    initial_condition_requirements = (
        stepper_config.get_prognostic_state_data_requirements()
    )
    stepper = config.load_stepper()
    data = get_inference_data(
        config=config.loader,
        total_coupled_steps=config.n_coupled_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
        dataset_info=stepper.training_dataset_info,
    )
    stepper.set_eval()

    aggregator_config: InferenceEvaluatorAggregatorConfig = config.aggregator
    batch = next(iter(data.loader))
    initial_time = batch.ocean_data.time.isel(time=0)
    variable_metadata = get_derived_variable_metadata() | data.variable_metadata
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
    n_timesteps_ocean = config.n_coupled_steps + stepper.ocean.n_ic_timesteps
    n_timesteps_atmosphere = (
        config.n_coupled_steps * stepper.n_inner_steps
        + stepper.atmosphere.n_ic_timesteps
    )
    aggregator = aggregator_config.build(
        dataset_info=dataset_info,
        n_timesteps_ocean=n_timesteps_ocean,
        n_timesteps_atmosphere=n_timesteps_atmosphere,
        initial_time=initial_time,
        ocean_normalize=stepper.ocean.normalizer.normalize,
        atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(data)

    timer.stop()
    logging.info("Starting inference")
    record_logs = get_record_to_wandb(label="inference")
    run_inference(
        predict=stepper.predict_paired,
        data=data,
        aggregator=aggregator,
        writer=writer,
        record_logs=record_logs,
    )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.finalize()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop()

    timer.stop_outer("inference")
    total_steps = (
        config.n_coupled_steps * stepper.n_inner_steps
    ) * config.loader.n_initial_conditions
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
