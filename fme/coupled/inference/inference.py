import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Literal

import dacite
import torch
import xarray as xr

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.inference.evaluator import validate_time_coarsen_config
from fme.ace.inference.inference import InitialConditionConfig, get_initial_condition
from fme.core.cli import prepare_config, prepare_directory
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.coupled.aggregator import InferenceAggregatorConfig
from fme.coupled.data_loading.batch_data import CoupledPrognosticState
from fme.coupled.data_loading.getters import get_forcing_data
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.data_loading.inference import CoupledForcingDataLoaderConfig
from fme.coupled.inference.data_writer import (
    CoupledDataWriterConfig,
    CoupledPairedDataWriter,
    DatasetMetadata,
)
from fme.coupled.stepper import CoupledStepper, CoupledStepperConfig

from .evaluator import (
    StandaloneComponentCheckpointsConfig,
    load_stepper,
    load_stepper_config,
)

StartIndices = InferenceInitialConditionIndices | ExplicitIndices | TimestampList


@dataclasses.dataclass
class ComponentInitialConditionConfig:
    """
    Parameters:
        path: Path to the component initial condition dataset.
        engine: Backend used in xarray.open_dataset call.
    """

    path: str
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"

    def get_dataset(self, start_indices: StartIndices | None = None) -> xr.Dataset:
        ic_config = InitialConditionConfig(
            path=self.path,
            engine=self.engine,
            start_indices=start_indices,
        )
        return ic_config.get_dataset()


@dataclasses.dataclass
class CoupledInitialConditionConfig:
    """
    Configuration for initial conditions in coupled inference.

    Parameters:
        ocean: Configuration for the ocean initial conditions.
        atmosphere: Configuration for the atmosphere initial conditions.
        start_indices: Indices to use for selecting initial conditions,
            should correspond to the ocean initial condition dataset.
    """

    ocean: ComponentInitialConditionConfig
    atmosphere: ComponentInitialConditionConfig
    start_indices: StartIndices | None = None

    def get_initial_condition(
        self,
        ocean_prognostic_names: Sequence[str],
        atmosphere_prognostic_names: Sequence[str],
        labels: list[str],
    ) -> CoupledPrognosticState:
        ocean = self.ocean.get_dataset(self.start_indices)
        # time is a required variable but not necessarily a dimension
        sample_dim_name = ocean.time.dims[0]
        atmos = self.atmosphere.get_dataset().sel(
            {sample_dim_name: ocean[sample_dim_name]}
        )
        return CoupledPrognosticState(
            ocean_data=get_initial_condition(
                ds=ocean,
                prognostic_names=ocean_prognostic_names,
                labels=labels,
            ),
            atmosphere_data=get_initial_condition(
                ds=atmos,
                prognostic_names=atmosphere_prognostic_names,
                labels=labels,
            ),
        )


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Parameters:
        experiment_dir: Directory to save results to.
        n_coupled_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to a CoupledStepper training checkpoint to load, or a
            mapping to two separate Stepper training checkpoints.
        logging: configuration for logging.
        initial_condition: Configuration for initial condition data.
        forcing_loader: Configuration for forcing data.
        coupled_steps_in_memory: Number of coupled steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference aggregator.
        labels: Dataset labels to use for inference.
    """

    experiment_dir: str
    n_coupled_steps: int
    checkpoint_path: str | StandaloneComponentCheckpointsConfig
    logging: LoggingConfig
    initial_condition: CoupledInitialConditionConfig
    forcing_loader: CoupledForcingDataLoaderConfig
    coupled_steps_in_memory: int = 1
    data_writer: CoupledDataWriterConfig = dataclasses.field(
        default_factory=lambda: CoupledDataWriterConfig()
    )
    aggregator: InferenceAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceAggregatorConfig()
    )
    labels: list[str] = dataclasses.field(default_factory=list)

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
        n_initial_conditions: int,
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
            n_initial_conditions=n_initial_conditions,
            n_timesteps_ocean=self.n_coupled_steps,
            n_timesteps_atmosphere=self.n_coupled_steps * data.n_inner_steps,
            ocean_timestep=data.ocean_timestep,
            atmosphere_timestep=data.atmosphere_timestep,
            variable_metadata=variable_metadata,
            coords=data.coords,
            dataset_metadata=coupled_dataset_metadata,
        )


def main(
    yaml_config: str,
    override_dotlist: Sequence[str] | None = None,
):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.experiment_dir, config_data)
    with torch.no_grad():
        with GlobalTimer():
            return run_inference_from_config(config)


def run_inference_from_config(config: InferenceConfig):
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
    data_requirements = stepper_config.get_forcing_window_data_requirements(
        n_coupled_steps=config.coupled_steps_in_memory
    )
    logging.info("Loading initial condition data")
    initial_condition = config.initial_condition.get_initial_condition(
        ocean_prognostic_names=stepper_config.ocean.stepper.prognostic_names,
        atmosphere_prognostic_names=stepper_config.atmosphere.stepper.prognostic_names,
        labels=config.labels,
    )
    stepper = config.load_stepper()
    stepper.set_eval()
    logging.info("Initializing forcing data loader")
    data = get_forcing_data(
        config=config.forcing_loader,
        total_coupled_steps=config.n_coupled_steps,
        window_requirements=data_requirements,
        initial_condition=initial_condition,
        dataset_info=stepper.training_dataset_info,
    )

    aggregator_config: InferenceAggregatorConfig = config.aggregator
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
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(
        n_initial_conditions=data.n_initial_conditions, data=data
    )
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
    ) * data.n_initial_conditions
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
