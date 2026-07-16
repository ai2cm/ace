import copy
import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Literal

import dacite
import torch
import xarray as xr

import fme
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.inference.data_writer.segment import SegmentContext
from fme.ace.inference.inference import InitialConditionConfig, get_initial_condition
from fme.ace.requirements import InitialConditionRequirements
from fme.core.cli import prepare_config, prepare_directory
from fme.core.cloud import exists, makedirs
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.coupled.aggregator import InferenceAggregatorConfig
from fme.coupled.data_loading.batch_data import CoupledPrognosticState
from fme.coupled.data_loading.getters import get_forcing_data
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.data_loading.inference import CoupledForcingDataLoaderConfig
from fme.coupled.inference.data_writer import (
    ATMOSPHERE_OUTPUT_DIR_NAME,
    OCEAN_OUTPUT_DIR_NAME,
    CoupledDataWriterConfig,
    CoupledPairedDataWriter,
    DatasetMetadata,
)
from fme.coupled.stepper import CoupledStepper, CoupledStepperConfig

from .evaluator import (
    StandaloneComponentCheckpointsConfig,
    _validate_coupled_steps_config,
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
        n_ensemble_per_ic: int,
    ) -> CoupledPrognosticState:
        ocean = self.ocean.get_dataset(self.start_indices)
        # time is a required variable but not necessarily a dimension
        sample_dim_name = ocean.time.dims[0]
        atmos = self.atmosphere.get_dataset()
        if sample_dim_name in ocean.indexes:
            atmos = atmos.sel({sample_dim_name: ocean[sample_dim_name]})
        else:
            # Datasets without a sample coordinate (e.g. paired restart files
            # written from a single CoupledPrognosticState) are positionally
            # aligned, so validate the alignment instead of selecting.
            if self.start_indices is not None:
                raise ValueError(
                    "start_indices cannot be used with an ocean initial "
                    f"condition dataset that has no '{sample_dim_name}' "
                    "coordinate to select by."
                )
            if atmos.sizes[sample_dim_name] != ocean.sizes[sample_dim_name]:
                raise ValueError(
                    "Ocean and atmosphere initial condition datasets have no "
                    f"'{sample_dim_name}' coordinate and different numbers of "
                    f"samples: {ocean.sizes[sample_dim_name]} and "
                    f"{atmos.sizes[sample_dim_name]}."
                )
            if not (atmos["time"].values == ocean["time"].values).all():
                raise ValueError(
                    "Ocean and atmosphere initial condition datasets have no "
                    f"'{sample_dim_name}' coordinate and different times; both "
                    "must be at the same coupled step boundary. Got ocean "
                    f"times {ocean['time'].values} and atmosphere times "
                    f"{atmos['time'].values}."
                )
        return CoupledPrognosticState(
            ocean_data=get_initial_condition(
                ds=ocean,
                requirements=InitialConditionRequirements(
                    prognostic_names=ocean_prognostic_names,
                    n_ensemble=n_ensemble_per_ic,
                ),
            ),
            atmosphere_data=get_initial_condition(
                ds=atmos,
                requirements=InitialConditionRequirements(
                    prognostic_names=atmosphere_prognostic_names,
                    n_ensemble=n_ensemble_per_ic,
                ),
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
        n_ensemble_per_ic: Number of ensemble members per initial condition
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
    n_ensemble_per_ic: int = 1

    def __post_init__(self):
        _validate_coupled_steps_config(
            self.n_coupled_steps, self.coupled_steps_in_memory
        )

    def configure_logging(self, log_filename: str):
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=False
        )

    def load_stepper(self) -> CoupledStepper:
        return load_stepper(self.checkpoint_path)

    def load_stepper_config(self) -> CoupledStepperConfig:
        return load_stepper_config(self.checkpoint_path)

    def get_data_writer(
        self,
        data: InferenceGriddedData,
        segment_context: SegmentContext | None = None,
    ) -> CoupledPairedDataWriter:
        if self.data_writer.ocean.time_coarsen is not None:
            try:
                self.data_writer.ocean.time_coarsen.validate(
                    self.coupled_steps_in_memory,
                    self.n_coupled_steps,
                )
            except ValueError as err:
                raise ValueError(
                    f"Ocean time_coarsen config invalid with error: {str(err)}"
                )
        if self.data_writer.atmosphere.time_coarsen is not None:
            try:
                self.data_writer.atmosphere.time_coarsen.validate(
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
            initial_condition_times=data.initial_time.to_numpy(),
            n_timesteps_ocean=self.n_coupled_steps,
            n_timesteps_atmosphere=self.n_coupled_steps * data.n_inner_steps,
            ocean_timestep=data.ocean_timestep,
            atmosphere_timestep=data.atmosphere_timestep,
            variable_metadata=variable_metadata,
            coords=data.coords,
            dataset_metadata=coupled_dataset_metadata,
            segment_context=segment_context,
        )


def main(
    yaml_config: str,
    segments: int | None = None,
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
        if segments is None:
            with GlobalTimer():
                return run_inference_from_config(config)
        else:
            config.configure_logging(log_filename="inference_out.log")
            run_segmented_inference(config, segments)


def run_segmented_inference(config: InferenceConfig, segments: int):
    """Run coupled inference in multiple segments.

    Each segment runs ``config.n_coupled_steps`` coupled steps, writing its
    outputs to a ``segment_{n:04d}`` subdirectory of the experiment directory.
    A segment is complete when both its ocean and atmosphere restart files
    exist; they are written last, after all other segment outputs. Completed
    segments are skipped, so an interrupted run resumes at the first incomplete
    segment when invoked again with the same configuration. Each segment after
    the first initializes from the previous segment's restart files, which sit
    at a coupled step boundary and therefore satisfy the ocean-anchored initial
    condition timing.

    Args:
        config: inference configuration to be used for each individual segment.
            The provided initial condition configuration will only be used for
            the first segment.
        segments: total number of segments desired. Only missing segments will
            be run.
    """
    if config.n_ensemble_per_ic > 1:
        raise ValueError(
            "Ensemble inference (n_ensemble_per_ic > 1) is not supported with "
            "segmented inference. A segment's restart already carries the "
            "broadcasted ensemble as its sample dimension, so later segments "
            "cannot re-broadcast it consistently. Run with n_ensemble_per_ic=1, "
            "or run a single non-segmented inference for ensemble runs."
        )
    logging.info(
        f"Starting segmented coupled inference with {segments} segments. "
        f"Saving to {config.experiment_dir}."
    )
    config_copy = copy.deepcopy(config)
    original_wandb_name = os.environ.get("WANDB_NAME")
    previous_segment_dir: str | None = None
    for segment in range(segments):
        segment_label = f"segment_{segment:04d}"
        segment_dir = os.path.join(config.experiment_dir, segment_label)
        ocean_restart_path = os.path.join(
            segment_dir, OCEAN_OUTPUT_DIR_NAME, "restart.nc"
        )
        atmosphere_restart_path = os.path.join(
            segment_dir, ATMOSPHERE_OUTPUT_DIR_NAME, "restart.nc"
        )
        if exists(ocean_restart_path) and exists(atmosphere_restart_path):
            logging.info(f"Skipping segment {segment} because it has already been run.")
        else:
            logging.info(f"Running segment {segment}.")
            config_copy.experiment_dir = segment_dir
            if original_wandb_name is not None:
                os.environ["WANDB_NAME"] = f"{original_wandb_name}-{segment_label}"
            segment_context = SegmentContext(
                segment_index=segment,
                total_segments=segments,
                run_dir=config.experiment_dir,
                segment_dir=segment_dir,
                previous_segment_dir=previous_segment_dir,
            )
            with GlobalTimer():
                run_inference_from_config(config_copy, segment_context=segment_context)
        previous_segment_dir = segment_dir
        config_copy.initial_condition = CoupledInitialConditionConfig(
            ocean=ComponentInitialConditionConfig(
                path=ocean_restart_path, engine="netcdf4"
            ),
            atmosphere=ComponentInitialConditionConfig(
                path=atmosphere_restart_path, engine="netcdf4"
            ),
        )


def run_inference_from_config(
    config: InferenceConfig,
    segment_context: SegmentContext | None = None,
):
    timer = GlobalTimer.get_instance()
    timer.start_outer("inference")
    timer.start("initialization")

    makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    stepper_config = config.load_stepper_config()
    data_requirements = stepper_config.get_forcing_window_data_requirements(
        n_coupled_steps=config.coupled_steps_in_memory
    )
    logging.info("Loading initial condition data")
    initial_condition = config.initial_condition.get_initial_condition(
        ocean_prognostic_names=stepper_config.ocean.stepper.prognostic_names,
        atmosphere_prognostic_names=stepper_config.atmosphere.stepper.prognostic_names,
        n_ensemble_per_ic=config.n_ensemble_per_ic,
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

    writer = config.get_data_writer(data=data, segment_context=segment_context)
    timer.stop("initialization")
    logging.info("Starting inference")
    logger = get_record_to_wandb(label="inference")
    run_inference(
        predict=stepper.predict_paired,
        data=data,
        aggregator=aggregator,
        writer=writer,
        record_logs=logger.log,
    )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.finalize()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop("final_writer_flush")

    timer.stop_outer("inference")
    total_steps = (
        config.n_coupled_steps * stepper.n_inner_steps
    ) * data.n_initial_conditions
    inference_duration = timer.get_duration("inference")
    wandb_logging_duration = timer.get_duration("inference/wandb_logging")
    total_steps_per_second = total_steps / (inference_duration - wandb_logging_duration)
    timer.log_durations()
    logging.info(
        "Total steps per second (ignoring wandb logging): "
        f"{total_steps_per_second:.2f} steps/second"
    )

    summary_logs = {
        "total_steps_per_second": total_steps_per_second,
        **aggregator.get_summary_logs(),
    }
    logger.log_to_current_step(summary_logs)
    logger.log_to_current_step(timer.get_durations())
