import copy
import dataclasses
import datetime
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Literal

import dacite
import numpy as np
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator.inference import InferenceAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.stepper import (
    Stepper,
    StepperOverrideConfig,
    load_stepper,
    load_stepper_config,
)
from fme.ace.stepper.single_module import StepperConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import IncompatibleDatasetInfo
from fme.core.dicts import to_flat_dict
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer

from .evaluator import resolve_variable_metadata, validate_time_coarsen_config

StartIndices = InferenceInitialConditionIndices | ExplicitIndices | TimestampList


@dataclasses.dataclass
class InitialConditionConfig:
    """
    Configuration for initial conditions.

    .. note::
        The data specified under path should contain a time dimension of at least
        length 1. If multiple times are present in the dataset specified by ``path``,
        the inference will start an ensemble simulation using each IC along a
        leading sample dimension. Specific times can be selected from the dataset
        by using ``start_indices``.

    Parameters:
        path: The path to the initial conditions dataset.
        engine: The engine used to open the dataset.
        start_indices: optional specification of the subset of
            initial conditions to use.
    """

    path: str
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"
    start_indices: StartIndices | None = None

    def get_dataset(self) -> xr.Dataset:
        ds = xr.open_dataset(
            self.path,
            engine=self.engine,
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
        )
        return self._subselect_initial_conditions(ds)

    def _subselect_initial_conditions(self, ds: xr.Dataset) -> xr.Dataset:
        if self.start_indices is None:
            ic_indices = slice(None, None)
        elif isinstance(self.start_indices, TimestampList):
            time_index = xr.CFTimeIndex(ds.time.values)
            ic_indices = self.start_indices.as_indices(time_index)
        else:
            ic_indices = self.start_indices.as_indices()
        # time is a required variable but not necessarily a dimension
        sample_dim_name = ds.time.dims[0]
        return ds.isel({sample_dim_name: ic_indices})


def get_initial_condition(
    ds: xr.Dataset, prognostic_names: Sequence[str]
) -> PrognosticState:
    """Given a dataset, extract a mapping of variables to tensors.
    and the time coordinate corresponding to the initial conditions.

    Args:
        ds: Dataset containing initial condition data. Must include prognostic_names
            as variables, and they must each have shape (n_samples, n_lat, n_lon).
            Dataset must also include a 'time' variable with length n_samples.
        prognostic_names: Names of prognostic variables to extract from the dataset.

    Returns:
        The initial condition and the time coordinate.
    """
    initial_condition = {}
    for name in prognostic_names:
        if len(ds[name].shape) != 3:
            raise ValueError(
                f"Initial condition variables {name} must have shape "
                f"(n_samples, n_lat, n_lon). Got shape {ds[name].shape}."
            )
        n_samples = ds[name].shape[0]
        initial_condition[name] = torch.tensor(ds[name].values).unsqueeze(dim=1)
    if "time" not in ds:
        raise ValueError("Initial condition dataset must have a 'time' variable.")
    initial_times = xr.DataArray(
        data=ds.time.values[:, None],
        dims=["sample", "time"],
    )
    if initial_times.shape[0] != n_samples:
        raise ValueError(
            "Length of 'time' variable must match first dimension of variables "
            f"in initial condition dataset. Got {initial_times.shape[0]} "
            f"and {n_samples}."
        )

    batch_data = BatchData.new_on_cpu(
        data=initial_condition,
        time=initial_times,
        horizontal_dims=["lat", "lon"],
    )
    return batch_data.get_start(prognostic_names, n_ic_timesteps=1)


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Parameters:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: Configuration for logging.
        initial_condition: Configuration for initial condition data.
        forcing_loader: Configuration for forcing data.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference aggregator.
        stepper_override: Configuration for overriding select stepper configuration
            options at inference time (optional).
        allow_incompatible_dataset: If True, allow the dataset used for inference
            to be incompatible with the dataset used for stepper training. This should
            be used with caution, as it may allow the stepper to make scientifically
            invalid predictions, but it can allow running inference with incorrectly
            formatted or missing grid information.
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    initial_condition: InitialConditionConfig
    forcing_loader: ForcingDataLoaderConfig
    forward_steps_in_memory: int = 10
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    aggregator: InferenceAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceAggregatorConfig()
    )
    stepper_override: StepperOverrideConfig | None = None
    allow_incompatible_dataset: bool = False

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
        self, env_vars: dict | None = None, resumable: bool = False, **kwargs
    ):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def load_stepper(self) -> Stepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path, self.stepper_override)

    def load_stepper_config(self) -> StepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path, self.stepper_override)

    def get_data_writer(
        self,
        n_initial_conditions: int,
        timestep: datetime.timedelta,
        coords: Mapping[str, np.ndarray],
        variable_metadata: Mapping[str, VariableMetadata],
    ) -> PairedDataWriter:
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            # each batch contains all samples, for different times
            n_initial_conditions=n_initial_conditions,
            n_timesteps=self.n_forward_steps,
            timestep=timestep,
            variable_metadata=variable_metadata,
            coords=coords,
            dataset_metadata=DatasetMetadata.from_env(),
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
    if segments is None:
        with GlobalTimer():
            return run_inference_from_config(config)
    else:
        config.configure_logging(log_filename="inference_out.log")
        run_segmented_inference(config, segments)


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
        n_forward_steps=config.forward_steps_in_memory
    )
    logging.info("Loading initial condition data")
    initial_condition = get_initial_condition(
        config.initial_condition.get_dataset(), stepper_config.prognostic_names
    )
    stepper = config.load_stepper()
    stepper.set_eval()
    logging.info("Initializing forcing data loader")
    data = get_forcing_data(
        config=config.forcing_loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=data_requirements,
        initial_condition=initial_condition,
        surface_temperature_name=stepper.surface_temperature_name,
        ocean_fraction_name=stepper.ocean_fraction_name,
    )
    if not config.allow_incompatible_dataset:
        try:
            stepper.training_dataset_info.assert_compatible_with(data.dataset_info)
        except IncompatibleDatasetInfo as err:
            raise IncompatibleDatasetInfo(
                "Inference dataset is not compatible with dataset used for stepper "
                "training. Set allow_incompatible_dataset to True to ignore this "
                f"error. The incompatiblity found was: {str(err)}"
            ) from err

    variable_metadata = resolve_variable_metadata(
        dataset_metadata=data.variable_metadata,
        stepper_metadata=stepper.training_variable_metadata,
        stepper_all_names=stepper_config.all_names,
    )
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
    aggregator = config.aggregator.build(
        dataset_info=dataset_info,
        n_timesteps=config.n_forward_steps + stepper.n_ic_timesteps,
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(
        n_initial_conditions=data.n_initial_conditions,
        timestep=data.timestep,
        coords=data.coords,
        variable_metadata=variable_metadata,
    )

    timer.stop()
    logging.info("Starting inference")
    record_logs = get_record_to_wandb(label="inference")
    run_inference(
        predict=stepper.predict_paired,
        data=data,
        writer=writer,
        aggregator=aggregator,
        record_logs=record_logs,
    )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.flush()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop()

    timer.stop_outer("inference")
    total_steps = config.n_forward_steps * data.n_initial_conditions
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


def run_segmented_inference(config: InferenceConfig, segments: int):
    """Run inference in multiple segments.

    Args:
        config: inference configuration to be used for each individual segment. The
            provided initial condition configuration will only be used for the first
            segment.
        segments: total number of segments desired. Only missing segments will be run.

    Note:
        This is useful when running very long simulations or when saving a large
        amount of output data to disk. The simulation outputs will be split across
        multiple folders, each corresponding to one of the segments and labeled by
        the segment number.
    """
    logging.info(
        f"Starting segmented inference with {segments} segments. "
        f"Saving to {config.experiment_dir}."
    )
    config_copy = copy.deepcopy(config)
    original_wandb_name = os.environ.get("WANDB_NAME")
    for segment in range(segments):
        segment_label = f"segment_{segment:04d}"
        segment_dir = os.path.join(config.experiment_dir, segment_label)
        restart_path = os.path.join(segment_dir, "restart.nc")
        if os.path.exists(restart_path):
            logging.info(f"Skipping segment {segment} because it has already been run.")
        else:
            logging.info(f"Running segment {segment}.")
            config_copy.experiment_dir = segment_dir
            if original_wandb_name is not None:
                os.environ["WANDB_NAME"] = f"{original_wandb_name}-{segment_label}"
            with GlobalTimer():
                run_inference_from_config(config_copy)
        config_copy.initial_condition = InitialConditionConfig(
            path=restart_path, engine="netcdf4"
        )
