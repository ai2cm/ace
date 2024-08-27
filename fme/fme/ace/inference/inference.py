import argparse
import dataclasses
import logging
import os
import time
from typing import Literal, Optional, Sequence, Tuple, Union

import dacite
import torch
import xarray as xr
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.inference.data_writer import DataWriter, DataWriterConfig
from fme.ace.inference.loop import run_inference, write_reduced_metrics
from fme.core import SingleModuleStepper
from fme.core.aggregator.inference import InferenceAggregatorConfig
from fme.core.data_loading.data_typing import GriddedData, SigmaCoordinates
from fme.core.data_loading.getters import get_forcing_data
from fme.core.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.core.dicts import to_flat_dict
from fme.core.logging_utils import LoggingConfig
from fme.core.ocean import OceanConfig
from fme.core.stepper import SingleModuleStepperConfig
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB

from .evaluator import load_stepper, load_stepper_config, validate_time_coarsen_config

StartIndices = Union[InferenceInitialConditionIndices, ExplicitIndices, TimestampList]


@dataclasses.dataclass
class InitialConditionConfig:
    """
    Configuration for initial conditions.

    .. note::
        The data specified under path should contain a time dimension of at least
        length 1. If multiple times are present in the dataset specified by `path`,
        the inference will start an ensemble simulation using each IC along a
        leading sample dimension. Specific times can be selected from the dataset
        by using `start_indices`.

    Attributes:
        path: The path to the initial conditions dataset.
        engine: The engine used to open the dataset.
        start_indices: optional specification of the subset of
            initial conditions to use.
    """

    path: str
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"
    start_indices: Optional[StartIndices] = None

    def get_dataset(self) -> xr.Dataset:
        ds = xr.open_dataset(self.path, engine=self.engine, use_cftime=True)
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
) -> Tuple[TensorMapping, xr.DataArray]:
    """Given a dataset, extract a mapping of variables to tensors.
    and the time coordinate corresponding to the initial conditions.

    Args:
        ds: Dataset containing initial condition data. Must include prognostic_names
            as variables, and they must each have shape (n_samples, n_lat, n_lon).
            Dataset must also include a 'time' variable with length n_samples.
        prognostic_names: Names of prognostic variables to extract from the dataset.

    Returns:
        A mapping of variable names to tensors and the time coordinate.
    """
    initial_condition = {}
    for name in prognostic_names:
        if len(ds[name].shape) != 3:
            raise ValueError(
                f"Initial condition variables {name} must have shape "
                f"(n_samples, n_lat, n_lon). Got shape {ds[name].shape}."
            )
        n_samples = ds[name].shape[0]
        initial_condition[name] = torch.tensor(ds[name].values).to(fme.get_device())
    if "time" not in ds:
        raise ValueError("Initial condition dataset must have a 'time' variable.")
    initial_times = ds.time
    if len(initial_times) != n_samples:
        raise ValueError(
            "Length of 'time' variable must match first dimension of variables "
            f"in initial condition dataset. Got {len(initial_times)} and {n_samples}."
        )
    return initial_condition, initial_times


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Attributes:
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
        ocean: Ocean configuration for running inference with a
            different one than what is used in training.
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
        self, area: Optional[torch.Tensor], sigma_coordinates: SigmaCoordinates
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
    ) -> DataWriter:
        return self.data_writer.build(
            experiment_dir=self.experiment_dir,
            n_samples=data.loader.dataset.n_samples,
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
        data_class=InferenceConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    run_inference_from_config(config)


def run_inference_from_config(config: InferenceConfig):
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
    data_requirements = stepper_config.get_forcing_data_requirements(
        n_forward_steps=config.n_forward_steps
    )
    logging.info("Loading initial condition data")
    initial_condition, initial_times = get_initial_condition(
        config.initial_condition.get_dataset(), stepper_config.prognostic_names
    )
    logging.info("Initializing forcing data loaded")
    data = get_forcing_data(
        config.forcing_loader,
        config.forward_steps_in_memory,
        data_requirements,
        initial_times,
    )

    stepper = config.load_stepper(
        data.horizontal_coordinates.area_weights,
        sigma_coordinates=data.sigma_coordinates.to(fme.get_device()),
    )
    if stepper.timestep != data.timestep:
        raise ValueError(
            f"Timestep of the loaded stepper, {stepper.timestep}, does not "
            f"match that of the forcing data, {data.timestep}."
        )

    aggregator = config.aggregator.build(
        gridded_operations=data.gridded_operations,
        sigma_coordinates=data.sigma_coordinates,
        timestep=data.timestep,
        n_timesteps=config.n_forward_steps + 1,
        metadata=data.metadata,
    )

    writer = config.get_data_writer(data, stepper.prognostic_names)

    logging.info("Starting inference")
    timers = run_inference(
        stepper=stepper,
        initial_condition=initial_condition,
        forcing_data=data,
        writer=writer,
        aggregator=aggregator,
    )

    final_flush_start_time = time.time()
    logging.info("Starting final flush of data writer")
    writer.flush()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    write_reduced_metrics(aggregator, data.coords, config.experiment_dir)
    final_flush_duration = time.time() - final_flush_start_time
    logging.info(f"Final writer flush duration: {final_flush_duration:.2f} seconds")
    timers["final_writer_flush"] = final_flush_duration

    duration = time.time() - start_time
    total_steps = config.n_forward_steps * data.loader.dataset.n_samples
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)
    args = parser.parse_args()
    main(yaml_config=args.yaml_config)
