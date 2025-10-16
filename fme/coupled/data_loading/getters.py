import datetime
import logging
import warnings
from collections.abc import Sequence

import torch
import torch.utils.data
import xarray as xr
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from fme.core.dataset.merged import MergeNoConcatDatasetConfig
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.coupled.data_loading.batch_data import CoupledBatchData, CoupledPrognosticState
from fme.coupled.data_loading.config import (
    CoupledDataLoaderConfig,
    CoupledDatasetConfig,
)
from fme.coupled.data_loading.data_typing import (
    CoupledDataset,
    CoupledDatasetItem,
    CoupledDatasetProperties,
)
from fme.coupled.data_loading.dataloader import CoupledDataLoader
from fme.coupled.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.coupled.data_loading.inference import (
    CoupledForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledPrognosticStateDataRequirements,
)

from .inference import ExplicitIndices


class CollateFn:
    def __init__(
        self, ocean_horizontal_dims: list[str], atmosphere_horizontal_dims: list[str]
    ):
        self.ocean_horizontal_dims = ocean_horizontal_dims
        self.atmosphere_horizontal_dims = atmosphere_horizontal_dims

    def __call__(self, samples: list[CoupledDatasetItem]) -> CoupledBatchData:
        return CoupledBatchData.collate_fn(
            samples,
            ocean_horizontal_dims=self.ocean_horizontal_dims,
            atmosphere_horizontal_dims=self.atmosphere_horizontal_dims,
        )


def get_dataset(
    config: CoupledDatasetConfig, requirements: CoupledDataRequirements
) -> tuple[CoupledDataset, CoupledDatasetProperties]:
    ocean_reqs = requirements.ocean_requirements
    atmosphere_reqs = requirements.atmosphere_requirements
    ocean: torch.utils.data.Dataset
    atmosphere: torch.utils.data.Dataset
    ocean, ocean_properties = config.ocean.build(
        ocean_reqs.names, ocean_reqs.n_timesteps
    )
    atmosphere, atmosphere_properties = config.atmosphere.build(
        atmosphere_reqs.names, atmosphere_reqs.n_timesteps
    )
    properties = CoupledDatasetProperties(
        ocean.sample_start_times, ocean_properties, atmosphere_properties
    )
    dataset = CoupledDataset(
        ocean=ocean,
        atmosphere=atmosphere,
        properties=properties,
        n_steps_fast=requirements.n_steps_fast,
    )
    return dataset, properties


def get_datasets(
    configs: Sequence[CoupledDatasetConfig],
    requirements: CoupledDataRequirements,
    strict: bool = True,
) -> tuple[torch.utils.data.ConcatDataset[CoupledDataset], CoupledDatasetProperties]:
    datasets = []
    properties: CoupledDatasetProperties | None = None
    for coupled_data_config in configs:
        ds, prop = get_dataset(coupled_data_config, requirements)
        datasets.append(ds)
        if properties is None:
            properties = prop
        elif not strict:
            try:
                properties.update(prop)
            except ValueError as e:
                warnings.warn(
                    f"Metadata for each ensemble member are not the same: {e}"
                )
        else:
            properties.update(prop)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")
    dataset = torch.utils.data.ConcatDataset(datasets)
    return dataset, properties


def get_gridded_data(
    config: CoupledDataLoaderConfig,
    train: bool,
    requirements: CoupledDataRequirements,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        train: Whether loader is intended for training or validation data; if True,
            then data will be shuffled.
        requirements: Data requirements for the model.
    """
    dataset, properties = get_datasets(
        config.dataset, requirements, strict=config.strict_ensemble
    )
    dist = Distributed.get_instance()
    if dist.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else None

    if config.zarr_engine_used:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    batch_size = dist.local_batch_size(int(config.batch_size))

    if config.prefetch_factor is None:
        # DataLoader default is not None so we must leave it unset
        kwargs = {}
    else:
        kwargs = {"prefetch_factor": config.prefetch_factor}

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_data_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=using_gpu(),
        collate_fn=CollateFn(
            ocean_horizontal_dims=list(properties.ocean.horizontal_coordinates.dims),
            atmosphere_horizontal_dims=list(
                properties.atmosphere.horizontal_coordinates.dims
            ),
        ),
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
        **kwargs,
    )

    if len(dataloader) == 0:
        raise ValueError(
            "No batches in dataloader: "
            f"{len(dataloader.dataset)} samples, {len(dataloader)} batches. "
            f"Batch size is {dataloader.batch_size}"
        )

    return GriddedData(
        loader=CoupledDataLoader(
            dataloader,
            sampler=sampler,
            dataset=dataset,
        ),
        properties=properties,
        sampler=sampler,
    )


def get_inference_data(
    config: InferenceDataLoaderConfig,
    total_coupled_steps: int,
    window_requirements: CoupledDataRequirements,
    initial_condition: CoupledPrognosticState | CoupledPrognosticStateDataRequirements,
    dataset_info: CoupledDatasetInfo | None = None,
) -> InferenceGriddedData:
    initial_time = None
    if isinstance(initial_condition, CoupledPrognosticState):
        initial_time = (
            initial_condition.ocean_data.as_batch_data().time
        )  # used only if no ocean forcing is specified
    dataset = InferenceDataset(
        config,
        total_coupled_steps,
        window_requirements,
        dataset_info,
        initial_time,
    )
    properties = dataset.properties

    if config.zarr_engine_used:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        # persist workers since startup is slow
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    logging.info(f"Multiprocessing inference context: {mp_context or 'fork'}")

    # we roll our own batching in InferenceDataset, which is why batch_size=None below
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_data_workers,
        shuffle=False,
        pin_memory=using_gpu(),
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
    )
    inference_data = InferenceGriddedData(
        loader=loader,
        initial_condition=initial_condition,
        properties=properties,
    )

    return inference_data


def _make_available_times_from_initial_time(
    initial_time: xr.DataArray,
    total_coupled_steps: int,
    time_step: datetime.timedelta,
) -> xr.CFTimeIndex:
    first_time = sorted(initial_time)[0]
    last_time = sorted(initial_time)[-1] + total_coupled_steps * time_step
    available_times = xr.date_range(
        start=first_time.values[0],
        end=last_time.values[0],
        freq=time_step,
        calendar=first_time.values[0].calendar,
        use_cftime=True,
    )
    return available_times


def get_forcing_data(
    config: CoupledForcingDataLoaderConfig,
    total_coupled_steps: int,
    window_requirements: CoupledDataRequirements,
    initial_condition: CoupledPrognosticState,
    dataset_info: CoupledDatasetInfo | None = None,
) -> InferenceGriddedData:
    """Return a GriddedData loader for forcing data based on the initial condition.
    This function determines the start indices for the forcing data based on the initial
    time in the provided initial condition.

    Args:
        config: Parameters for the forcing data loader.
        total_coupled_steps: Total number of forward steps to take over the course of
            inference.
        window_requirements: Data requirements for the forcing data.
        initial_condition: Initial condition for the inference.
        dataset_info: Dataset info loaded from the stepper.

    Returns:
        A data loader for forcing data with coordinates and metadata.
    """
    initial_time = initial_condition.ocean_data.as_batch_data().time
    if initial_time.shape[1] != 1:
        raise NotImplementedError("code assumes initial time only has 1 timestep")
    if config.ocean is None:
        available_times = _make_available_times_from_initial_time(
            initial_time, total_coupled_steps, window_requirements.ocean_timestep
        )
    else:
        if isinstance(config.ocean.dataset, XarrayDataConfig):
            available_times = XarrayDataset(
                config.ocean.dataset,
                window_requirements.ocean_requirements.names,
                window_requirements.ocean_requirements.n_timesteps,
            ).all_times
        elif isinstance(config.ocean.dataset, MergeNoConcatDatasetConfig):
            # Some forcing variables may not be in the first dataset,
            # use an empty data requirements to get all times
            if isinstance(config.ocean.dataset.merge[0], XarrayDataConfig):
                available_times = XarrayDataset(
                    config.ocean.dataset.merge[0],
                    names=[],
                    n_timesteps=window_requirements.ocean_requirements.n_timesteps,
                ).all_times
            else:
                raise ValueError("Forcing data cannot be concatenated.")
    start_time_indices = []
    for time in initial_time.values[:, 0]:
        start_time_indices.append(available_times.get_loc(time))
    inference_config = config.build_inference_config(
        start_indices=ExplicitIndices(start_time_indices),
    )
    return get_inference_data(
        config=inference_config,
        total_coupled_steps=total_coupled_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition,
        dataset_info=dataset_info,
    )
