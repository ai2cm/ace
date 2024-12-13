import logging
from typing import Optional, Union

import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.requirements import PrognosticStateDataRequirements
from fme.core.dataset.getters import get_dataset
from fme.core.dataset.requirements import DataRequirements
from fme.core.dataset.xarray import XarrayDataset
from fme.core.device import using_gpu
from fme.core.distributed import Distributed

from .batch_data import GriddedData, InferenceGriddedData, PrognosticState
from .config import DataLoaderConfig
from .inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)

logger = logging.getLogger(__name__)


def get_data_loader(
    config: DataLoaderConfig,
    train: bool,
    requirements: DataRequirements,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        train: Whether loader is intended for training or validation data; if True,
            then data will be shuffled.
        requirements: Data requirements for the model.
    """
    dataset, properties = get_dataset(
        config.dataset, requirements, strict=config.strict_ensemble
    )
    dist = Distributed.get_instance()

    if dist.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else None

    if properties.is_remote:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    def collate_fn(samples):
        return BatchData.from_sample_tuples(
            samples,
            horizontal_dims=list(properties.horizontal_coordinates.dims),
        )

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
        collate_fn=collate_fn,
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
        loader=dataloader,
        properties=properties,
        sampler=sampler,
    )


def get_inference_data(
    config: InferenceDataLoaderConfig,
    total_forward_steps: int,
    window_requirements: DataRequirements,
    initial_condition: Union[PrognosticState, PrognosticStateDataRequirements],
    surface_temperature_name: Optional[str] = None,
    ocean_fraction_name: Optional[str] = None,
) -> InferenceGriddedData:
    """
    Args:
        config: Parameters for the data loader.
        total_forward_steps: Total number of forward steps to take over the course of
            inference.
        window_requirements: Data requirements for the model.
        initial_condition: Initial condition for the inference, or a requirements object
            specifying how to extract the initial condition from the first batch of
            data
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.

    Returns:
        A data loader for inference with coordinates and metadata.
    """
    dataset = InferenceDataset(
        config,
        total_forward_steps,
        window_requirements,
        surface_temperature_name,
        ocean_fraction_name,
    )
    properties = dataset.properties

    if properties.is_remote:
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
    gridded_data = InferenceGriddedData(
        loader=loader,
        initial_condition=initial_condition,
        properties=properties,
    )

    return gridded_data


def get_forcing_data(
    config: ForcingDataLoaderConfig,
    total_forward_steps: int,
    window_requirements: DataRequirements,
    initial_condition: PrognosticState,
    surface_temperature_name: Optional[str] = None,
    ocean_fraction_name: Optional[str] = None,
) -> InferenceGriddedData:
    """Return a GriddedData loader for forcing data based on the initial condition.
    This function determines the start indices for the forcing data based on the initial
    time in the provided initial condition.

    Args:
        config: Parameters for the forcing data loader.
        total_forward_steps: Total number of forward steps to take over the course of
            inference.
        window_requirements: Data requirements for the forcing data.
        initial_condition: Initial condition for the inference.
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.

    Returns:
        A data loader for forcing data with coordinates and metadata.
    """
    initial_time = initial_condition.as_batch_data().time
    if initial_time.shape[1] != 1:
        raise NotImplementedError("code assumes initial time only has 1 timestep")
    available_times = XarrayDataset(config.dataset, window_requirements).all_times
    start_time_indices = []
    for time in initial_time.values[:, 0]:
        start_time_indices.append(available_times.get_loc(time))
    inference_config = config.build_inference_config(
        start_indices=ExplicitIndices(start_time_indices)
    )
    return get_inference_data(
        config=inference_config,
        total_forward_steps=total_forward_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition,
        surface_temperature_name=surface_temperature_name,
        ocean_fraction_name=ocean_fraction_name,
    )
