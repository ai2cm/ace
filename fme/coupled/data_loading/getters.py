import logging
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from fme.core.dataset.getters import get_xarray_dataset
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
from fme.coupled.data_loading.gridded_data import GriddedData, InferenceGriddedData
from fme.coupled.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceDataset,
)
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledPrognosticStateDataRequirements,
)


def get_dataset(
    config: CoupledDatasetConfig, requirements: CoupledDataRequirements
) -> Tuple[CoupledDataset, CoupledDatasetProperties]:
    ocean_reqs = requirements.ocean_requirements
    atmosphere_reqs = requirements.atmosphere_requirements
    ocean, ocean_properties = get_xarray_dataset(
        config.ocean, ocean_reqs.names, ocean_reqs.n_timesteps
    )
    atmosphere, atmosphere_properties = get_xarray_dataset(
        config.atmosphere, atmosphere_reqs.names, atmosphere_reqs.n_timesteps
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
) -> Tuple[torch.utils.data.ConcatDataset[CoupledDataset], CoupledDatasetProperties]:
    datasets = []
    properties: Optional[CoupledDatasetProperties] = None
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


def get_data_loader(
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

    if properties.is_remote:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    def collate_fn(samples: List[CoupledDatasetItem]) -> CoupledBatchData:
        return CoupledBatchData.collate_fn(
            samples,
            horizontal_dims=list(properties.atmosphere.horizontal_coordinates.dims),
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
    total_coupled_steps: int,
    window_requirements: CoupledDataRequirements,
    initial_condition: Union[
        CoupledPrognosticState, CoupledPrognosticStateDataRequirements
    ],
) -> InferenceGriddedData:
    dataset = InferenceDataset(
        config,
        total_coupled_steps,
        window_requirements,
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
    inference_data = InferenceGriddedData(
        loader=loader,
        initial_condition=initial_condition,
        properties=properties,
    )

    return inference_data
