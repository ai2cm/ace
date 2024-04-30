from typing import Union

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fme.core.data_loading.config import (
    DataLoaderConfig,
    Slice,
    TimeSlice,
    XarrayDataConfig,
)
from fme.core.device import using_gpu
from fme.core.distributed import Distributed

from ._xarray import XarrayDataset, as_index_slice, get_datasets_at_path, subset_dataset
from .data_typing import Dataset, GriddedData
from .inference import InferenceDataLoaderConfig, InferenceDataset
from .requirements import DataRequirements
from .utils import BatchData


def get_ensemble_dataset(
    params: XarrayDataConfig,
    requirements: DataRequirements,
    subset: Union[Slice, TimeSlice],
) -> Dataset:
    """Returns a dataset that is a concatenation of the datasets for each
    ensemble member.
    """
    datasets = get_datasets_at_path(params, requirements, subset=subset)
    ensemble = torch.utils.data.ConcatDataset(datasets)
    ensemble.metadata = datasets[0].metadata  # type: ignore
    ensemble.area_weights = datasets[0].area_weights  # type: ignore
    ensemble.sigma_coordinates = datasets[0].sigma_coordinates  # type: ignore
    ensemble.horizontal_coordinates = datasets[0].horizontal_coordinates  # type: ignore
    return ensemble


def get_dataset(
    config: DataLoaderConfig,
    requirements: DataRequirements,
) -> Dataset:
    if config.data_type == "xarray":
        dataset = XarrayDataset(config.dataset, requirements)
        subset_slice = as_index_slice(config.subset, dataset)
        dataset = subset_dataset(dataset, subset_slice)
    elif config.data_type == "ensemble_xarray":
        return get_ensemble_dataset(config.dataset, requirements, config.subset)
    else:
        raise NotImplementedError(
            f"{config.data_type} does not have an implemented data loader"
        )
    return dataset


def get_data_loader(
    config: DataLoaderConfig,
    train: bool,
    requirements: DataRequirements,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        train: Whether to use the training or validation data.
        requirements: Data requirements for the model.
        window_time_slice: Time slice within each window to use for the data loader,
            if given the loader will only return data from this time slice.
            By default it will return the full windows.
    """
    dataset = get_dataset(config, requirements)
    dist = Distributed.get_instance()
    sampler = (
        DistributedSampler(dataset, shuffle=train) if dist.is_distributed() else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dist.local_batch_size(int(config.batch_size)),
        num_workers=config.num_data_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=using_gpu(),
        collate_fn=BatchData.from_sample_tuples,
    )

    if len(dataloader) == 0:
        raise ValueError(
            "No batches in dataloader: "
            f"{len(dataloader.dataset)} samples, {len(dataloader)} batches. "
            "Batch size is {dataloader.batch_size}"
        )

    return GriddedData(
        loader=dataloader,
        metadata=dataset.metadata,
        area_weights=dataset.area_weights,
        sampler=sampler,
        sigma_coordinates=dataset.sigma_coordinates,
        horizontal_coordinates=dataset.horizontal_coordinates,
    )


def get_inference_data(
    config: InferenceDataLoaderConfig,
    forward_steps_in_memory: int,
    requirements: DataRequirements,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        forward_steps_in_memory: Number of forward steps to keep in memory at once.
        requirements: Data requirements for the model.

    Returns:
        A data loader for inference with coordinates and metadata.
    """
    dataset = InferenceDataset(config, forward_steps_in_memory, requirements)
    # we roll our own batching in InferenceDataset, which is why batch_size=None below
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=config.num_data_workers,
        shuffle=False,
        pin_memory=using_gpu(),
    )
    return GriddedData(
        loader=loader,
        metadata=dataset.metadata,
        area_weights=dataset.area_weights,
        sigma_coordinates=dataset.sigma_coordinates,
        horizontal_coordinates=dataset.horizontal_coordinates,
    )
