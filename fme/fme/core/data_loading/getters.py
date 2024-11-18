import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch.utils.data
import xarray as xr
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.config import DataLoaderConfig, XarrayDataConfig
from fme.core.device import using_gpu
from fme.core.distributed import Distributed

from ._xarray import XarrayDataset, as_index_selection, transfer_properties
from .batch_data import GriddedData
from .data_typing import Dataset
from .inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceDataLoaderConfig,
    InferenceDataset,
)
from .requirements import DataRequirements

logger = logging.getLogger(__name__)


def _all_same(iterable, cmp=lambda x, y: x == y):
    it = iter(iterable)
    try:
        first = next(it)
    except StopIteration:
        return True
    return all(cmp(first, rest) for rest in it)


def get_xarray_dataset(
    config: XarrayDataConfig, requirements: DataRequirements
) -> XarrayDataset:
    dataset = XarrayDataset(config, requirements)
    index_slice = as_index_selection(config.subset, dataset)
    dataset = dataset.subset(index_slice)
    dataset.n_steps = dataset.dataset.n_steps
    return dataset


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig], requirements: DataRequirements
) -> List[XarrayDataset]:
    datasets = []
    for config in dataset_configs:
        dataset = get_xarray_dataset(config, requirements)
        datasets.append(dataset)
    return datasets


def validate_ensemble(datasets: List[Dataset], strict: bool = True):
    if not _all_same([d.variable_metadata for d in datasets]):
        if strict:
            raise ValueError("Metadata for each ensemble member should be the same.")
        else:
            warnings.warn(
                "Metadata for each ensemble member are not the same. You may be "
                "concatenating incompatible datasets."
            )
    sigma_coords = [d.sigma_coordinates for d in datasets]
    ak, bk = list(
        zip(*[(s.ak.cpu().numpy(), s.bk.cpu().numpy()) for s in sigma_coords])
    )
    if not (_all_same(ak, cmp=np.allclose) and _all_same(bk, cmp=np.allclose)):
        if strict:
            raise ValueError(
                "Sigma coordinates for each ensemble member should be the same."
            )
        else:
            warnings.warn(
                "Vertical coordinates for each ensemble member are not the same. You "
                "may be concatenating incompatible datasets."
            )


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
) -> torch.utils.data.ConcatDataset[XarrayDataset]:
    datasets = get_datasets(dataset_configs, requirements)
    validate_ensemble(datasets, strict=strict)

    ensemble = torch.utils.data.ConcatDataset(datasets)
    override: Dict[str, Any] = {
        "is_remote": any(d.is_remote for d in datasets),
    }

    try:
        timestep = datasets[0].timestep
        override["timestep"] = timestep
    except ValueError:
        logger.debug(
            "Timestep not found in dataset, skipping property inclusion for ensemble"
        )
        pass

    transfer_properties(datasets[0], ensemble, attr_override=override)
    return ensemble


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
    dataset = get_dataset(config.dataset, requirements, strict=config.strict_ensemble)
    dist = Distributed.get_instance()

    if dist.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else None

    if dataset.is_remote:
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    def collate_fn(samples):
        return BatchData.atmospheric_from_sample_tuples(
            samples,
            sigma_coordinates=dataset.sigma_coordinates,
            horizontal_dims=list(dataset.horizontal_coordinates.dims),
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
        variable_metadata=dataset.variable_metadata,
        sampler=sampler,
        sigma_coordinates=dataset.sigma_coordinates,
        timestep=dataset.timestep,
        horizontal_coordinates=dataset.horizontal_coordinates,
    )


def get_inference_data(
    config: InferenceDataLoaderConfig,
    forward_steps_in_memory: int,
    requirements: DataRequirements,
    surface_temperature_name: Optional[str] = None,
    ocean_fraction_name: Optional[str] = None,
) -> GriddedData:
    """
    Args:
        config: Parameters for the data loader.
        forward_steps_in_memory: Number of forward steps to keep in memory at once.
        requirements: Data requirements for the model.
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.

    Returns:
        A data loader for inference with coordinates and metadata.
    """
    dataset = InferenceDataset(
        config,
        forward_steps_in_memory,
        requirements,
        surface_temperature_name,
        ocean_fraction_name,
    )

    if dataset.is_remote:
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
    gridded_data = GriddedData(
        loader=loader,
        variable_metadata=dataset.variable_metadata,
        sigma_coordinates=dataset.sigma_coordinates,
        timestep=dataset.timestep,
        horizontal_coordinates=dataset.horizontal_coordinates,
    )

    return gridded_data


def get_forcing_data(
    config: ForcingDataLoaderConfig,
    forward_steps_in_memory: int,
    requirements: DataRequirements,
    initial_times: xr.DataArray,
    surface_temperature_name: Optional[str] = None,
    ocean_fraction_name: Optional[str] = None,
) -> GriddedData:
    """Return a GriddedData loader for forcing data only. This function determines the
    start indices for the forcing data based on the initial times provided.

    Args:
        config: Parameters for the forcing data loader.
        forward_steps_in_memory: Number of forward steps to provide per window of
            forcing data that will be returned by loader.
        requirements: Data requirements for the forcing data.
        initial_times: Desired initial times for the forcing data. This must be a 1D
            data array, whose length determines the ensemble size.
        surface_temperature_name: Name of the surface temperature variable. Can be
            set to None if no ocean temperature prescribing is being used.
        ocean_fraction_name: Name of the ocean fraction variable. Can be set to None
            if no ocean temperature prescribing is being used.

    Returns:
        A data loader for forcing data with coordinates and metadata.
    """
    if initial_times.shape[1] != 1:
        raise NotImplementedError("code assumes initial times only has 1 timestep")
    available_times = XarrayDataset(config.dataset, requirements).all_times
    start_time_indices = []
    for time in initial_times.values[:, 0]:
        start_time_indices.append(available_times.get_loc(time))
    inference_config = config.build_inference_config(
        start_indices=ExplicitIndices(start_time_indices)
    )
    return get_inference_data(
        inference_config,
        forward_steps_in_memory,
        requirements,
        surface_temperature_name,
        ocean_fraction_name,
    )
