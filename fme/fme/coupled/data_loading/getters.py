from typing import Any, Dict, List

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from fme.core.data_loading._xarray import transfer_properties
from fme.core.data_loading.batch_data import CPU
from fme.core.data_loading.getters import get_xarray_dataset, validate_ensemble
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledGriddedData,
)
from fme.coupled.data_loading.config import CoupledDataConfig, CoupledDataLoaderConfig
from fme.coupled.data_loading.data_typing import CoupledDataset, CoupledDatasetItem
from fme.coupled.data_loading.requirements import CoupledDataRequirements


def get_coupled_dataset(
    config: CoupledDataConfig, requirements: CoupledDataRequirements
) -> CoupledDataset:
    ocean_reqs = requirements.ocean_requirements
    atmosphere_reqs = requirements.atmosphere_requirements
    ocean = get_xarray_dataset(config.ocean, ocean_reqs)
    atmosphere = get_xarray_dataset(config.atmosphere, atmosphere_reqs)
    dataset = CoupledDataset(
        ocean,
        atmosphere,
        requirements.ocean_timestep,
        requirements.atmosphere_timestep,
    )
    return dataset


def get_coupled_data_loader(
    config: CoupledDataLoaderConfig,
    train: bool,
    requirements: CoupledDataRequirements,
) -> CoupledGriddedData:
    """
    Args:
        config: Parameters for the data loader.
        train: Whether loader is intended for training or validation data; if True,
            then data will be shuffled.
        requirements: Data requirements for the model.
    """
    datasets: List[CoupledDataset] = [
        get_coupled_dataset(
            coupled_data_config,
            requirements,
        )
        for i, coupled_data_config in enumerate(config.dataset)
    ]
    validate_ensemble(datasets, strict=config.strict_ensemble)
    dataset = torch.utils.data.ConcatDataset(datasets)
    dist = Distributed.get_instance()
    if dist.is_distributed():
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else None

    if any([ds.is_remote for ds in datasets]):
        # GCSFS and S3FS are not fork-safe, so we need to use forkserver
        mp_context = "forkserver"
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    # TODO: this needs to be replaced with a pickleable collate function
    def collate_fn(samples: List[CoupledDatasetItem]) -> CoupledBatchData[CPU]:
        return CoupledBatchData.collate_fn(
            samples,
            sigma_coordinates=datasets[0].sigma_coordinates,
            horizontal_dims=list(datasets[0].horizontal_coordinates.dims),
        )

    batch_size = dist.local_batch_size(int(config.batch_size))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config.num_data_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=using_gpu(),
        collate_fn=collate_fn,
        prefetch_factor=config.prefetch_factor,
        multiprocessing_context=mp_context,
        persistent_workers=persistent_workers,
    )

    if len(dataloader) == 0:
        raise ValueError(
            "No batches in dataloader: "
            f"{len(dataloader.dataset)} samples, {len(dataloader)} batches. "
            f"Batch size is {dataloader.batch_size}"
        )

    override: Dict[str, Any] = {
        "is_remote": any(d.is_remote for d in datasets),
    }
    transfer_properties(datasets[0], dataset, attr_override=override)
    dataset.n_forward_steps = datasets[0].n_forward_steps

    return CoupledGriddedData(
        loader=dataloader,
        metadata=dataset.metadata,
        sampler=sampler,
        sigma_coordinates=dataset.sigma_coordinates,
        timestep=max(requirements.timesteps),
        horizontal_coordinates=dataset.horizontal_coordinates,
    )
