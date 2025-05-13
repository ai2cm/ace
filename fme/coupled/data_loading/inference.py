import dataclasses
from math import ceil

import torch

from fme.ace.data_loading.inference import (
    ExplicitIndices,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.core.dataset.getters import get_xarray_dataset
from fme.core.distributed import Distributed
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.config import CoupledDatasetConfig
from fme.coupled.data_loading.data_typing import (
    CoupledDataset,
    CoupledDatasetProperties,
)
from fme.coupled.requirements import CoupledDataRequirements


@dataclasses.dataclass
class InferenceDataLoaderConfig:
    """
    Configuration for inference data.

    This is like the `DataLoaderConfig` class, but with some additional
    constraints. During inference, we have only one batch, so the number of
    samples directly determines the size of that batch.

    Parameters:
        dataset: Configuration to define the dataset.
        start_indices: Configuration of the indices for initial conditions
            during inference. This can be a list of timestamps, a list of
            integer indices, or a slice configuration of the integer indices.
            Values following the initial condition will still come from
            the full dataset.
        num_data_workers: Number of parallel workers to use for data loading.
    """

    dataset: CoupledDatasetConfig
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList
    num_data_workers: int = 0

    def __post_init__(self):
        self._zarr_engine_used = any(
            ds.zarr_engine_used for ds in self.dataset.data_configs
        )

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any dataset uses the zarr engine.
        """
        return self._zarr_engine_used

    @property
    def n_initial_conditions(self) -> int:
        return self.start_indices.n_initial_conditions


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: InferenceDataLoaderConfig,
        total_coupled_steps: int,
        requirements: CoupledDataRequirements,
    ):
        ocean_reqs = requirements.ocean_requirements
        atmosphere_reqs = requirements.atmosphere_requirements
        ocean, ocean_properties = get_xarray_dataset(
            config.dataset.ocean,
            ocean_reqs.names,
            ocean_reqs.n_timesteps,
        )
        atmosphere, atmosphere_properties = get_xarray_dataset(
            config.dataset.atmosphere,
            atmosphere_reqs.names,
            atmosphere_reqs.n_timesteps,
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
        self._dataset = dataset
        self._properties = properties
        self._coupled_steps_in_memory = requirements.ocean_requirements.n_timesteps - 1
        self._total_coupled_steps = total_coupled_steps
        self._n_initial_conditions = config.n_initial_conditions

        if isinstance(config.start_indices, TimestampList):
            self._start_indices = config.start_indices.as_indices(dataset.all_ic_times)
        else:
            self._start_indices = config.start_indices.as_indices()

    def _get_batch_data(self, index) -> CoupledBatchData:
        dist = Distributed.get_instance()
        i_start = index * self._coupled_steps_in_memory
        samples = []
        for i_member in range(self._n_initial_conditions):
            # check if sample is one this local rank should process
            if i_member % dist.world_size != dist.rank:
                continue
            i_window_start = i_start + self._start_indices[i_member]
            samples.append(self._dataset[i_window_start])
        return CoupledBatchData.collate_fn(
            samples,
            horizontal_dims=list(self.properties.horizontal_coordinates.dims),
        )

    def __getitem__(self, index) -> CoupledBatchData:
        dist = Distributed.get_instance()
        result = self._get_batch_data(index)
        assert (
            result.ocean_data.time.shape[0]
            == self._n_initial_conditions // dist.world_size
        )
        return result

    def __len__(self) -> int:
        # The ceil is necessary so if the last batch is smaller
        # than the rest the ratio will be rounded up and the last batch
        # will be included in the loading
        return int(ceil(self._total_coupled_steps / self._coupled_steps_in_memory))

    @property
    def properties(self) -> CoupledDatasetProperties:
        return self._properties
