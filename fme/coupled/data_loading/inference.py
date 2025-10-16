import dataclasses
import logging
from math import ceil

import torch
import xarray as xr

from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.requirements import DataRequirements
from fme.core.dataset.dummy import DummyDataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.distributed import Distributed
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.config import (
    CoupledDatasetConfig,
    CoupledDatasetWithOptionalOceanConfig,
)
from fme.coupled.data_loading.data_typing import (
    CoupledDataset,
    CoupledDatasetProperties,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
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

    dataset: CoupledDatasetConfig | CoupledDatasetWithOptionalOceanConfig
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList
    num_data_workers: int = 0

    def __post_init__(self):
        self._zarr_engine_used = any(
            ds.zarr_engine_used for ds in self.dataset.data_configs if ds is not None
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
        dataset_info: CoupledDatasetInfo | None = None,
        initial_time: xr.DataArray | None = None,
    ):
        ocean_reqs = requirements.ocean_requirements
        atmosphere_reqs = requirements.atmosphere_requirements
        ocean: torch.utils.data.Dataset
        atmosphere: torch.utils.data.Dataset
        if config.dataset.ocean is not None:
            ocean, ocean_properties = config.dataset.ocean.build(
                ocean_reqs.names, ocean_reqs.n_timesteps
            )
        else:
            assert dataset_info is not None
            ocean, ocean_properties = _make_dummy_ocean_forcing(
                dataset_info=dataset_info,
                initial_time=initial_time,
                total_coupled_steps=total_coupled_steps,
                ocean_reqs=ocean_reqs,
            )
        all_ic_times = ocean.sample_start_times
        ocean_properties = self._update_ocean_mask(ocean_properties, dataset_info)
        atmosphere, atmosphere_properties = config.dataset.atmosphere.build(
            atmosphere_reqs.names, atmosphere_reqs.n_timesteps
        )
        properties = CoupledDatasetProperties(
            all_ic_times, ocean_properties, atmosphere_properties
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

    def _update_ocean_mask(
        self,
        ocean_properties: DatasetProperties,
        dataset_info: CoupledDatasetInfo | None,
    ) -> DatasetProperties:
        if dataset_info is None:
            return ocean_properties
        ocean_mask_is_empty = not ocean_properties.mask_provider.masks
        identical_masks = (
            len(ocean_properties.mask_provider.masks) > 0
            and len(dataset_info.ocean.mask_provider.masks) > 0
            and ocean_properties.mask_provider == dataset_info.ocean.mask_provider
        )
        if ocean_mask_is_empty or identical_masks:
            ocean_properties.update_mask_provider(dataset_info.ocean.mask_provider)
        else:
            logging.warning(
                "Not updating ocean mask provider from dataset info in the checkpoint"
                "because the existing mask provider is not empty or the masks are not"
                "identical."
            )
        return ocean_properties

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
            ocean_horizontal_dims=list(
                self.properties.horizontal_coordinates.ocean.dims
            ),
            atmosphere_horizontal_dims=list(
                self.properties.horizontal_coordinates.atmosphere.dims
            ),
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


@dataclasses.dataclass
class CoupledForcingDataLoaderConfig:
    atmosphere: ForcingDataLoaderConfig
    ocean: ForcingDataLoaderConfig | None = None
    num_data_workers: int = 0

    def build_inference_config(
        self,
        start_indices: ExplicitIndices,
    ):
        if self.ocean is None:
            return InferenceDataLoaderConfig(
                dataset=CoupledDatasetWithOptionalOceanConfig(
                    atmosphere=self.atmosphere.dataset,
                ),
                start_indices=start_indices,
                num_data_workers=self.num_data_workers,
            )
        return InferenceDataLoaderConfig(
            dataset=CoupledDatasetConfig(
                atmosphere=self.atmosphere.dataset,
                ocean=self.ocean.dataset,
            ),
            start_indices=start_indices,
            num_data_workers=self.num_data_workers,
        )


def _make_dummy_ocean_forcing(
    dataset_info: CoupledDatasetInfo,
    initial_time: xr.DataArray,
    total_coupled_steps: int,
    ocean_reqs: DataRequirements,
) -> tuple[torch.utils.data.Dataset, DatasetProperties]:
    ocean_property = DatasetProperties(
        variable_metadata=dict(dataset_info.ocean.variable_metadata),
        vertical_coordinate=dataset_info.ocean.vertical_coordinate,
        horizontal_coordinates=dataset_info.ocean.horizontal_coordinates,
        mask_provider=dataset_info.ocean.mask_provider,
        timestep=dataset_info.ocean.timestep,
        is_remote=False,
        all_labels=set(),
    )
    ts = dataset_info.ocean.timestep
    ocean = DummyDataset(
        start_time=initial_time.squeeze().values.flat[0],
        end_time=initial_time.squeeze().values.flat[-1] + ts * total_coupled_steps,
        timestep=ts,
        n_timesteps=ocean_reqs.n_timesteps,
        horizontal_coordinates=dataset_info.ocean.horizontal_coordinates,
    )
    return ocean, ocean_property
