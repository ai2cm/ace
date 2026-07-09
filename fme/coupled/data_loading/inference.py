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
from fme.core.dataset.merged import MergedXarrayDataset
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarraySubset
from fme.core.distributed import Distributed
from fme.core.typing_ import Slice
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.config import (
    CoupledAtmosphereIceOceanDatasetConfig,
    CoupledDatasetConfig,
    CoupledDatasetWithOptionalOceanConfig,
    CoupledIceAtmosphereDatasetConfig,
    CoupledIceOceanDatasetConfig,
    build_coupled_dataset_config,
)
from fme.coupled.data_loading.data_typing import (
    CoupledDataset,
    CoupledDatasetProperties,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.requirements import CoupledDataRequirements

ComponentDatasetType = XarraySubset | MergedXarrayDataset


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

    dataset: (
        CoupledDatasetWithOptionalOceanConfig
        | CoupledDatasetConfig
        | CoupledIceOceanDatasetConfig
        | CoupledIceAtmosphereDatasetConfig
        | CoupledAtmosphereIceOceanDatasetConfig
    )
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList
    num_data_workers: int = 0

    def __post_init__(self):
        self._zarr_engine_used = any(
            ds.zarr_engine_used for ds in self.dataset.data_configs if ds is not None
        )
        for ds in self.dataset.data_configs:
            if ds is not None and ds.subset != Slice(None, None, None):
                raise ValueError(
                    "'subset' cannot be used in dataset configs during inference."
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
        all_reqs = {
            "ocean": requirements.ocean_requirements,
            "ice": requirements.ice_requirements,
            "atmosphere": requirements.atmosphere_requirements,
        }

        # The "anchor" is the slow/outer component that all others are aligned to.
        # Each config class declares its anchor via anchor_component_name.
        anchor_name = config.dataset.anchor_component_name
        anchor_config = getattr(config.dataset, anchor_name)
        anchor_reqs = all_reqs[anchor_name]

        built: dict[
            str, tuple[ComponentDatasetType | DummyDataset, DatasetProperties]
        ] = {}

        # Build anchor component (real dataset or dummy)
        if anchor_config is not None:
            assert anchor_reqs is not None
            if anchor_name == "ice" and initial_time is not None:
                first_ic_time = sorted(initial_time)[0].values[0]
                anchor_config.update_subset(TimeSlice(start_time=first_ic_time))
            built[anchor_name] = anchor_config.build(
                anchor_reqs.names, anchor_reqs.n_timesteps_schedule
            )
        else:
            assert dataset_info is not None
            assert anchor_reqs is not None
            make_dummy = (
                _make_dummy_ocean_forcing
                if anchor_name == "ocean"
                else _make_dummy_ice_forcing
            )
            built[anchor_name] = make_dummy(
                dataset_info, initial_time, total_coupled_steps, anchor_reqs
            )

        anchor_ds, anchor_properties = built[anchor_name]

        # Apply ocean-specific mask update
        if anchor_name == "ocean":
            anchor_properties = self._update_ocean_mask(anchor_properties, dataset_info)
            built[anchor_name] = (anchor_ds, anchor_properties)

        # Build remaining components, aligning each to the anchor's start time
        for comp_name, comp_reqs in all_reqs.items():
            if comp_name == anchor_name:
                continue
            comp_config = getattr(config.dataset, comp_name)
            if comp_config is None:
                continue
            assert comp_reqs is not None
            comp_config.update_subset(TimeSlice(start_time=anchor_ds.first_time))
            built[comp_name] = comp_config.build(
                comp_reqs.names, comp_reqs.n_timesteps_schedule
            )

        ocean = built.get("ocean", (None, None))[0]
        ocean_properties = built.get("ocean", (None, None))[1]
        ice = built.get("ice", (None, None))[0]
        ice_properties = built.get("ice", (None, None))[1]
        atmosphere = built.get("atmosphere", (None, None))[0]
        atmosphere_properties = built.get("atmosphere", (None, None))[1]

        properties = CoupledDatasetProperties(
            ocean=ocean_properties,
            atmosphere=atmosphere_properties,
            ice=ice_properties,
        )

        dataset = CoupledDataset(
            ocean=ocean,
            atmosphere=atmosphere,
            ice=ice,
            properties=properties,
            n_steps_fast=requirements.n_steps_fast,
        )

        self._dataset = dataset
        self._properties = properties
        assert anchor_reqs is not None
        self._coupled_steps_in_memory = (
            anchor_reqs.n_timesteps_schedule.get_value(0) - 1
        )
        self._total_coupled_steps = total_coupled_steps
        self._n_initial_conditions = config.n_initial_conditions

        if isinstance(config.start_indices, TimestampList):
            self._start_indices = config.start_indices.as_indices(dataset.all_ic_times)
        else:
            self._start_indices = config.start_indices.as_indices()

        self._dataset.validate_inference_length(
            max_start_index=max(self._start_indices),
            max_window_len=self._total_coupled_steps + 1,
        )

    def _update_ocean_mask(
        self,
        ocean_properties: DatasetProperties,
        dataset_info: CoupledDatasetInfo | None,
    ) -> DatasetProperties:
        if dataset_info is None:
            return ocean_properties
        assert dataset_info.ocean is not None
        ocean_mask_is_empty = not ocean_properties.spatial_mask_provider.masks
        identical_masks = (
            len(ocean_properties.spatial_mask_provider.masks) > 0
            and len(dataset_info.ocean.spatial_mask_provider.masks) > 0
            and ocean_properties.spatial_mask_provider
            == dataset_info.ocean.spatial_mask_provider
        )
        if ocean_mask_is_empty or identical_masks:
            ocean_properties.update_spatial_mask_provider(
                dataset_info.ocean.spatial_mask_provider
            )
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

        component_horizontal_dims = {
            name: list(hcoord.dims)
            for name, hcoord in (
                self.properties.horizontal_coordinates._components.items()
            )
        }
        return CoupledBatchData.collate_fn(
            samples,
            component_horizontal_dims=component_horizontal_dims,
        )

    def __getitem__(self, index) -> CoupledBatchData:
        dist = Distributed.get_instance()
        result = self._get_batch_data(index)
        # Validate batch dimension using the first available component
        for comp_data in result._components.values():
            assert (
                comp_data.time.shape[0] == self._n_initial_conditions // dist.world_size
            )
            break
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
    atmosphere: ForcingDataLoaderConfig | None = None
    ice: ForcingDataLoaderConfig | None = None
    ocean: ForcingDataLoaderConfig | None = None
    num_data_workers: int = 0

    def build_inference_config(
        self,
        start_indices: ExplicitIndices,
    ):
        ice = None
        ocean = None
        atmosphere = None
        if self.atmosphere is not None:
            atmosphere = self.atmosphere.dataset
        if self.ice is not None:
            ice = self.ice.dataset
        if self.ocean is not None:
            ocean = self.ocean.dataset

        return InferenceDataLoaderConfig(
            dataset=build_coupled_dataset_config(
                atmosphere=atmosphere,
                ice=ice,
                ocean=ocean,
            ),
            start_indices=start_indices,
            num_data_workers=self.num_data_workers,
        )


def _make_dummy_ocean_forcing(
    dataset_info: CoupledDatasetInfo,
    initial_time: xr.DataArray,
    total_coupled_steps: int,
    ocean_reqs: DataRequirements,
) -> tuple[DummyDataset, DatasetProperties]:
    assert dataset_info.ocean is not None
    ocean_property = DatasetProperties(
        variable_metadata=dict(dataset_info.ocean.variable_metadata),
        vertical_coordinate=dataset_info.ocean.vertical_coordinate,
        horizontal_coordinates=dataset_info.ocean.horizontal_coordinates,
        spatial_mask_provider=dataset_info.ocean.spatial_mask_provider,
        timestep=dataset_info.ocean.timestep,
        is_remote=False,
        all_labels=set(),
    )
    ts = dataset_info.ocean.timestep
    ocean = DummyDataset(
        start_time=initial_time.squeeze().values.flat[0],
        end_time=initial_time.squeeze().values.flat[-1] + ts * total_coupled_steps,
        timestep=ts,
        n_timesteps=ocean_reqs.n_timesteps_schedule,
        horizontal_coordinates=dataset_info.ocean.horizontal_coordinates,
        labels=None,
    )
    return ocean, ocean_property


def _make_dummy_ice_forcing(
    dataset_info: CoupledDatasetInfo,
    initial_time: xr.DataArray,
    total_coupled_steps: int,
    ice_reqs: DataRequirements,
) -> tuple[DummyDataset, DatasetProperties]:
    assert dataset_info.ice is not None
    ice_property = DatasetProperties(
        variable_metadata=dict(dataset_info.ice.variable_metadata),
        vertical_coordinate=dataset_info.ice.vertical_coordinate,
        horizontal_coordinates=dataset_info.ice.horizontal_coordinates,
        spatial_mask_provider=dataset_info.ice.spatial_mask_provider,
        timestep=dataset_info.ice.timestep,
        is_remote=False,
        all_labels=set(),
    )
    ts = dataset_info.ice.timestep
    ice = DummyDataset(
        start_time=initial_time.squeeze().values.flat[0],
        end_time=initial_time.squeeze().values.flat[-1] + ts * total_coupled_steps,
        timestep=ts,
        n_timesteps=ice_reqs.n_timesteps_schedule,
        horizontal_coordinates=dataset_info.ice.horizontal_coordinates,
        labels=None,
    )
    return ice, ice_property
