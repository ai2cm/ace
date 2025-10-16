import dataclasses
import logging
from collections.abc import Sequence
from math import ceil

import cftime
import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.data_loading.perturbation import SSTPerturbation
from fme.ace.requirements import DataRequirements
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.merged import (
    MergedXarrayDataset,
    MergeNoConcatDatasetConfig,
    get_per_dataset_names,
)
from fme.core.dataset.properties import DatasetProperties
from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset
from fme.core.distributed import Distributed
from fme.core.typing_ import Slice


@dataclasses.dataclass
class TimestampList:
    """
    Configuration for a list of timestamps.

    Parameters:
        times: List of timestamps.
        timestamp_format: Format of the timestamps.
    """

    times: Sequence[str]
    timestamp_format: str = "%Y-%m-%dT%H:%M:%S"

    def as_indices(self, time_index: xr.CFTimeIndex) -> np.ndarray:
        datetimes = [
            cftime.datetime.strptime(
                t, self.timestamp_format, calendar=time_index.calendar
            )
            for t in self.times
        ]
        (indices,) = time_index.isin(datetimes).nonzero()
        if len(indices) != len(self.times):
            missing_times = set(datetimes) - set(time_index[indices])
            raise ValueError(
                f"Inference initial condition timestamps {missing_times} "
                "were not found in the dataset."
            )
        return indices

    @property
    def n_initial_conditions(self) -> int:
        return len(self.times)


@dataclasses.dataclass
class InferenceInitialConditionIndices:
    """
    Configuration of the indices for initial conditions during inference.

    Parameters:
        n_initial_conditions: Number of initial conditions to use.
        first: Index of the first initial condition.
        interval: Interval between initial conditions.
    """

    n_initial_conditions: int
    first: int = 0
    interval: int = 1

    def __post_init__(self):
        if self.interval < 0:
            raise ValueError("interval must be positive")

    def as_indices(self) -> np.ndarray:
        stop = self.n_initial_conditions * self.interval + self.first
        return np.arange(self.first, stop, self.interval)


@dataclasses.dataclass
class ExplicitIndices:
    """
    Configure indices providing them explicitly.

    Parameters:
        list: List of integer indices.
    """

    list: Sequence[int]

    def as_indices(self) -> np.ndarray:
        return np.array(self.list)

    @property
    def n_initial_conditions(self) -> int:
        return len(self.list)


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
        perturbations: Configuration for SST perturbations.
        persistence_names: Names of variables for which all returned values
            will be the same as the initial condition. When evaluating initial
            condition predictability, set this to forcing variables that should
            not be updated during inference (e.g. surface temperature).
    """

    dataset: XarrayDataConfig | MergeNoConcatDatasetConfig
    start_indices: InferenceInitialConditionIndices | ExplicitIndices | TimestampList
    num_data_workers: int = 0
    perturbations: SSTPerturbation | None = None
    persistence_names: Sequence[str] | None = None

    def __post_init__(self):
        if isinstance(self.dataset, XarrayDataConfig):
            if self.dataset.subset != Slice(None, None, None):
                raise ValueError("Inference data may not be subset.")
        elif isinstance(self.dataset, MergeNoConcatDatasetConfig):
            for data in self.dataset.merge:
                if data.subset != Slice(None, None, None):
                    raise ValueError(f"Inference data may not be subset.")
        self._zarr_engine_used = self.dataset.zarr_engine_used

    @property
    def n_initial_conditions(self) -> int:
        return self.start_indices.n_initial_conditions

    @property
    def zarr_engine_used(self) -> bool:
        """
        Whether any of the configured datasets are using the Zarr engine.
        """
        return self._zarr_engine_used


@dataclasses.dataclass
class ForcingDataLoaderConfig:
    """
    Configuration for the forcing data.

    Parameters:
        dataset: Configuration to define the dataset.
        num_data_workers: Number of parallel workers to use for data loading.
        perturbations: Configuration for SST perturbations
            used in forcing data.
        persistence_names: Names of variables for which all returned values
            will be the same as the initial condition. When evaluating initial
            condition predictability, set this to forcing variables that should
            not be updated during inference (e.g. surface temperature).
    """

    dataset: XarrayDataConfig | MergeNoConcatDatasetConfig
    num_data_workers: int = 0
    perturbations: SSTPerturbation | None = None
    persistence_names: Sequence[str] | None = None

    def __post_init__(self):
        if isinstance(self.dataset, XarrayDataConfig):
            if self.dataset.subset != Slice(None, None, None):
                raise ValueError("Inference data may not be subset.")
        elif isinstance(self.dataset, MergeNoConcatDatasetConfig):
            for data in self.dataset.merge:
                if data.subset != Slice(None, None, None):
                    raise ValueError(f"Inference data may not be subset.")

    def build_inference_config(self, start_indices: ExplicitIndices):
        return InferenceDataLoaderConfig(
            dataset=self.dataset,
            num_data_workers=self.num_data_workers,
            start_indices=start_indices,
            perturbations=self.perturbations,
            persistence_names=self.persistence_names,
        )


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: InferenceDataLoaderConfig,
        total_forward_steps: int,
        requirements: DataRequirements,
        label_override: list[str] | None = None,
        surface_temperature_name: str | None = None,
        ocean_fraction_name: str | None = None,
    ):
        """
        Parameters:
            config: Configuration for the inference data.
            total_forward_steps: Number of forward steps to take.
            requirements: Requirements for the inference data.
            label_override: Labels to override the labels in the dataset. If provided,
                these labels will be provided on each sample instead of the labels
                in the dataset.
            surface_temperature_name: Name of the surface temperature variable.
            ocean_fraction_name: Name of the ocean fraction variable.
        """
        self._label_override = (
            set(label_override) if label_override is not None else None
        )
        if isinstance(config.dataset, XarrayDataConfig):
            dataset: XarrayDataset | MergedXarrayDataset = XarrayDataset(
                config.dataset, requirements.names, requirements.n_timesteps
            )
            properties = dataset.properties
        elif isinstance(config.dataset, MergeNoConcatDatasetConfig):
            dataset = self._resolve_merged_datasets(config.dataset, requirements)
            properties = dataset.properties
        self._properties = properties
        self._dataset = dataset
        self._forward_steps_in_memory = requirements.n_timesteps - 1
        self._total_forward_steps = total_forward_steps
        self._perturbations = config.perturbations
        self._surface_temperature_name = surface_temperature_name
        self._ocean_fraction_name = ocean_fraction_name
        self._n_initial_conditions = config.n_initial_conditions
        if isinstance(config.start_indices, TimestampList):
            self._start_indices = config.start_indices.as_indices(
                self._dataset.all_times
            )
        else:
            self._start_indices = config.start_indices.as_indices()
        self._validate_n_forward_steps()
        if isinstance(self._properties.horizontal_coordinates, LatLonCoordinates):
            self._lats, self._lons = self._properties.horizontal_coordinates.meshgrid
        else:
            if self._perturbations is not None:
                raise ValueError(
                    "Currently, SST perturbations are only supported \
                    for lat/lon coordinates."
                )
        if self._perturbations is not None and (
            self._surface_temperature_name is None or self._ocean_fraction_name is None
        ):
            raise ValueError(
                "No ocean configuration found, \
                SST perturbations require an ocean configuration."
            )

        self._persistence_data: BatchData | None = None
        if config.persistence_names is not None:
            first_sample = self._get_batch_data(0)
            self._persistence_data = first_sample.subset_names(
                config.persistence_names
            ).select_time_slice(slice(0, 1))

    def _get_batch_data(self, index) -> BatchData:
        dist = Distributed.get_instance()
        i_start = index * self._forward_steps_in_memory
        sample_tuples = []
        for i_member in range(self._n_initial_conditions):
            # check if sample is one this local rank should process
            if i_member % dist.world_size != dist.rank:
                continue
            i_window_start = i_start + self._start_indices[i_member]
            i_window_end = i_window_start + self._forward_steps_in_memory + 1
            if i_window_end > (
                self._total_forward_steps + self._start_indices[i_member]
            ):
                i_window_end = (
                    self._total_forward_steps + self._start_indices[i_member] + 1
                )
            window_time_slice = slice(i_window_start, i_window_end)
            tensors, time, labels = self._dataset.get_sample_by_time_slice(
                window_time_slice
            )
            if self._label_override is not None:
                labels = self._label_override
            if self._perturbations is not None:
                if (
                    self._surface_temperature_name is None
                    or self._ocean_fraction_name is None
                ):
                    raise ValueError(
                        "Surface temperature and ocean fraction names must be provided \
                        to apply SST perturbations."
                    )
                logging.debug("Applying SST perturbations to forcing data")
                for perturbation in self._perturbations.perturbations:
                    perturbation.apply_perturbation(
                        tensors[self._surface_temperature_name],
                        self._lats,
                        self._lons,
                        tensors[self._ocean_fraction_name],
                    )
            sample_tuples.append((tensors, time, labels))
        return BatchData.from_sample_tuples(
            sample_tuples,
            horizontal_dims=list(self.properties.horizontal_coordinates.dims),
        )

    def __getitem__(self, index) -> BatchData:
        dist = Distributed.get_instance()
        result = self._get_batch_data(index)
        if self._persistence_data is not None:
            updated_data = {}
            for key, value in self._persistence_data.data.items():
                updated_data[key] = value.expand_as(result.data[key])
            result.data = {**result.data, **updated_data}
        assert result.time.shape[0] == self._n_initial_conditions // dist.world_size
        return result

    def __len__(self) -> int:
        # The ceil is necessary so if the last batch is smaller
        # than the rest the ratio will be rounded up and the last batch
        # will be included in the loading
        return int(ceil(self._total_forward_steps / self._forward_steps_in_memory))

    @property
    def properties(self) -> DatasetProperties:
        return self._properties

    @property
    def n_forward_steps(self) -> int:
        return self._total_forward_steps

    def _validate_n_forward_steps(self):
        max_steps = self._dataset.total_timesteps - max(self._start_indices) - 1
        if self._total_forward_steps > max_steps:
            raise ValueError(
                f"The number of forward inference steps ({self._total_forward_steps}) "
                "must be less than or equal to the number of possible steps "
                f"({max_steps}) in dataset after the last initial condition's "
                "start index."
            )

    def _resolve_merged_datasets(
        self,
        merged_datasets_config: MergeNoConcatDatasetConfig,
        requirements: DataRequirements,
    ):
        per_dataset_names = get_per_dataset_names(
            merged_datasets_config, requirements.names
        )
        merged_xarray_datasets = []
        config_counter = 0
        for config in merged_datasets_config.merge:
            current_dataset = XarrayDataset(
                config,
                per_dataset_names[config_counter],
                requirements.n_timesteps,
            )
            merged_xarray_datasets.append(current_dataset)
            config_counter += 1
        merged_datasets = MergedXarrayDataset(datasets=merged_xarray_datasets)
        return merged_datasets
