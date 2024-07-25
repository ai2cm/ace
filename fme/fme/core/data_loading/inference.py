import dataclasses
import datetime
from math import ceil
from typing import Sequence, Union

import cftime
import numpy as np
import torch
import xarray as xr

from fme.core.data_loading._xarray import XarrayDataset
from fme.core.data_loading.config import Slice, XarrayDataConfig
from fme.core.data_loading.data_typing import HorizontalCoordinates, SigmaCoordinates
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData
from fme.core.distributed import Distributed


@dataclasses.dataclass
class TimestampList:
    """
    Configuration for a list of timestamps.

    Attributes:
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

    Attributes:
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

    Attributes:
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

    Attributes:
        dataset: Configuration to define the dataset.
        start_indices: Configuration of the indices for initial conditions
            during inference. This can be a list of timestamps, a list of
            integer indices, or a slice configuration of the integer indices.
            Values following the initial condition will still come from
            the full dataset.
        num_data_workers: Number of parallel workers to use for data loading.
    """

    dataset: XarrayDataConfig
    start_indices: Union[
        InferenceInitialConditionIndices, ExplicitIndices, TimestampList
    ]
    num_data_workers: int = 0

    def __post_init__(self):
        if self.dataset.subset != Slice(None, None, None):
            raise ValueError("Inference data may not be subset.")

    @property
    def n_samples(self) -> int:
        return self.start_indices.n_initial_conditions


@dataclasses.dataclass
class ForcingDataLoaderConfig:
    """
    Configuration for the forcing data.

    Attributes:
        dataset: Configuration to define the dataset.
        num_data_workers: Number of parallel workers to use for data loading.
    """

    dataset: XarrayDataConfig
    num_data_workers: int = 0

    def __post_init__(self):
        if self.dataset.subset != Slice(None, None, None):
            raise ValueError("Inference data may not be subset.")

    def build_inference_config(self, start_indices: ExplicitIndices):
        return InferenceDataLoaderConfig(
            dataset=self.dataset,
            num_data_workers=self.num_data_workers,
            start_indices=start_indices,
        )


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: InferenceDataLoaderConfig,
        forward_steps_in_memory: int,
        requirements: DataRequirements,
    ):
        dataset = XarrayDataset(config.dataset, requirements=requirements)
        self._dataset = dataset
        self._sigma_coordinates = dataset.sigma_coordinates
        self._metadata = dataset.metadata
        self._area_weights = dataset.area_weights
        self._horizontal_coordinates = dataset.horizontal_coordinates
        self._timestep = dataset.timestep
        self._forward_steps_in_memory = forward_steps_in_memory
        self._total_steps = requirements.n_timesteps - 1
        self._is_remote = dataset.is_remote
        self.n_samples = config.n_samples  # public attribute
        if isinstance(config.start_indices, TimestampList):
            self._start_indices = config.start_indices.as_indices(dataset.all_times)
        else:
            self._start_indices = config.start_indices.as_indices()
        self._validate_n_forward_steps()

    def __getitem__(self, index) -> BatchData:
        dist = Distributed.get_instance()
        i_start = index * self._forward_steps_in_memory
        sample_tuples = []
        for i_sample in range(self.n_samples):
            # check if sample is one this local rank should process
            if i_sample % dist.world_size != dist.rank:
                continue
            i_window_start = i_start + self._start_indices[i_sample]
            i_window_end = i_window_start + self._forward_steps_in_memory + 1
            if i_window_end > (self._total_steps + self._start_indices[i_sample]):
                i_window_end = self._total_steps + self._start_indices[i_sample] + 1
            window_time_slice = slice(i_window_start, i_window_end)
            sample_tuples.append(
                self._dataset.get_sample_by_time_slice(window_time_slice)
            )
        result = BatchData.from_sample_tuples(sample_tuples)
        assert result.times.shape[0] == self.n_samples // dist.world_size
        return result

    def __len__(self) -> int:
        # The ceil is necessary so if the last batch is smaller
        # than the rest the ratio will be rounded up and the last batch
        # will be included in the loading
        return int(ceil(self._total_steps / self._forward_steps_in_memory))

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._sigma_coordinates

    @property
    def metadata(self) -> xr.Dataset:
        return self._metadata

    @property
    def area_weights(self) -> xr.DataArray:
        return self._area_weights

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    @property
    def n_forward_steps(self) -> int:
        return self._total_steps

    def _validate_n_forward_steps(self):
        max_steps = self._dataset.total_timesteps - self._start_indices[-1] - 1
        if self._total_steps > max_steps:
            raise ValueError(
                f"The number of forward inference steps ({self._total_steps}) must "
                f"be less than or equal to the number of possible steps ({max_steps})"
                f"in dataset after the last initial condition's start index."
            )
