import dataclasses
from typing import Mapping

import numpy as np
import xarray as xr

from fme.core.data_loading._xarray import XarrayDataset
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.params import DataLoaderParams, XarrayDataParams
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData
from fme.core.distributed import Distributed


@dataclasses.dataclass
class InferenceInitialConditionIndices:
    """
    Configuration of the indices for initial conditions during inference.
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
class InferenceDataLoaderParams:
    """
    Configuration for inference data.

    This is like the `DataLoaderParams` class, but with some additional
    constraints. During inference, we have only one batch, so the number of
    samples directly determines the size of that batch.

    Attributes:
        dataset: Parameters to define the dataset.
        start_indices: Slice indicating the set of indices to consider for initial
            conditions of inference series of data. Values following the initial
            condition will still come from the full dataset.
            By default load all initial conditions.
    """

    dataset: XarrayDataParams
    start_indices: InferenceInitialConditionIndices

    def __post_init__(self):
        dist = Distributed.get_instance()
        self._data_loader_params = DataLoaderParams(
            dataset=self.dataset,
            data_type="xarray",
            batch_size=dist.world_size,  # not used for inference
            num_data_workers=0,  # not used for inference
        )

    @property
    def n_samples(self) -> int:
        return self.start_indices.n_initial_conditions

    @property
    def data_loader_params(self) -> DataLoaderParams:
        return self._data_loader_params


class InferenceDataLoader:
    def __init__(
        self,
        params: InferenceDataLoaderParams,
        forward_steps_in_memory: int,
        requirements: DataRequirements,
    ):
        dataset = XarrayDataset(params.data_loader_params, requirements=requirements)
        self._dataset = dataset
        self._sigma_coordinates = dataset.sigma_coordinates
        self._metadata = dataset.metadata
        self._area_weights = dataset.area_weights
        self._coords = {
            **dataset.horizontal_coordinates.coords,
            **dataset.sigma_coordinates.coords,
        }
        self._forward_steps_in_memory = forward_steps_in_memory
        self._total_steps = requirements.n_timesteps - 1
        if self._total_steps % self._forward_steps_in_memory != 0:
            raise ValueError(
                f"Total number of steps ({self._total_steps}) must be divisible by "
                f"forward_steps_in_memory ({self._forward_steps_in_memory})."
            )
        self.n_samples = params.n_samples  # public attribute
        self._start_indices = params.start_indices.as_indices()

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
            window_time_slice = slice(i_window_start, i_window_end)
            sample_tuples.append(
                self._dataset.get_sample_by_time_slice(window_time_slice)
            )
            assert sample_tuples[-1][1].shape[0] == self._forward_steps_in_memory + 1
        result = BatchData.from_sample_tuples(sample_tuples)
        assert result.times.shape[1] == self._forward_steps_in_memory + 1
        assert result.times.shape[0] == self.n_samples // dist.world_size
        return result

    def __len__(self) -> int:
        return self._total_steps // self._forward_steps_in_memory

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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
    def coords(self) -> Mapping[str, np.ndarray]:
        return self._coords
