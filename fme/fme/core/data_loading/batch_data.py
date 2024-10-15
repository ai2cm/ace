import abc
import dataclasses
import datetime
import logging
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.core.data_loading.data_typing import (
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.device import move_tensordict_to_device
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class BatchData:
    """A container for the data and time coordinates of a batch.

    Attributes:
        data: Data for each variable in each sample, concatenated along samples
            to make a batch. To be used directly in training, validation, and
            inference.
        times: An array of times for each sample in the batch, concatenated along
            samples to make a batch. To be used in writing out inference
            predictions with time coordinates, not directly in ML.

    """

    data: TensorMapping
    times: xr.DataArray

    def __post_init__(self):
        self._device_data: Optional[TensorMapping] = None

    @property
    def device_data(self) -> TensorMapping:
        if self._device_data is None:
            self._device_data = move_tensordict_to_device(self.data)
        return self._device_data

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
    ) -> "BatchData":
        """
        Collate function for use with PyTorch DataLoader. Needed since samples contain
        both tensor mapping and xarray time coordinates, the latter of which we do
        not want to convert to tensors.
        """
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim=sample_dim_name)
        return cls(batch_data, batch_times)


T = TypeVar("T", covariant=True)


class DataLoader(Protocol, Generic[T]):
    def __iter__(self) -> Iterator[T]:
        ...


class GriddedDataABC(abc.ABC, Generic[T]):
    @property
    @abc.abstractmethod
    def loader(self) -> DataLoader[T]:
        ...

    @property
    @abc.abstractmethod
    def sigma_coordinates(self) -> SigmaCoordinates:
        ...

    @property
    @abc.abstractmethod
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        ...

    @property
    @abc.abstractmethod
    def timestep(self) -> datetime.timedelta:
        ...

    @property
    @abc.abstractmethod
    def gridded_operations(self) -> GriddedOperations:
        ...

    @property
    @abc.abstractmethod
    def n_samples(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_batches(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_forward_steps(self) -> int:
        ...

    @abc.abstractmethod
    def set_epoch(self, epoch: int):
        ...

    @abc.abstractmethod
    def log_info(self, name: str):
        """
        Report information about the data using logging.info.
        """
        ...


class GriddedData(GriddedDataABC[BatchData]):
    """
    Data as required for pytorch training.

    The data is assumed to be gridded, and attributes are included for
    performing operations on gridded data.
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        metadata: Mapping[str, VariableMetadata],
        sigma_coordinates: SigmaCoordinates,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: datetime.timedelta,
        sampler: Optional[torch.utils.data.Sampler] = None,
    ):
        """
        Args:
            loader: torch DataLoader, which returns batches of type
                TensorMapping where keys indicate variable name.
                Each tensor has shape
                [batch_size, face, time_window_size, n_channels, n_x_coord, n_y_coord].
            metadata: Metadata for each variable.
            area_weights: Weights for each grid cell, used for computing area-weighted
                averages. Has shape [n_x_coord, n_y_coord].
            sigma_coordinates: Sigma coordinates for each grid cell, used for computing
                pressure levels.
            horizontal_coordinates: horizontal coordinates for the data.
            timestep: Timestep of the model.
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.
        """
        self._loader = loader
        self._metadata = metadata
        self._sigma_coordinates = sigma_coordinates
        self._horizontal_coordinates = horizontal_coordinates
        self._timestep = timestep
        self._sampler = sampler
        self._batch_size: Optional[int] = None

    @property
    def loader(self) -> DataLoader[BatchData]:
        return self._loader

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        return self._metadata

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._sigma_coordinates

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self.sigma_coordinates.coords,
        }

    @property
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        return self.horizontal_coordinates.grid

    @property
    def gridded_operations(self) -> GriddedOperations:
        return self.horizontal_coordinates.gridded_operations

    @property
    def n_samples(self) -> int:
        return len(self._loader.dataset)

    @property
    def n_batches(self) -> int:
        return len(self._loader)

    @property
    def _first_time(self) -> Any:
        return self._loader.dataset[0][1].values[0]

    @property
    def _last_time(self) -> Any:
        return self._loader.dataset[-1][1].values[0]

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            example_data = next(iter(self.loader)).data
            example_tensor = next(iter(example_data.values()))
            self._batch_size = example_tensor.shape[0]
        return self._batch_size

    @property
    def n_forward_steps(self) -> int:
        return self._loader.dataset.n_forward_steps

    def log_info(self, name: str):
        logging.info(
            f"{name} data: {self.n_samples} samples, " f"{self.n_batches} batches"
        )
        logging.info(f"{name} data: first sample's initial time: {self._first_time}")
        logging.info(f"{name} data: last sample's initial time: {self._last_time}")

    def set_epoch(self, epoch: int):
        """
        Set the epoch for the data loader sampler, if it is a distributed sampler.
        """
        if self._sampler is not None and isinstance(
            self._sampler, torch.utils.data.DistributedSampler
        ):
            self._sampler.set_epoch(epoch)
