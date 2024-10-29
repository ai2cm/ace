import abc
import dataclasses
import datetime
import logging
from typing import (
    Any,
    Callable,
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

from fme.ace.inference.derived_variables import (
    compute_derived_quantities,  # TODO: move to core or move stepper to ace
)
from fme.core.data_loading.data_typing import (
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.device import move_tensordict_to_device
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping


def get_atmospheric_batch_data(
    data: TensorMapping,
    times: xr.DataArray,
    sigma_coordinates: SigmaCoordinates,
) -> "BatchData":
    """
    Get atmospheric batch data.

    Args:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch.
        times: An array of times of shape (sample, time) for each sample in the batch,
            concatenated along samples to make a batch.
        sigma_coordinates: Sigma coordinates for the data.

    Returns:
        BatchData: The batch data.
    """
    if times.shape[1] < 2:
        raise ValueError(
            "Times must have at least two timesteps to compute derived variables. "
            "If you don't need to compute derived variables, initialize BatchData "
            "directly."
        )
    timestep = times.values[0, 1] - times.values[0, 0]

    def _derive_func(
        batch_data: TensorMapping, forcing_data: TensorMapping
    ) -> TensorDict:
        return compute_derived_quantities(
            dict(batch_data),
            sigma_coordinates=sigma_coordinates,
            timestep=timestep,
            forcing_data=dict(forcing_data),
        )

    return BatchData(
        data=data,
        times=times,
        derive_func=_derive_func,
    )


@dataclasses.dataclass
class BatchData:
    """A container for the data and time coordinates of a batch.

    Attributes:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch.
        times: An array of times of shape (sample, time) for each sample in the batch,
            concatenated along samples to make a batch. To be used in writing out
            inference predictions with time coordinates, not directly in ML.
        derive_func: A function that takes a batch of data and a batch of forcing data
            and returns a batch of derived data. If not given, no derived variables
            are computed.
    """

    data: TensorMapping
    times: xr.DataArray
    derive_func: Callable[
        [TensorMapping, TensorMapping], TensorDict
    ] = lambda x, _: dict(x)

    def __post_init__(self):
        self._device_data: Optional[TensorMapping] = None
        if len(self.times.shape) != 2:
            raise ValueError(
                "Expected times to have shape (n_samples, n_times), got shape "
                f"{self.times.shape}."
            )

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
        derive_func: Callable[
            [TensorMapping, TensorMapping], TensorDict
        ] = lambda x, _: dict(x),
    ) -> "BatchData":
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim=sample_dim_name)
        return cls(
            data=batch_data,
            times=batch_times,
            derive_func=derive_func,
        )

    @classmethod
    def atmospheric_from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sigma_coordinates: SigmaCoordinates,
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
        return get_atmospheric_batch_data(
            data=batch_data, times=batch_times, sigma_coordinates=sigma_coordinates
        )

    def compute_derived_variables(self, forcing_data: "BatchData") -> "BatchData":
        if not np.all(forcing_data.times.values == self.times.values):
            raise ValueError("Forcing data must have the same times as the batch data.")
        derived_data = self.derive_func(self.data, forcing_data.data)
        return BatchData(
            data={**self.data, **derived_data},
            times=self.times,
            derive_func=self.derive_func,
        )

    def remove_initial_condition(self, n_ic_timesteps: int) -> "BatchData":
        if n_ic_timesteps == 0:
            raise RuntimeError("No initial condition timesteps to remove.")
        return BatchData(
            {k: v[:, n_ic_timesteps:] for k, v in self.data.items()},
            times=self.times[:, n_ic_timesteps:],
        )

    def subset_names(self, names: Sequence[str]) -> "BatchData":
        return BatchData(
            {k: v for k, v in self.data.items() if k in names},
            times=self.times,
        )


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
