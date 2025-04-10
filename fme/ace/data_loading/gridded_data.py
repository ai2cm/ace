import datetime
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sized,
    TypeVar,
    Union,
)

import numpy as np
import torch

from fme.ace.data_loading.augmentation import BatchModifierABC, NullModifier
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.requirements import PrognosticStateDataRequirements
from fme.core.coordinates import HorizontalCoordinates, VerticalCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.xarray import DatasetProperties
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.data import DataLoader, GriddedDataABC, InferenceDataABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.step.step import InferenceDataProtocol

T = TypeVar("T", covariant=True)


U = TypeVar("U")


class SizedMap(Generic[T, U], Sized, Iterable[U]):
    def __init__(self, func: Callable[[T], U], iterable: DataLoader[T]):
        self._func = func
        self._iterable = iterable

    def __len__(self) -> int:
        return len(self._iterable)

    def __iter__(self) -> Iterator[U]:
        return map(self._func, self._iterable)


class GriddedData(GriddedDataABC[BatchData]):
    """
    Data as required for pytorch training.

    The data is assumed to be gridded, and attributes are included for
    performing operations on gridded data.

    All data exposed from this class is on the current device.
    """

    def __init__(
        self,
        loader: DataLoader[BatchData],
        properties: DatasetProperties,
        modifier: BatchModifierABC = NullModifier(),
        sampler: Optional[torch.utils.data.Sampler] = None,
    ):
        """
        Args:
            loader: torch DataLoader, which returns batches of type
                TensorMapping where keys indicate variable name.
                Each tensor has shape
                [batch_size, face, time_window_size, n_channels, n_x_coord, n_y_coord].
                Data can be on any device (but will typically be on CPU).
            properties: Batch-constant properties for the dataset, such as variable
                metadata and coordinate information. Data can be on any device.
            modifier: Modifier for the data loader.
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.

        Note:
            While input data can be on any device, all data exposed from this class
            will be on the current device.
        """
        self._loader = loader
        self._properties = properties.to_device()
        self._timestep = self._properties.timestep
        self._vertical_coordinate = self._properties.vertical_coordinate
        self._gridded_operations = (
            self._properties.horizontal_coordinates.gridded_operations
        )
        self._sampler = sampler
        self._modifier = modifier
        self._batch_size: Optional[int] = None
        img_shape = None
        for batch in self.loader:
            shapes = {k: v.shape for k, v in batch.data.items()}
            for value in shapes.values():
                img_shape = value[-2:]
                break
            break
        if img_shape is None:
            raise ValueError("No data found in loader")
        self._img_shape = img_shape

    @property
    def loader(self) -> DataLoader[BatchData]:
        def modify_and_on_device(batch: BatchData) -> BatchData:
            return self._modifier(batch).to_device()

        return SizedMap(modify_and_on_device, self._loader)

    @property
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            img_shape=self._img_shape,
            gridded_operations=self._gridded_operations,
            vertical_coordinate=self._vertical_coordinate,
            timestep=self._timestep,
        )

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self._vertical_coordinate.coords,
        }

    @property
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        return self.horizontal_coordinates.grid

    @property
    def n_samples(self) -> int:
        return len(self._loader.dataset)  # type: ignore

    @property
    def n_batches(self) -> int:
        return len(self._loader)  # type: ignore

    @property
    def _first_time(self) -> Any:
        return self._loader.dataset[0][1].values[0]  # type: ignore

    @property
    def _last_time(self) -> Any:
        return self._loader.dataset[-1][1].values[0]  # type: ignore

    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            example_data = next(iter(self.loader)).data
            example_tensor = next(iter(example_data.values()))
            self._batch_size = example_tensor.shape[0]
        return self._batch_size

    def log_info(self, name: str):
        logging.info(f"{name} data: {self.n_samples} samples, {self.n_batches} batches")
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


def get_initial_condition(
    loader: DataLoader[BatchData],
    requirements: PrognosticStateDataRequirements,
) -> PrognosticState:
    for batch in loader:
        return batch.to_device().get_start(
            prognostic_names=requirements.names,
            n_ic_timesteps=requirements.n_timesteps,
        )
    raise ValueError("No initial condition found in loader")


class InferenceGriddedData(InferenceDataABC[PrognosticState, BatchData]):
    """
    Data as required for inference.

    All data exposed from this class is on the current device.
    """

    def __init__(
        self,
        loader: DataLoader[BatchData],
        initial_condition: Union[PrognosticState, PrognosticStateDataRequirements],
        properties: DatasetProperties,
    ):
        """
        Args:
            loader: torch DataLoader, which returns batches of type
                TensorMapping where keys indicate variable name.
                Each tensor has shape
                [batch_size, face, time_window_size, n_channels, n_x_coord, n_y_coord].
                Data can be on any device (but will typically be on CPU).
            initial_condition: Initial condition for the inference, or a requirements
                object specifying how to extract the initial condition from the first
                batch of data. Data can be on any device.
            properties: Batch-constant properties for the dataset, such as variable
                metadata and coordinate information. Data can be on any device.

        Note:
            While input data can be on any device, all data exposed from this class
            will be on the current device.
        """
        self._loader = loader
        self._properties = properties.to_device()
        self._n_initial_conditions: Optional[int] = None
        if isinstance(initial_condition, PrognosticStateDataRequirements):
            self._initial_condition: PrognosticState = get_initial_condition(
                loader, initial_condition
            )
        else:
            self._initial_condition = initial_condition.to_device()

        _: InferenceDataProtocol = self

    @property
    def loader(self) -> DataLoader[BatchData]:
        def on_device(batch: BatchData) -> BatchData:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def variable_metadata(self) -> Dict[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        return self._properties.vertical_coordinate

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._properties.horizontal_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._properties.timestep

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        return {
            **self.horizontal_coordinates.coords,
            **self.vertical_coordinate.coords,
        }

    @property
    def gridded_operations(self) -> GriddedOperations:
        return self.horizontal_coordinates.gridded_operations

    @property
    def _n_samples(self) -> int:
        return len(self._loader.dataset)  # type: ignore

    @property
    def _n_batches(self) -> int:
        return len(self._loader)  # type: ignore

    @property
    def _first_time(self) -> Any:
        return self._loader.dataset[0][1].values[0]  # type: ignore

    @property
    def _last_time(self) -> Any:
        return self._loader.dataset[-1][1].values[0]  # type: ignore

    @property
    def n_initial_conditions(self) -> int:
        if self._n_initial_conditions is None:
            example_data = next(iter(self.loader)).data
            example_tensor = next(iter(example_data.values()))
            self._n_initial_conditions = example_tensor.shape[0]
        return self._n_initial_conditions

    @property
    def initial_condition(self) -> PrognosticState:
        return self._initial_condition

    def log_info(self, name: str):
        logging.info(
            f"{name} data: {self._n_samples} samples, {self._n_batches} batches"
        )
        logging.info(f"{name} data: first sample's initial time: {self._first_time}")
        logging.info(f"{name} data: last sample's initial time: {self._last_time}")
