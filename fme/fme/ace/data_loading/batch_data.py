import dataclasses
import datetime
import logging
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.ace.requirements import PrognosticStateDataRequirements
from fme.core.coordinates import HorizontalCoordinates, HybridSigmaPressureCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.xarray import DatasetProperties
from fme.core.device import get_device
from fme.core.generics.data import DataLoader, GriddedDataABC, InferenceDataABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping

SelfType = TypeVar("SelfType", bound="BatchData")


def _check_device(data: TensorMapping, device: torch.device):
    for v in data.values():
        if v.device != device:
            raise ValueError(f"data must be on {device}")


class PrognosticState:
    """
    Thin typing wrapper around BatchData to indicate that the data is a prognostic
    state, such as an initial condition or final state when evolving forward in time.
    """

    def __init__(self, data: "BatchData"):
        """
        Initialize the state.

        Args:
            data: The data to initialize the state with.
        """
        self._data = data

    def to_device(self) -> "PrognosticState":
        return PrognosticState(self._data.to_device())

    def as_batch_data(self) -> "BatchData":
        return self._data


@dataclasses.dataclass
class BatchData:
    """A container for the data and time coordinates of a batch.

    Parameters:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch. To be used directly in training,
            validation, and inference.
        time: An array representing time coordinates for each sample in the batch,
            concatenated along samples to make a batch. To be used in writing out
            inference predictions with time coordinates, not directly in ML.
        horizontal_dims: Horizontal dimensions of the data. Used for writing to
            netCDF files.
    """

    data: TensorMapping
    time: xr.DataArray
    horizontal_dims: List[str] = dataclasses.field(
        default_factory=lambda: ["lat", "lon"]
    )

    @property
    def dims(self) -> List[str]:
        return ["sample", "time"] + self.horizontal_dims

    def to_device(self) -> "BatchData":
        return self.__class__(
            data={k: v.to(get_device()) for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
        )

    @classmethod
    def _get_kwargs(cls, horizontal_dims: Optional[List[str]]) -> Dict[str, Any]:
        if horizontal_dims is None:
            kwargs = {}
        else:
            kwargs = {"horizontal_dims": horizontal_dims}
        return kwargs

    @classmethod
    def new_on_cpu(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        horizontal_dims: Optional[List[str]] = None,
    ) -> "BatchData":
        _check_device(data, torch.device("cpu"))
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData(
            data=data,
            time=time,
            **kwargs,
        )

    @classmethod
    def new_on_device(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        horizontal_dims: Optional[List[str]] = None,
    ) -> "BatchData":
        """
        Move the data to the current global device specified by get_device().
        """
        _check_device(data, get_device())
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData(
            data=data,
            time=time,
            **kwargs,
        )

    def __post_init__(self):
        if len(self.time.shape) != 2:
            raise ValueError(
                "Expected time to have shape (n_samples, n_times), got shape "
                f"{self.time.shape}."
            )
        for k, v in self.data.items():
            if v.shape[:2] != self.time.shape[:2]:
                raise ValueError(
                    f"Data for variable {k} has shape {v.shape}, expected shape "
                    f"(n_samples, n_times) for time but got shape "
                    f"{self.time.shape}."
                )

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
        horizontal_dims: Optional[List[str]] = None,
    ) -> "BatchData":
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_time = xr.concat(sample_times, dim=sample_dim_name)
        return BatchData.new_on_cpu(
            data=batch_data,
            time=batch_time,
            horizontal_dims=horizontal_dims,
        )

    def compute_derived_variables(
        self: SelfType,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        forcing_data: SelfType,
    ) -> SelfType:
        """
        Compute derived variables from the data and forcing data.

        The forcing data must have the same time coordinate as the batch data.

        Args:
            derive_func: A function that takes the data and forcing data and returns a
                dictionary of derived variables.
            forcing_data: The forcing data to compute derived variables from.
        """
        if not np.all(forcing_data.time.values == self.time.values):
            raise ValueError(
                "Forcing data must have the same time coordinate as the batch data."
            )
        derived_data = derive_func(self.data, forcing_data.data)
        return self.__class__(
            data={**self.data, **derived_data},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
        )

    def remove_initial_condition(self: SelfType, n_ic_timesteps: int) -> SelfType:
        """
        Remove the initial condition timesteps from the data.
        """
        if n_ic_timesteps == 0:
            raise RuntimeError("No initial condition timesteps to remove.")
        return self.__class__(
            {k: v[:, n_ic_timesteps:] for k, v in self.data.items()},
            time=self.time.isel(time=slice(n_ic_timesteps, None)),
            horizontal_dims=self.horizontal_dims,
        )

    def subset_names(self: SelfType, names: Collection[str]) -> SelfType:
        """
        Subset the data to only include the given names.
        """
        return self.__class__(
            {k: v for k, v in self.data.items() if k in names},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
        )

    def get_start(
        self: SelfType, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticState:
        """
        Get the initial condition state.
        """
        return PrognosticState(
            self.subset_names(prognostic_names).select_time_slice(
                slice(0, n_ic_timesteps)
            )
        )

    def get_end(
        self: SelfType, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticState:
        """
        Get the final state which can be used as a new initial condition.
        """
        return PrognosticState(
            self.subset_names(prognostic_names).select_time_slice(
                slice(-n_ic_timesteps, None)
            )
        )

    def select_time_slice(self: SelfType, time_slice: slice) -> SelfType:
        """
        Select a window of data from the batch.
        """
        return self.__class__(
            {k: v[:, time_slice] for k, v in self.data.items()},
            time=self.time[:, time_slice],
            horizontal_dims=self.horizontal_dims,
        )

    def prepend(self: SelfType, initial_condition: PrognosticState) -> SelfType:
        """
        Prepend the initial condition to the data.
        """
        initial_batch_data = initial_condition.as_batch_data()
        filled_data = {**initial_batch_data.data}
        example_tensor = list(initial_batch_data.data.values())[0]
        state_data_device = list(self.data.values())[0].device
        for k in self.data:
            if k not in filled_data:
                filled_data[k] = torch.full_like(example_tensor, fill_value=np.nan)
        return self.__class__(
            data={
                k: torch.cat([filled_data[k].to(state_data_device), v], dim=1)
                for k, v in self.data.items()
            },
            time=xr.concat([initial_batch_data.time, self.time], dim="time"),
            horizontal_dims=self.horizontal_dims,
        )


@dataclasses.dataclass
class PairedData:
    """A container for the data and time coordinate of a batch, with paired
    prediction and target data.
    """

    prediction: TensorMapping
    target: TensorMapping
    time: xr.DataArray

    @classmethod
    def from_batch_data(
        cls,
        prediction: BatchData,
        target: BatchData,
    ) -> "PairedData":
        if not np.all(prediction.time.values == target.time.values):
            raise ValueError("Prediction and target time coordinate must be the same.")
        return PairedData(prediction.data, target.data, prediction.time)

    @classmethod
    def new_on_device(
        cls,
        prediction: TensorMapping,
        target: TensorMapping,
        time: xr.DataArray,
    ) -> "PairedData":
        device = get_device()
        _check_device(prediction, device)
        _check_device(target, device)
        return PairedData(prediction, target, time)

    @classmethod
    def new_on_cpu(
        cls,
        prediction: TensorMapping,
        target: TensorMapping,
        time: xr.DataArray,
    ) -> "PairedData":
        _check_device(prediction, torch.device("cpu"))
        _check_device(target, torch.device("cpu"))
        return PairedData(prediction, target, time)


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

    @property
    def loader(self) -> DataLoader[BatchData]:
        def on_device(batch: BatchData) -> BatchData:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
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
            f"{name} data: {self._n_samples} samples, " f"{self._n_batches} batches"
        )
        logging.info(f"{name} data: first sample's initial time: {self._first_time}")
        logging.info(f"{name} data: last sample's initial time: {self._last_time}")


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
            sampler: Optional sampler for the data loader. Provided to allow support for
                distributed training.

        Note:
            While input data can be on any device, all data exposed from this class
            will be on the current device.
        """
        self._loader = loader
        self._properties = properties.to_device()
        self._sampler = sampler
        self._batch_size: Optional[int] = None

    @property
    def loader(self) -> DataLoader[BatchData]:
        def on_device(batch: BatchData) -> BatchData:
            return batch.to_device()

        return SizedMap(on_device, self._loader)

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._properties.variable_metadata

    @property
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
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
    def grid(self) -> Literal["equiangular", "legendre-gauss", "healpix"]:
        return self.horizontal_coordinates.grid

    @property
    def gridded_operations(self) -> GriddedOperations:
        return self.horizontal_coordinates.gridded_operations

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
