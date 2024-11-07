import abc
import dataclasses
import datetime
import logging
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterator,
    List,
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
from fme.core.device import get_device
from fme.core.generics.state import PrognosticStateABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping


class AnyDevice:
    """
    Indicates the object can be on any device.

    Used in type hints to indicate the device an object is on.
    """

    pass


class CPU(AnyDevice):
    """
    Indicates the object must be on the CPU.
    """

    pass


class CurrentDevice(AnyDevice):
    """
    Indicates the object must be on the current global device
    specified by fme.get_device().
    """

    pass


SelfType = TypeVar("SelfType", bound="BatchData")

DeviceType = TypeVar("DeviceType", bound=AnyDevice)


def _check_device(data: TensorMapping, device: torch.device):
    for v in data.values():
        if v.device != device:
            raise ValueError(f"data must be on {device}")


class AtmosphericCollateFn:
    def __init__(self, sigma_coordinates: SigmaCoordinates, horizontal_dims: List[str]):
        self.sigma_coordinates = sigma_coordinates.to("cpu")
        self.horizontal_dims = horizontal_dims

    def __call__(
        self, samples: Sequence[Tuple[TensorMapping, xr.DataArray]]
    ) -> "BatchData":
        """
        Collate function for use with PyTorch DataLoader. Needed since samples contain
        both tensor mapping and xarray time coordinates, the latter of which we do
        not want to convert to tensors.
        """
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim="sample")
        return get_atmospheric_batch_data(
            data=batch_data,
            times=batch_times,
            sigma_coordinates=self.sigma_coordinates,
            horizontal_dims=self.horizontal_dims,
        )


class AtmosphericDeriveFn:
    def __init__(
        self, sigma_coordinates: SigmaCoordinates, timestep: datetime.timedelta
    ):
        self.sigma_coordinates = sigma_coordinates.to(
            "cpu"
        )  # must be on cpu for multiprocessing fork context
        self.timestep = timestep

    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        return compute_derived_quantities(
            dict(data),
            sigma_coordinates=self.sigma_coordinates.to(get_device()),
            timestep=self.timestep,
            forcing_data=dict(forcing_data),
        )


@dataclasses.dataclass
class BatchData(Generic[DeviceType]):
    """A container for the data and time coordinates of a batch.

    Attributes:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch. To be used directly in training,
            validation, and inference.
        times: An array of times for each sample in the batch, concatenated along
            samples to make a batch. To be used in writing out inference
            predictions with time coordinates, not directly in ML.
        horizontal_dims: Horizontal dimensions of the data. Used for writing to
            netCDF files.
        derive_func: A function that takes a batch of data and a batch of forcing data
            and returns a batch of derived data. If not given, no derived variables
            are computed.
    """

    data: TensorMapping
    times: xr.DataArray
    horizontal_dims: List[str] = dataclasses.field(
        default_factory=lambda: ["lat", "lon"]
    )
    derive_func: Callable[
        [TensorMapping, TensorMapping], TensorDict
    ] = lambda x, _: dict(x)

    @property
    def dims(self) -> List[str]:
        return ["sample", "time"] + self.horizontal_dims

    def to_device(self) -> "BatchData[CurrentDevice]":
        return self.__class__.new_on_device(
            data={k: v.to(get_device()) for k, v in self.data.items()},
            times=self.times,
            horizontal_dims=self.horizontal_dims,
            derive_func=self.derive_func,
        )

    @classmethod
    def _get_kwargs(cls, horizontal_dims: Optional[List[str]]) -> Dict[str, Any]:
        if horizontal_dims is None:
            kwargs = {}
        else:
            kwargs = {"horizontal_dims": horizontal_dims}
        return kwargs

    @classmethod
    def new(
        cls,
        data: TensorMapping,
        times: xr.DataArray,
        horizontal_dims: Optional[List[str]] = None,
        derive_func: Callable[
            [TensorMapping, TensorMapping], TensorDict
        ] = lambda x, _: dict(x),
    ) -> "BatchData[AnyDevice]":
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData[AnyDevice](
            data=data,
            times=times,
            derive_func=derive_func,
            **kwargs,
        )

    @classmethod
    def new_on_cpu(
        cls,
        data: TensorMapping,
        times: xr.DataArray,
        horizontal_dims: Optional[List[str]] = None,
        derive_func: Callable[
            [TensorMapping, TensorMapping], TensorDict
        ] = lambda x, _: dict(x),
    ) -> "BatchData[CPU]":
        _check_device(data, torch.device("cpu"))
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData[CPU](
            data=data,
            times=times,
            derive_func=derive_func,
            **kwargs,
        )

    @classmethod
    def new_on_device(
        cls,
        data: TensorMapping,
        times: xr.DataArray,
        horizontal_dims: Optional[List[str]] = None,
        derive_func: Callable[
            [TensorMapping, TensorMapping], TensorDict
        ] = lambda x, _: dict(x),
    ) -> "BatchData[CurrentDevice]":
        """
        Move the data to the current global device specified by get_device().
        """
        _check_device(data, get_device())
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData[CurrentDevice](
            data=data,
            times=times,
            derive_func=derive_func,
            **kwargs,
        )

    def __post_init__(self):
        if len(self.times.shape) != 2:
            raise ValueError(
                "Expected times to have shape (n_samples, n_times), got shape "
                f"{self.times.shape}."
            )
        for k, v in self.data.items():
            if v.shape[:2] != self.times.shape[:2]:
                raise ValueError(
                    f"Data for variable {k} has shape {v.shape}, expected shape "
                    f"(n_samples, n_times) for times but got shape "
                    f"{self.times.shape}."
                )

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sample_dim_name: str = "sample",
        horizontal_dims: Optional[List[str]] = None,
        derive_func: Callable[
            [TensorMapping, TensorMapping], TensorDict
        ] = lambda x, _: dict(x),
    ) -> "BatchData[CPU]":
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim=sample_dim_name)
        return BatchData.new_on_cpu(
            data=batch_data,
            times=batch_times,
            derive_func=derive_func,
            horizontal_dims=horizontal_dims,
        )

    @classmethod
    def atmospheric_from_sample_tuples(
        cls,
        samples: Sequence[Tuple[TensorMapping, xr.DataArray]],
        sigma_coordinates: SigmaCoordinates,
        sample_dim_name: str = "sample",
        horizontal_dims: Optional[List[str]] = None,
    ) -> "BatchData[CPU]":
        """
        Collate function for use with PyTorch DataLoader. Needed since samples contain
        both tensor mapping and xarray time coordinates, the latter of which we do
        not want to convert to tensors.
        """
        sample_data, sample_times = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_times = xr.concat(sample_times, dim=sample_dim_name)
        return get_atmospheric_batch_data(
            data=batch_data,
            times=batch_times,
            sigma_coordinates=sigma_coordinates,
            horizontal_dims=horizontal_dims,
        )

    def compute_derived_variables(self: SelfType, forcing_data: SelfType) -> SelfType:
        """
        Compute derived variables from the data and forcing data.

        The forcing data must have the same times as the batch data.
        """
        if not np.all(forcing_data.times.values == self.times.values):
            raise ValueError("Forcing data must have the same times as the batch data.")
        derived_data = self.derive_func(self.data, forcing_data.data)
        return self.__class__(
            data={**self.data, **derived_data},
            times=self.times,
            horizontal_dims=self.horizontal_dims,
            derive_func=self.derive_func,
        )

    def remove_initial_condition(self: SelfType, n_ic_timesteps: int) -> SelfType:
        """
        Remove the initial condition timesteps from the data.
        """
        if n_ic_timesteps == 0:
            raise RuntimeError("No initial condition timesteps to remove.")
        return self.__class__(
            {k: v[:, n_ic_timesteps:] for k, v in self.data.items()},
            times=self.times.isel(time=slice(n_ic_timesteps, None)),
            horizontal_dims=self.horizontal_dims,
            derive_func=self.derive_func,
        )

    def subset_names(self: SelfType, names: Sequence[str]) -> SelfType:
        """
        Subset the data to only include the given names.
        """
        return self.__class__(
            {k: v for k, v in self.data.items() if k in names},
            times=self.times,
            horizontal_dims=self.horizontal_dims,
            derive_func=self.derive_func,
        )

    def get_start(
        self, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticStateABC["BatchData"]:
        """
        Get the initial condition state.
        """
        return _PrognosticState.from_batch_data_start(
            self, prognostic_names, n_ic_timesteps
        )

    def get_end(
        self, prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> PrognosticStateABC["BatchData"]:
        """
        Get the final state which can be used as a new initial condition.
        """
        return _PrognosticState.from_batch_data_end(
            self, prognostic_names, n_ic_timesteps
        )

    def prepend(
        self: SelfType, initial_condition: PrognosticStateABC[SelfType]
    ) -> SelfType:
        """
        Prepend the initial condition to the data.
        """
        initial_batch_data = initial_condition.as_state()
        filled_data = {**initial_batch_data.data}
        example_tensor = list(initial_batch_data.data.values())[0]
        state_data_device = list(self.data.values())[
            0
        ].device  # not the same as state.device_data[k].device
        for k in self.data:
            if k not in filled_data:
                filled_data[k] = torch.full_like(example_tensor, fill_value=np.nan)
        return self.__class__(
            data={
                k: torch.cat([filled_data[k].to(state_data_device), v], dim=1)
                for k, v in self.data.items()
            },
            times=xr.concat([initial_batch_data.times, self.times], dim="time"),
            derive_func=self.derive_func,
            horizontal_dims=self.horizontal_dims,
        )


def get_atmospheric_batch_data(
    data: TensorMapping,
    times: xr.DataArray,
    sigma_coordinates: SigmaCoordinates,
    horizontal_dims: Optional[List[str]] = None,
    # Note in practice return value is the cls type, just impossible to express
    # in python type hints. Use cast on the output if needed to get around this.
) -> BatchData[CPU]:
    """
    Get atmospheric batch data.

    Args:
        data: Data for each variable in each sample of shape (sample, time, ...),
            concatenated along samples to make a batch.
        times: An array of times of shape (sample, time) for each sample in the batch,
            concatenated along samples to make a batch.
        sigma_coordinates: Sigma coordinates for the data.
        horizontal_dims: Horizontal dimensions of the data. Used for writing to
            netCDF files.

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

    return BatchData.new_on_cpu(
        data=data,
        times=times,
        derive_func=AtmosphericDeriveFn(sigma_coordinates, timestep),
        horizontal_dims=horizontal_dims,
    )


DeviceArgType = TypeVar("DeviceArgType", bound=AnyDevice)


@dataclasses.dataclass
class PairedData(Generic[DeviceType]):
    """A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    prediction: TensorMapping
    target: TensorMapping
    times: xr.DataArray

    @classmethod
    def from_batch_data(
        cls,
        prediction: BatchData[DeviceArgType],
        target: BatchData[DeviceArgType],
    ) -> "PairedData[DeviceArgType]":
        if not np.all(prediction.times.values == target.times.values):
            raise ValueError("Prediction and target times must be the same.")
        return PairedData(prediction.data, target.data, prediction.times)

    @classmethod
    def new_on_device(
        cls,
        prediction: TensorMapping,
        target: TensorMapping,
        times: xr.DataArray,
    ) -> "PairedData[CurrentDevice]":
        device = get_device()
        _check_device(prediction, device)
        _check_device(target, device)
        return PairedData(prediction, target, times)

    @classmethod
    def new_on_cpu(
        cls,
        prediction: TensorMapping,
        target: TensorMapping,
        times: xr.DataArray,
    ) -> "PairedData[CPU]":
        _check_device(prediction, torch.device("cpu"))
        _check_device(target, torch.device("cpu"))
        return PairedData(prediction, target, times)


class _PrognosticState(PrognosticStateABC[BatchData]):
    """
    PrognosticStateABC implementation for BatchData.

    This should not be used directly, instead type hint as PrognosticStateABC[BatchData]
    or initialize from BatchData using the from_start or from_end methods.
    """

    def __init__(self, data: BatchData, _direct_init=True):
        """
        Initialize the state. Should not be used directly, instead use the
        initialization classmethods.

        Args:
            data: The data to initialize the state with.
            _direct_init: Whether the state was initialized directly from itself.
                Do not set this directly, as it allows you to bypass the checks in
                the true initialization methods.
        """
        if _direct_init:
            raise NotImplementedError(
                "Direct initialization not implemented, use from_start_state or "
                "from_end_state instead."
            )
        self._data = data

    def as_state(self) -> BatchData:
        return self._data

    @classmethod
    def from_batch_data_start(
        cls, state: "BatchData", prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> "_PrognosticState":
        return cls(
            BatchData(
                {
                    k: v[:, :n_ic_timesteps]
                    for k, v in state.data.items()
                    if k in prognostic_names
                },
                times=state.times[:, :n_ic_timesteps],
                derive_func=state.derive_func,
                horizontal_dims=state.horizontal_dims,
            ),
            _direct_init=False,
        )

    @classmethod
    def from_batch_data_end(
        cls, state: "BatchData", prognostic_names: Collection[str], n_ic_timesteps: int
    ) -> "_PrognosticState":
        return cls(
            BatchData(
                {
                    k: v[:, -n_ic_timesteps:]
                    for k, v in state.data.items()
                    if k in prognostic_names
                },
                times=state.times[:, -n_ic_timesteps:],
                derive_func=state.derive_func,
                horizontal_dims=state.horizontal_dims,
            ),
            _direct_init=False,
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


class GriddedData(GriddedDataABC[BatchData[CurrentDevice]]):
    """
    Data as required for pytorch training.

    The data is assumed to be gridded, and attributes are included for
    performing operations on gridded data.
    """

    def __init__(
        self,
        loader: DataLoader[BatchData[CPU]],
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
    def loader(self) -> DataLoader[BatchData[CurrentDevice]]:
        def on_device(batch: BatchData[CPU]) -> BatchData[CurrentDevice]:
            return batch.to_device()

        return map(on_device, self._loader)

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

    @property
    def n_forward_steps(self) -> int:
        return self._loader.dataset.n_forward_steps  # type: ignore

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
