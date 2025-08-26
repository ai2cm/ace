import dataclasses
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, TypeVar

import cftime
import numpy as np
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.core.device import get_device
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
        labels: Labels for each sample in the batch.
        horizontal_dims: Horizontal dimensions of the data. Used for writing to
            netCDF files.
    """

    data: TensorMapping
    time: xr.DataArray
    labels: list[set[str]]
    horizontal_dims: list[str] = dataclasses.field(
        default_factory=lambda: ["lat", "lon"]
    )

    @classmethod
    def new_for_testing(
        cls,
        names: Iterable[str],
        n_samples: int = 2,
        n_timesteps: int = 10,
        t_initial: cftime.datetime = cftime.datetime(2020, 1, 1),
        freq="6h",
        calendar="julian",
        img_shape: tuple[int, ...] = (9, 18),
        horizontal_dims: list[str] = ["lat", "lon"],
        labels: list[set[str]] | None = None,
        device: torch.device | None = None,
    ) -> "BatchData":
        """
        Create a new batch data object for testing.

        Args:
            names: The names of the variables to create.
            n_samples: The number of samples to create.
            n_timesteps: The number of timesteps to create.
            t_initial: The initial time.
            freq: The frequency of the time steps.
            calendar: The calendar of the time steps.
            img_shape: The shape of the horizontal dimensions of the data.
            horizontal_dims: The horizontal dimensions of the data.
            labels: The labels of the data.
            device: The device to create the data on. By default, the device is
                determined by the global device specified by get_device().
        """
        if device is None:
            device = get_device()
        time = xr.DataArray(
            data=xr.date_range(
                start=t_initial,
                periods=n_timesteps,
                freq=freq,
                calendar=calendar,
                use_cftime=True,
            ),
            dims=["time"],
        ).drop_vars(["time"])
        sample_times = xr.concat([time] * n_samples, dim="sample")
        if labels is None:
            labels = [set() for _ in range(n_samples)]
        return BatchData(
            data={
                k: torch.randn(n_samples, n_timesteps, *img_shape).to(device)
                for k in names
            },
            time=sample_times,
            labels=labels,
            horizontal_dims=horizontal_dims,
        )

    @property
    def dims(self) -> list[str]:
        return ["sample", "time"] + self.horizontal_dims

    @property
    def n_timesteps(self) -> int:
        return self.time["time"].values.size

    def to_device(self) -> "BatchData":
        return self.__class__(
            data={k: v.to(get_device()) for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            labels=self.labels,
        )

    def to_cpu(self) -> "BatchData":
        return self.__class__(
            data={k: v.cpu() for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            labels=self.labels,
        )

    @classmethod
    def _get_kwargs(cls, horizontal_dims: list[str] | None) -> dict[str, Any]:
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
        labels: list[set[str]],
        horizontal_dims: list[str] | None = None,
    ) -> "BatchData":
        _check_device(data, torch.device("cpu"))
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData(
            data=data,
            time=time,
            labels=labels,
            **kwargs,
        )

    @classmethod
    def new_on_device(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        labels: list[set[str]],
        horizontal_dims: list[str] | None = None,
    ) -> "BatchData":
        """
        Move the data to the current global device specified by get_device().
        """
        _check_device(data, get_device())
        kwargs = cls._get_kwargs(horizontal_dims)
        return BatchData(
            data=data,
            time=time,
            labels=labels,
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
        samples: Sequence[tuple[TensorMapping, xr.DataArray, set[str]]],
        sample_dim_name: str = "sample",
        horizontal_dims: list[str] | None = None,
    ) -> "BatchData":
        sample_data, sample_times, sample_labels = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_time = xr.concat(sample_times, dim=sample_dim_name)
        return BatchData.new_on_cpu(
            data=batch_data,
            time=batch_time,
            labels=list(sample_labels),
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
            labels=self.labels,
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
            labels=self.labels,
        )

    def subset_names(self: SelfType, names: Collection[str]) -> SelfType:
        """
        Subset the data to only include the given names.
        """
        return self.__class__(
            {k: v for k, v in self.data.items() if k in names},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            labels=self.labels,
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
            labels=self.labels,
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
            labels=self.labels,
        )


@dataclasses.dataclass
class PairedData:
    """A container for the data and time coordinate of a batch, with paired
    prediction and target data.
    """

    prediction: TensorMapping
    reference: TensorMapping
    labels: list[set[str]]
    time: xr.DataArray

    @property
    def forcing(self) -> TensorMapping:
        return {k: v for k, v in self.reference.items() if k not in self.prediction}

    @property
    def target(self) -> TensorMapping:
        return {k: v for k, v in self.reference.items() if k in self.prediction}

    @classmethod
    def from_batch_data(
        cls,
        prediction: BatchData,
        reference: BatchData,
    ) -> "PairedData":
        if not np.all(prediction.time.values == reference.time.values):
            raise ValueError("Prediction and target time coordinate must be the same.")
        return PairedData(
            prediction=prediction.data,
            reference=reference.data,
            labels=prediction.labels,
            time=prediction.time,
        )

    @classmethod
    def new_on_device(
        cls,
        prediction: TensorMapping,
        reference: TensorMapping,
        labels: list[set[str]],
        time: xr.DataArray,
    ) -> "PairedData":
        device = get_device()
        _check_device(prediction, device)
        _check_device(reference, device)
        return PairedData(
            prediction=prediction,
            reference=reference,
            labels=labels,
            time=time,
        )

    @classmethod
    def new_on_cpu(
        cls,
        prediction: TensorMapping,
        reference: TensorMapping,
        labels: list[set[str]],
        time: xr.DataArray,
    ) -> "PairedData":
        _check_device(prediction, torch.device("cpu"))
        _check_device(reference, torch.device("cpu"))
        return PairedData(
            prediction=prediction,
            reference=reference,
            labels=labels,
            time=time,
        )
