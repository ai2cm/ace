import dataclasses
import warnings
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, TypeVar

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import default_collate

from fme.core.dataset.dataset import DatasetItem
from fme.core.device import get_device
from fme.core.labels import BatchLabels, LabelEncoding
from fme.core.tensors import repeat_interleave_batch_dim, unfold_ensemble_dim
from fme.core.typing_ import EnsembleTensorDict, TensorDict, TensorMapping

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
        epoch: The epoch number for the batch data.
        n_ensemble: The number of ensemble members represented in the batch data.
            This is a suggestion for the purpose of computing ensemble metrics.
            For example, an ensemble is something you would want to compute CRPS
            or ensemble mean RMSE over.
    """

    data: TensorMapping
    time: xr.DataArray
    labels: BatchLabels | None = None
    horizontal_dims: list[str] = dataclasses.field(
        default_factory=lambda: ["lat", "lon"]
    )
    epoch: int | None = None
    n_ensemble: int = 1

    @classmethod
    def new_for_testing(
        cls,
        names: Iterable[str],
        n_samples: int = 2,
        n_timesteps: int = 10,
        t_initial: cftime.datetime = cftime.datetime(2020, 1, 1),
        freq="6h",
        increment_times: bool = False,
        calendar="julian",
        img_shape: tuple[int, ...] = (9, 18),
        horizontal_dims: list[str] = ["lat", "lon"],
        epoch: int | None = 0,
        labels: BatchLabels | None = None,
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
            increment_times: Whether to increment the initial time for each sample
                when creating the time coordinate.
            calendar: The calendar of the time steps.
            img_shape: The shape of the horizontal dimensions of the data.
            horizontal_dims: The horizontal dimensions of the data.
            epoch: The epoch number for the batch data.
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
        if increment_times:
            sample_times = xr.concat(
                [time + pd.to_timedelta(freq) * i for i in range(n_samples)],
                dim="sample",
            )
        else:
            sample_times = xr.concat([time] * n_samples, dim="sample")
        return BatchData(
            data={
                k: torch.randn(n_samples, n_timesteps, *img_shape).to(device)
                for k in names
            },
            time=sample_times,
            labels=labels,
            horizontal_dims=horizontal_dims,
            epoch=epoch,
        )

    @property
    def dims(self) -> list[str]:
        return ["sample", "time"] + self.horizontal_dims

    @property
    def n_timesteps(self) -> int:
        return self.time["time"].values.size

    @property
    def ensemble_data(self) -> EnsembleTensorDict:
        """
        Add an explicit ensemble dimension to a data tensor dict.

        Returns:
            The tensor dict with an explicit ensemble dimension.
        """
        return unfold_ensemble_dim(TensorDict(self.data), n_ensemble=self.n_ensemble)

    def to_device(self) -> "BatchData":
        device = get_device()
        return self.__class__(
            data={k: v.to(device) for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
            labels=self.labels.to(device) if self.labels is not None else None,
        )

    def to_cpu(self) -> "BatchData":
        return self.__class__(
            data={k: v.cpu() for k, v in self.data.items()},
            time=self.time,
            horizontal_dims=self.horizontal_dims,
            epoch=self.epoch,
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
        epoch: int | None = None,
        labels: BatchLabels | None = None,
        horizontal_dims: list[str] | None = None,
        n_ensemble: int = 1,
    ) -> "BatchData":
        _check_device(data, torch.device("cpu"))
        kwargs = cls._get_kwargs(horizontal_dims)
        if isinstance(labels, list):
            warnings.warn(
                "Passing labels as a list is deprecated, and they will be ignored. "
                "Please pass a BatchLabels object "
                "instead, or None to indicate no label information.",
                DeprecationWarning,
            )
            labels = None
        return BatchData(
            data=data,
            time=time,
            labels=labels,
            epoch=epoch,
            n_ensemble=n_ensemble,
            **kwargs,
        )

    @classmethod
    def new_on_device(
        cls,
        data: TensorMapping,
        time: xr.DataArray,
        epoch: int | None = None,
        labels: BatchLabels | None = None,
        horizontal_dims: list[str] | None = None,
        n_ensemble: int = 1,
    ) -> "BatchData":
        """
        Move the data to the current global device specified by get_device().
        """
        _check_device(data, get_device())
        kwargs = cls._get_kwargs(horizontal_dims)
        if isinstance(labels, list):
            warnings.warn(
                "Passing labels as a list is deprecated, and they will be ignored. "
                "Please pass a BatchLabels object "
                "instead, or None to indicate no label information.",
                DeprecationWarning,
            )
            labels = None
        return BatchData(
            data=data,
            time=time,
            epoch=epoch,
            labels=labels,
            n_ensemble=n_ensemble,
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
        if (
            self.labels is not None
            and self.labels.tensor.shape[0] != self.time.shape[0]
        ):
            raise ValueError(
                "Labels tensor first dimension must match number of samples in "
                f"time. Got labels shape {self.labels.tensor.shape} and time shape "
                f"{self.time.shape}."
            )

    @classmethod
    def from_sample_tuples(
        cls,
        samples: Sequence[DatasetItem],
        sample_dim_name: str = "sample",
        horizontal_dims: list[str] | None = None,
        label_encoding: LabelEncoding | None = None,
    ) -> "BatchData":
        sample_data, sample_times, sample_labels, sample_epochs = zip(*samples)
        if not all(epoch == sample_epochs[0] for epoch in sample_epochs):
            raise ValueError("All samples must have the same epoch.")
        batch_data = default_collate(sample_data)
        batch_time = xr.concat(sample_times, dim=sample_dim_name)
        if label_encoding is None:
            if sample_labels[0] is not None:
                raise ValueError("label_encoding must be provided if labels are used.")
            labels = None
        else:
            labels = label_encoding.encode(list(sample_labels))
        return BatchData.new_on_cpu(
            data=batch_data,
            time=batch_time,
            labels=labels,
            horizontal_dims=horizontal_dims,
            epoch=sample_epochs[0],
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
            epoch=self.epoch,
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
            epoch=self.epoch,
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
            epoch=self.epoch,
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
            epoch=self.epoch,
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
            epoch=self.epoch,
            labels=self.labels,
        )

    def broadcast_ensemble(self: SelfType, n_ensemble: int) -> SelfType:
        """
        Broadcast a singleton ensemble to a new BatchData obj with n_ensemble members
        per ensemble.
        """
        if self.n_ensemble != 1:
            raise ValueError(
                "Can only broadcast singleton ensembles, but this BatchData has "
                f"n_ensemble={self.n_ensemble} and cannot be broadcast."
            )
        data = repeat_interleave_batch_dim(self.data, n_ensemble)
        time = xr.concat([self.time] * n_ensemble, dim="sample")
        if self.labels is None:
            labels = None
        else:
            labels = BatchLabels(
                torch.repeat_interleave(self.labels.tensor, n_ensemble, dim=0),
                self.labels.names,
            )
        return self.__class__(
            data={k: v.to(get_device()) for k, v in data.items()},
            time=time,
            horizontal_dims=self.horizontal_dims,
            labels=labels,
            epoch=self.epoch,
            n_ensemble=n_ensemble,
        )

    def pin_memory(self: SelfType) -> SelfType:
        """Used by torch.utils.data.DataLoader when pin_memory=True to page-lock
        tensors in CPU memory, resulting in faster transfers from CPU to GPU.

        See https://docs.pytorch.org/docs/stable/data.html#memory-pinning

        """
        self.data = {name: tensor.pin_memory() for name, tensor in self.data.items()}
        return self


@dataclasses.dataclass
class PairedData:
    """A container for the data and time coordinate of a batch, with paired
    prediction and target data.
    """

    prediction: TensorMapping
    reference: TensorMapping
    time: xr.DataArray
    labels: BatchLabels | None = None

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
        time: xr.DataArray,
        labels: BatchLabels | None = None,
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
        time: xr.DataArray,
        labels: BatchLabels | None = None,
    ) -> "PairedData":
        _check_device(prediction, torch.device("cpu"))
        _check_device(reference, torch.device("cpu"))
        return PairedData(
            prediction=prediction,
            reference=reference,
            labels=labels,
            time=time,
        )
