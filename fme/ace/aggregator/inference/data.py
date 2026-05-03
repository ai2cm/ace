from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Any, Protocol

import cftime
import torch
import xarray as xr

from fme.core.typing_ import TensorMapping


class Unavailable(ValueError):
    """Raised when accessing an InferenceBatchData field that was not provided."""


def make_dummy_time(n_sample: int, n_time: int) -> xr.DataArray:
    base = cftime.DatetimeProlepticGregorian(2000, 1, 1)
    times = [base + datetime.timedelta(hours=6 * i) for i in range(n_time)]
    return xr.DataArray(
        [times for _ in range(n_sample)],
        dims=["sample", "time"],
    )


class InferenceBatchData:
    """All data available for a single batch during inference.

    Attributes:
        prediction: Denormalized prediction tensors.
        prediction_norm: Normalized prediction tensors (may be unavailable).
        target: Denormalized target tensors (may be unavailable).
        target_norm: Normalized target tensors (may be unavailable).
        time: Datetime array of shape (sample, time).
        i_time_start: Global timestep offset for this batch.
    """

    def __init__(
        self,
        *,
        prediction: TensorMapping,
        time: xr.DataArray,
        i_time_start: int,
        prediction_norm: TensorMapping | None = None,
        target: TensorMapping | None = None,
        target_norm: TensorMapping | None = None,
    ):
        self._prediction = prediction
        self._prediction_norm = prediction_norm
        self._target = target
        self._target_norm = target_norm
        self._time = time
        self._i_time_start = i_time_start

    def replace(self, **kwargs: Any) -> InferenceBatchData:
        defaults = {
            "prediction": self._prediction,
            "prediction_norm": self._prediction_norm,
            "target": self._target,
            "target_norm": self._target_norm,
            "time": self._time,
            "i_time_start": self._i_time_start,
        }
        defaults.update(kwargs)
        return InferenceBatchData(**defaults)

    @property
    def prediction(self) -> TensorMapping:
        return self._prediction

    @property
    def prediction_norm(self) -> TensorMapping:
        if self._prediction_norm is None:
            raise Unavailable(
                "prediction_norm is not available; "
                "it was not provided when constructing this InferenceBatchData."
            )
        return self._prediction_norm

    @property
    def target(self) -> TensorMapping:
        if self._target is None:
            raise Unavailable(
                "target is not available; "
                "it was not provided when constructing this InferenceBatchData."
            )
        return self._target

    @property
    def target_norm(self) -> TensorMapping:
        if self._target_norm is None:
            raise Unavailable(
                "target_norm is not available; "
                "it was not provided when constructing this InferenceBatchData."
            )
        return self._target_norm

    @property
    def has_prediction_norm(self) -> bool:
        return self._prediction_norm is not None

    @property
    def has_target(self) -> bool:
        return self._target is not None

    @property
    def has_target_norm(self) -> bool:
        return self._target_norm is not None

    @property
    def time(self) -> xr.DataArray:
        return self._time

    @property
    def i_time_start(self) -> int:
        return self._i_time_start

    @classmethod
    def new_test_data(
        cls,
        names: Sequence[str] = ("a",),
        shape: tuple[int, ...] = (2, 3, 4, 8),
        target: bool = False,
        time: xr.DataArray | None = None,
        i_time_start: int = 0,
    ) -> InferenceBatchData:
        if time is None:
            time = make_dummy_time(n_sample=shape[0], n_time=shape[1])
        prediction = {name: torch.randn(*shape) for name in names}
        if target:
            target_data: TensorMapping | None = {
                name: torch.randn(*shape) for name in names
            }
            target_norm: TensorMapping | None = {
                name: torch.randn(*shape) for name in names
            }
        else:
            target_data = None
            target_norm = None
        return cls(
            prediction=prediction,
            prediction_norm={name: torch.randn(*shape) for name in names},
            target=target_data,
            target_norm=target_norm,
            time=time,
            i_time_start=i_time_start,
        )


class SubAggregator(Protocol):
    def record_batch(self, data: InferenceBatchData) -> None: ...

    def get_logs(self, label: str) -> dict[str, Any]: ...

    def get_dataset(self) -> xr.Dataset: ...


class TimeSeriesLogs(Protocol):
    def get_logs(self, label: str, step_slice: slice = ...) -> dict[str, Any]: ...
