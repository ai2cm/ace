from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Protocol

import torch
import xarray as xr

from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class InferenceBatchData:
    """All data available for a single batch during inference.

    Attributes:
        prediction: Denormalized prediction tensors.
        prediction_norm: Normalized prediction tensors.
        target: Denormalized target tensors, or None for single-series inference.
        target_norm: Normalized target tensors, or None for single-series inference.
        time: Datetime array of shape (sample, time), or None if not available.
        i_time_start: Global timestep offset for this batch.
    """

    prediction: TensorMapping
    prediction_norm: TensorMapping
    target: TensorMapping | None
    target_norm: TensorMapping | None
    time: xr.DataArray | None
    i_time_start: int

    @classmethod
    def new_test_data(
        cls,
        names: Sequence[str] = ("a",),
        shape: tuple[int, ...] = (2, 3, 4, 8),
        target: bool = False,
        time: xr.DataArray | None = None,
        i_time_start: int = 0,
    ) -> InferenceBatchData:
        prediction = {name: torch.randn(*shape) for name in names}
        if target:
            target_data = {name: torch.randn(*shape) for name in names}
            target_norm = {name: torch.randn(*shape) for name in names}
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
