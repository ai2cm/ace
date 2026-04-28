from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class SubAggregator(Protocol):
    def record_batch(self, data: InferenceBatchData) -> None: ...

    def get_logs(self, label: str) -> dict[str, Any]: ...

    def get_dataset(self) -> xr.Dataset: ...
