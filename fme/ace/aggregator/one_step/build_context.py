from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Protocol

import xarray as xr

from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import EnsembleTensorDict, TensorMapping


class _Aggregator(Protocol):
    def get_logs(self, label: str) -> TensorMapping: ...

    def record_batch(
        self,
        loss: float,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
    ) -> None: ...

    def get_dataset(self) -> xr.Dataset: ...


class _EnsembleAggregator(Protocol):
    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
        i_time_start: int = ...,
    ) -> None: ...

    def get_logs(self, label: str) -> TensorMapping: ...


class MetricNotSupportedError(Exception):
    """Raised when a metric cannot be built for the current grid type."""


@dataclasses.dataclass
class OneStepBuildContext:
    ops: GriddedOperations
    horizontal_coordinates: HorizontalCoordinates
    variable_metadata: Mapping[str, VariableMetadata] | None
    channel_mean_names: Sequence[str] | None


@dataclasses.dataclass
class OneStepMetricBuildResult:
    deterministic: _Aggregator | None = None
    ensemble: _EnsembleAggregator | None = None
