import dataclasses
import datetime
from collections.abc import Mapping, Sequence
from typing import Any

import xarray as xr

from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.gridded_ops import GriddedOperations

from .data import InferenceBatchData, SubAggregator


class MetricNotSupportedError(Exception):
    """Raised when a metric cannot be built for the current grid type."""


class VariableFilterAdapter:
    """Wraps a sub-aggregator to filter InferenceBatchData to specified variables."""

    def __init__(self, inner: SubAggregator, variables: Sequence[str]):
        self._inner = inner
        self._variables = frozenset(variables)

    def record_batch(self, data: InferenceBatchData) -> None:
        vs = self._variables
        filtered = data.replace(
            prediction={k: v for k, v in data.prediction.items() if k in vs},
            prediction_norm=(
                {k: v for k, v in data.prediction_norm.items() if k in vs}
                if data.has_prediction_norm
                else None
            ),
            target=(
                {k: v for k, v in data.target.items() if k in vs}
                if data.has_target
                else None
            ),
            target_norm=(
                {k: v for k, v in data.target_norm.items() if k in vs}
                if data.has_target_norm
                else None
            ),
        )
        self._inner.record_batch(filtered)

    def get_logs(self, label: str, **kwargs: Any) -> dict[str, Any]:
        return self._inner.get_logs(label, **kwargs)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


@dataclasses.dataclass
class MetricBuildContext:
    """Runtime context passed to each metric's ``build()`` method.

    Groups the dataset and run information that individual metrics need
    to construct their aggregators, so that each ``build()`` signature
    stays simple while still having access to grid operations, coordinate
    metadata, reference data, and time-axis sizing.
    """

    ops: GriddedOperations
    horizontal_coordinates: HorizontalCoordinates
    n_timesteps: int
    n_ic_steps: int
    timestep: datetime.timedelta
    variable_metadata: Mapping[str, VariableMetadata] | None
    channel_mean_names: Sequence[str] | None
    monthly_reference_data: xr.Dataset | None
    time_mean_reference_data: xr.Dataset | None
    initial_time: xr.DataArray

    @property
    def n_forward_steps(self) -> int:
        return self.n_timesteps - self.n_ic_steps


def maybe_filter(agg: SubAggregator, variables: list[str] | None) -> SubAggregator:
    if variables is not None:
        return VariableFilterAdapter(agg, variables)
    return agg
