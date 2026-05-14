"""Inference SubAggregator for tendency variance ratio."""

import dataclasses
from typing import Any, Literal

import xarray as xr

from ..tendency_variance import TendencyVarianceAccumulator
from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator


class TendencyVarianceRatioAggregator:
    """Computes Var_spatial(gen tendency) / Var_spatial(target tendency).

    Expects paired prediction/target data with a time dimension.
    Temporal differences are computed within each batch window.
    """

    def __init__(self):
        self._inner = TendencyVarianceAccumulator()

    def record_batch(self, data: InferenceBatchData) -> None:
        if not data.has_target:
            return
        self._inner.record(data.prediction, data.target)

    def get_logs(self, label: str) -> dict[str, Any]:
        return self._inner.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


@dataclasses.dataclass
class TendencyVarianceRatioMetricConfig:
    """Metric config for the tendency variance ratio.

    Parameters:
        variables: Optional variable filter. If ``None``, all variables
            present in both prediction and target are included.
        name: Name used as the aggregator key in logs.
    """

    type: Literal["tendency_variance_ratio"] = "tendency_variance_ratio"
    variables: list[str] | None = None
    name: str = "tendency_variance"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = TendencyVarianceRatioAggregator()
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))
