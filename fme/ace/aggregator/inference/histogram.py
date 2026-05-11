import dataclasses
from typing import Literal

import torch
import xarray as xr

from fme.core.histogram import ComparedDynamicHistograms

from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator


@dataclasses.dataclass
class HistogramMetricConfig:
    type: Literal["histogram"] = "histogram"
    variables: list[str] | None = None
    name: str = "histogram"
    enabled: bool = True

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = HistogramAggregator()
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class HistogramAggregator:
    def __init__(self):
        self._histograms = ComparedDynamicHistograms(n_bins=200, percentiles=[99.9999])

    @torch.no_grad()
    def record_batch(
        self,
        data: InferenceBatchData,
    ):
        self._histograms.record_batch(data.target, data.prediction)

    @torch.no_grad()
    def get_logs(self, label: str):
        logs = self._histograms.get_wandb()
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return self._histograms.get_dataset()
