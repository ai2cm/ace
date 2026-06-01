import dataclasses

import torch
import xarray as xr

from fme.core.histogram import ComparedDynamicHistograms

from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator


@dataclasses.dataclass
class HistogramMetricConfig:
    """
    Parameters:
        variables: when set, filter the aggregator to these variables only —
            no histogram (plot or percentile) is emitted for variables not
            in this list.
        name: log prefix and wandb key prefix.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built.
        percentile_variables: when set, only these variables get the
            99.9999th-percentile (and any other configured percentile)
            scalar metrics emitted. The histogram plot is still emitted
            for every variable that passed ``variables``. Defaults to
            None (emit percentile keys for every variable that passed
            ``variables`` — current behaviour). Use to restrict the noisy
            tail-percentile keys to a small list (e.g. precipitation
            only) while keeping the histogram plot cohort-wide.
    """

    variables: list[str] | None = None
    name: str = "histogram"
    enabled: bool = False
    strict: bool = True
    percentile_variables: list[str] | None = None

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = HistogramAggregator(
            percentile_variables=self.percentile_variables
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class HistogramAggregator:
    def __init__(self, percentile_variables: list[str] | None = None):
        self._histograms = ComparedDynamicHistograms(n_bins=200, percentiles=[99.9999])
        self._percentile_variables: set[str] | None = (
            None if percentile_variables is None else set(percentile_variables)
        )

    @torch.no_grad()
    def record_batch(
        self,
        data: InferenceBatchData,
    ):
        self._histograms.record_batch(data.target, data.prediction)

    @torch.no_grad()
    def get_logs(self, label: str):
        logs = self._histograms.get_wandb(
            percentile_variables=self._percentile_variables
        )
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return self._histograms.get_dataset()
