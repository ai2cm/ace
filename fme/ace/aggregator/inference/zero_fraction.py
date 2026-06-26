import dataclasses
from collections.abc import Callable

import torch
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping

from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator

AreaWeightedMean = Callable[..., torch.Tensor]


@dataclasses.dataclass
class ZeroFractionMetricConfig:
    """Area-weighted fraction of cells at or below a threshold.

    Reports, per variable, the area-weighted fraction of cells whose value is
    ``<= threshold`` for the prediction, and its difference from the target. For
    precipitation (``PRATEsfc``) with the default threshold of 0 this is the
    fraction of exactly-zero (dry) cells, which the existing
    RMSE/bias/histogram metrics do not surface directly — useful for diagnosing
    a model's dry-cell / drizzle behaviour against the target distribution.
    Disabled by default.

    Parameters:
        variables: variables to compute the metric for (e.g. ``["PRATEsfc"]``).
            Must be non-empty when ``enabled``.
        threshold: a cell counts toward the fraction when its value is
            ``<= threshold``. Defaults to 0.0. Applied to any variable not
            listed in ``per_variable_threshold``.
        per_variable_threshold: optional per-variable thresholds overriding
            ``threshold`` for the named variables.
        name: log prefix and wandb key prefix.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built.
    """

    variables: list[str] = dataclasses.field(default_factory=list)
    threshold: float = 0.0
    per_variable_threshold: dict[str, float] = dataclasses.field(default_factory=dict)
    name: str = "zero_threshold_fraction"
    enabled: bool = False
    strict: bool = True

    def __post_init__(self):
        if self.enabled and not self.variables:
            raise ValueError(
                "ZeroFractionMetricConfig is enabled but no variables were given; "
                "specify the variables to compute the metric for."
            )

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = ZeroFractionAggregator(
            area_weighted_mean=ctx.ops.area_weighted_mean,
            threshold=self.threshold,
            per_variable_threshold=self.per_variable_threshold,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class ZeroFractionAggregator:
    """Accumulates the area-weighted fraction of cells with value ``<= threshold``.

    For each variable it reports the predicted fraction and, when a target is
    available, its difference from the target (gen - target), averaged over all
    sample/time entries seen. The metric name is supplied by the caller as the
    ``label`` prefix in :meth:`get_logs`.
    """

    def __init__(
        self,
        area_weighted_mean: AreaWeightedMean,
        threshold: float = 0.0,
        per_variable_threshold: dict[str, float] | None = None,
    ):
        self._area_weighted_mean = area_weighted_mean
        self._threshold = threshold
        self._per_variable_threshold = per_variable_threshold or {}
        self._dist = Distributed.get_instance()
        self._gen_sum: dict[str, torch.Tensor] = {}
        self._gen_count: dict[str, int] = {}
        self._target_sum: dict[str, torch.Tensor] = {}
        self._target_count: dict[str, int] = {}

    def _threshold_for(self, name: str) -> float:
        return self._per_variable_threshold.get(name, self._threshold)

    def _accumulate(
        self,
        data: TensorMapping,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
    ) -> None:
        for name, tensor in data.items():
            below = (tensor <= self._threshold_for(name)).to(tensor.dtype)
            # area_weighted_mean reduces the horizontal dims, leaving the
            # (sample, time) leading dims; each entry is that snapshot's
            # area-weighted fraction at or below the threshold.
            frac = self._area_weighted_mean(below, name=name)
            sums[name] = sums.get(name, frac.new_zeros(())) + frac.sum()
            counts[name] = counts.get(name, 0) + frac.numel()

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        self._accumulate(data.prediction, self._gen_sum, self._gen_count)
        if data.has_target:
            self._accumulate(data.target, self._target_sum, self._target_count)

    def _reduced_means(
        self, sums: dict[str, torch.Tensor], counts: dict[str, int]
    ) -> dict[str, float]:
        means: dict[str, float] = {}
        for name, total in sums.items():
            local_mean = total / counts[name]
            means[name] = self._dist.reduce_mean(local_mean).item()
        return means

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float]:
        gen_means = self._reduced_means(self._gen_sum, self._gen_count)
        target_means = self._reduced_means(self._target_sum, self._target_count)
        logs: dict[str, float] = {}
        for name, value in gen_means.items():
            logs[f"gen/{name}"] = value
            if name in target_means:
                logs[f"gen_minus_target/{name}"] = value - target_means[name]
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return xr.Dataset()
