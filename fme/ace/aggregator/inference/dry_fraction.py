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
class DryFractionMetricConfig:
    """Area-weighted fraction of cells at or below a threshold.

    TEMPORARY diagnostic for the ``keep_gradient_through_clamps`` investigation
    (exp/keep-gradient-clamps-validation). The precipitation ``force_positive``
    clamp produces exactly-zero cells; the hypothesis is that restoring gradient
    through that clamp (straight-through estimator) changes the model's dry-cell
    fraction / drizzle behaviour. This metric surfaces that scalar directly so
    the STE and baseline runs can be A/B compared. Existing metrics
    (RMSE/bias/histogram) do not report a dry-cell fraction.

    Parameters:
        variables: when set, only compute the metric for these variables
            (typically ``["PRATEsfc"]``).
        threshold: a cell counts as "dry" when its value is ``<= threshold``.
            Defaults to 0.0, i.e. exactly-zero for non-negative precipitation.
        name: log prefix and wandb key prefix.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built.
    """

    variables: list[str] | None = None
    threshold: float = 0.0
    name: str = "dry_fraction"
    enabled: bool = False
    strict: bool = True

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = DryFractionAggregator(
            area_weighted_mean=ctx.ops.area_weighted_mean,
            threshold=self.threshold,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class DryFractionAggregator:
    """Accumulates the area-weighted fraction of dry (``<= threshold``) cells.

    For each variable it reports the predicted dry fraction, the target dry
    fraction (when a target is available), and their difference (gen - target),
    averaged over all sample/time entries seen.
    """

    def __init__(self, area_weighted_mean: AreaWeightedMean, threshold: float = 0.0):
        self._area_weighted_mean = area_weighted_mean
        self._threshold = threshold
        self._dist = Distributed.get_instance()
        self._gen_sum: dict[str, torch.Tensor] = {}
        self._gen_count: dict[str, int] = {}
        self._target_sum: dict[str, torch.Tensor] = {}
        self._target_count: dict[str, int] = {}

    def _accumulate(
        self,
        data: TensorMapping,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
    ) -> None:
        for name, tensor in data.items():
            dry = (tensor <= self._threshold).to(tensor.dtype)
            # area_weighted_mean reduces the horizontal dims, leaving the
            # (sample, time) leading dims; each entry is that snapshot's
            # area-weighted dry fraction.
            frac = self._area_weighted_mean(dry, name=name)
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
            logs[f"dry_fraction/gen/{name}"] = value
        for name, value in target_means.items():
            logs[f"dry_fraction/target/{name}"] = value
            if name in gen_means:
                logs[f"dry_fraction/gen_minus_target/{name}"] = gen_means[name] - value
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return xr.Dataset()
