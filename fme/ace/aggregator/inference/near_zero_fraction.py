import dataclasses
from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .build_context import MetricBuildContext, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator

AreaWeightedMean = Callable[..., torch.Tensor]


@dataclasses.dataclass
class NearZeroFractionMetricConfig:
    """Area-weighted fraction of cells at or below a (small, non-negative) ``eps``.

    Reports, per variable, the area-weighted fraction of cells whose value is
    ``<= eps`` for the prediction, and its difference from the target. For
    precipitation (``PRATEsfc``) with a small ``eps`` this is the fraction of
    near-zero (dry / drizzle) cells, which the existing RMSE/bias/histogram
    metrics do not surface directly — useful for diagnosing a model's dry-cell /
    drizzle behaviour against the target distribution. Disabled by default.

    ``eps`` is required to be non-negative (when ``enabled``). A value of exactly
    ``0`` is permitted: when a corrector clamps a variable to a physical floor
    (e.g. precipitation at zero) the at-or-below-zero fraction is a meaningful
    exact-dry-cell count, not a degenerate edge case.

    Parameters:
        variables: variables to compute the metric for (e.g. ``["PRATEsfc"]``).
            Must be non-empty when ``enabled``.
        eps: a cell counts toward the fraction when its value is ``<= eps``.
            Must be ``>= 0`` when ``enabled``. Applied to any variable not listed
            in ``per_variable_eps``.
        per_variable_eps: optional per-variable ``eps`` overriding ``eps`` for the
            named variables. Each must be ``>= 0`` when ``enabled``.
        name: log prefix and wandb key prefix.
        include_maps: if True, also log 2D maps of the per-cell time-mean
            at-or-below-``eps`` fraction: a side-by-side generated/target
            comparison and the generated-minus-target error map. Defaults to
            False.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built.
    """

    variables: list[str] = dataclasses.field(default_factory=list)
    eps: float = 0.0
    per_variable_eps: dict[str, float] = dataclasses.field(default_factory=dict)
    name: str = "near_zero_fraction"
    include_maps: bool = False
    enabled: bool = False
    strict: bool = True

    def __post_init__(self):
        if not self.enabled:
            return
        if not self.variables:
            raise ValueError(
                "NearZeroFractionMetricConfig is enabled but no variables were "
                "given; specify the variables to compute the metric for."
            )
        if self.eps < 0:
            raise ValueError(
                f"NearZeroFractionMetricConfig.eps must be >= 0, got {self.eps}."
            )
        negative = {
            var: value for var, value in self.per_variable_eps.items() if value < 0
        }
        if negative:
            raise ValueError(
                "NearZeroFractionMetricConfig.per_variable_eps values must "
                f"be >= 0, got {negative}."
            )

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        agg: SubAggregator = NearZeroFractionAggregator(
            area_weighted_mean=ctx.ops.area_weighted_mean,
            eps=self.eps,
            per_variable_eps=self.per_variable_eps,
            include_maps=self.include_maps,
            variable_metadata=ctx.variable_metadata,
            horizontal_dims=ctx.horizontal_coordinates.dims,
        )
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))


class NearZeroFractionAggregator:
    """Accumulates the area-weighted fraction of cells with value ``<= eps``.

    For each variable it reports the predicted fraction and, when a target is
    available, its difference from the target fraction (gen - target), averaged
    over all sample/time entries seen. The initial-condition timestep of the
    first batch (``i_time_start == 0``) is excluded, since it is prescribed
    rather than predicted (as in :class:`TimeMeanAggregator`). The metric name is
    supplied by the caller as the ``label`` prefix in :meth:`get_logs`.

    When ``include_maps`` is set it additionally accumulates the per-cell
    time-mean at-or-below-``eps`` fraction (the fraction of sample/time
    entries for which each cell is ``<= eps``) and, in :meth:`get_logs`,
    plots a side-by-side generated/target comparison plus the
    generated-minus-target error map. The maps are plain per-cell fractions
    (not area-weighted); the scalar fraction is the area-weighted mean of the
    same indicator, so the two agree only when read back with area weighting.
    """

    def __init__(
        self,
        area_weighted_mean: AreaWeightedMean,
        eps: float = 0.0,
        per_variable_eps: dict[str, float] | None = None,
        include_maps: bool = False,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        horizontal_dims: Sequence[str] | None = None,
    ):
        self._area_weighted_mean = area_weighted_mean
        self._eps = eps
        self._per_variable_eps = per_variable_eps or {}
        self._include_maps = include_maps
        self._variable_metadata = variable_metadata or {}
        self._horizontal_dims = (
            list(horizontal_dims) if horizontal_dims else ["lat", "lon"]
        )
        self._dist = Distributed.get_instance()
        self._gen_sum: dict[str, torch.Tensor] = {}
        self._gen_count: dict[str, int] = {}
        self._target_sum: dict[str, torch.Tensor] = {}
        self._target_count: dict[str, int] = {}
        # per-cell [lat, lon] sums of the at-or-below-eps indicator
        self._gen_map_sum: dict[str, torch.Tensor] = {}
        self._gen_map_count: dict[str, int] = {}
        self._target_map_sum: dict[str, torch.Tensor] = {}
        self._target_map_count: dict[str, int] = {}

    def _eps_for(self, name: str) -> float:
        return self._per_variable_eps.get(name, self._eps)

    def _accumulate(
        self,
        data: TensorMapping,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
        map_sums: dict[str, torch.Tensor],
        map_counts: dict[str, int],
        ignore_initial: bool,
    ) -> None:
        time_dim = 1
        time_slice = slice(1, None) if ignore_initial else slice(None)
        for name, tensor in data.items():
            # Drop the initial-condition timestep of the first batch: it is
            # ground-truth-initialized, not predicted, so counting it would
            # pull the generated fraction toward the target (mirrors
            # TimeMeanAggregator's ignore_initial handling).
            tensor = tensor[:, time_slice]
            # NaN <= eps is False, so NaN cells count as "not below" rather than
            # propagating NaN; this metric targets never-NaN fields like
            # PRATEsfc.
            below = (tensor <= self._eps_for(name)).to(tensor.dtype)
            # area_weighted_mean reduces the horizontal dims, leaving the
            # (sample, time) leading dims; each entry is that snapshot's
            # area-weighted fraction at or below eps.
            frac = self._area_weighted_mean(below, name=name)
            sums[name] = sums.get(name, frac.new_zeros(())) + frac.sum()
            counts[name] = counts.get(name, 0) + frac.numel()
            if self._include_maps:
                # sum the indicator over the (sample, time) leading dims,
                # leaving a [lat, lon] map; divided by the count later this is
                # each cell's fraction of entries at or below eps.
                sample_dim = 0
                cell_sum = below.sum(dim=time_dim).sum(dim=sample_dim)
                if name in map_sums:
                    map_sums[name] = map_sums[name] + cell_sum
                else:
                    map_sums[name] = cell_sum
                map_counts[name] = map_counts.get(name, 0) + below.size(
                    sample_dim
                ) * below.size(time_dim)

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        ignore_initial = data.i_time_start == 0
        self._accumulate(
            data.prediction,
            self._gen_sum,
            self._gen_count,
            self._gen_map_sum,
            self._gen_map_count,
            ignore_initial,
        )
        if data.has_target:
            self._accumulate(
                data.target,
                self._target_sum,
                self._target_count,
                self._target_map_sum,
                self._target_map_count,
                ignore_initial,
            )

    def _reduced_means(
        self, sums: dict[str, torch.Tensor], counts: dict[str, int]
    ) -> dict[str, float]:
        means: dict[str, float] = {}
        for name in sorted(sums):
            local_mean = sums[name] / counts[name]
            means[name] = self._dist.reduce_mean(local_mean).item()
        return means

    def _reduced_maps(
        self, sums: dict[str, torch.Tensor], counts: dict[str, int]
    ) -> dict[str, torch.Tensor]:
        maps: dict[str, torch.Tensor] = {}
        for name in sorted(sums):
            local_map = sums[name] / counts[name]
            maps[name] = self._dist.reduce_mean(local_map)
        return maps

    def _caption_name_units(self, name: str) -> tuple[str, str]:
        if name in self._variable_metadata:
            return (
                self._variable_metadata[name].display_long_name(name),
                self._variable_metadata[name].display_units(),
            )
        return name, "unknown_units"

    def _map_logs(self) -> dict[str, Image]:
        gen_maps = self._reduced_maps(self._gen_map_sum, self._gen_map_count)
        target_maps = self._reduced_maps(self._target_map_sum, self._target_map_count)
        logs: dict[str, Image] = {}
        for name, gen_map in gen_maps.items():
            caption_name, units = self._caption_name_units(name)
            eps = self._eps_for(name)
            fraction_desc = (
                f"{caption_name} fraction of time at or below {eps:g} {units}"
            )
            if name in target_maps:
                target_map = target_maps[name]
                comparison_image = plot_paneled_data(
                    [[gen_map.cpu().numpy()], [target_map.cpu().numpy()]],
                    diverging=False,
                    caption=f"{fraction_desc}; (top) generated and (bottom) target",
                )
                error_image = plot_paneled_data(
                    [[(gen_map - target_map).cpu().numpy()]],
                    diverging=True,
                    caption=f"{fraction_desc} error (generated - target)",
                )
                logs[f"gen_target_map/{name}"] = comparison_image
                logs[f"error_map/{name}"] = error_image
            else:
                gen_image = plot_paneled_data(
                    [[gen_map.cpu().numpy()]],
                    diverging=False,
                    caption=f"{fraction_desc}; generated",
                )
                logs[f"gen_map/{name}"] = gen_image
        plt.close("all")
        return logs

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float | Image]:
        gen_means = self._reduced_means(self._gen_sum, self._gen_count)
        target_means = self._reduced_means(self._target_sum, self._target_count)
        logs: dict[str, float | Image] = {}
        for name, value in gen_means.items():
            logs[f"gen/{name}"] = value
            if name in target_means:
                logs[f"gen_minus_target/{name}"] = value - target_means[name]
        if self._include_maps:
            logs.update(self._map_logs())
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        if not self._include_maps:
            return xr.Dataset()
        gen_maps = self._reduced_maps(self._gen_map_sum, self._gen_map_count)
        target_maps = self._reduced_maps(self._target_map_sum, self._target_map_count)
        dims = self._horizontal_dims
        data: dict[str, xr.DataArray] = {}
        for name, gen_map in gen_maps.items():
            data[f"gen_map-{name}"] = xr.DataArray(gen_map.cpu(), dims=dims)
            if name in target_maps:
                target_map = target_maps[name]
                data[f"target_map-{name}"] = xr.DataArray(target_map.cpu(), dims=dims)
                data[f"error_map-{name}"] = xr.DataArray(
                    (gen_map - target_map).cpu(), dims=dims
                )
        return xr.Dataset(data)
