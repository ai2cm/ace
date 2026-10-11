import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Literal, get_args

import numpy as np
import torch
import xarray as xr

from fme.core.constants import EARTH_RADIUS
from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping

from .build_context import MetricBuildContext
from .data import InferenceBatchData, MetricBuildResult
from .reduced import (
    AreaWeightedFunction,
    AreaWeightedReducedMetric,
    AreaWeightedSingleTargetFunction,
    MeanMetric,
    SeriesData,
    compute_metric_on,
    data_to_table,
    get_series_data,
    series_data_to_dataset,
)
from .utils import hemisphere_weights

Region = Literal["global", "north", "south"]
Reduction = Literal["mean", "sum"]
SPHERE_AREA_M2 = 4 * math.pi * EARTH_RADIUS**2

Reduce = AreaWeightedSingleTargetFunction


def _get_reduce(
    ops: GriddedOperations,
    regional_weights: torch.Tensor | None,
    reduction: Reduction,
) -> Reduce:
    """Area-weighted mean, or integral in m**2 * units, over one region.

    regional_weights None means the whole (masked) sphere.
    """

    def reduce(tensors: TensorMapping) -> TensorDict:
        if regional_weights is None:
            if reduction == "mean":
                return ops.area_weighted_mean_dict(tensors)
            normalized = ops.area_weighted_sum_dict(tensors)
        else:
            if reduction == "mean":
                return ops.regional_area_weighted_mean_dict(tensors, regional_weights)
            normalized = ops.regional_area_weighted_sum_dict(tensors, regional_weights)
        return {k: SPHERE_AREA_M2 * v for k, v in normalized.items()}

    return reduce


def _difference(reduce: Reduce) -> AreaWeightedFunction:
    def metric(truth: TensorMapping, predicted: TensorMapping) -> TensorDict:
        return reduce({k: predicted[k] - truth[k] for k in truth})

    return metric


def _rmse(reduce: Reduce) -> AreaWeightedFunction:
    def metric(truth: TensorMapping, predicted: TensorMapping) -> TensorDict:
        squared = reduce({k: (predicted[k] - truth[k]) ** 2 for k in truth})
        return {k: torch.sqrt(v) for k, v in squared.items()}

    return metric


def _get_metric_functions(
    reduce: Reduce, reduction: Reduction
) -> dict[str, AreaWeightedFunction]:
    if reduction == "mean":
        return {
            "weighted_mean_gen": compute_metric_on("gen", reduce),
            "weighted_mean_target": compute_metric_on("target", reduce),
            "weighted_bias": _difference(reduce),
            "weighted_rmse": _rmse(reduce),
        }
    return {
        "weighted_sum_gen": compute_metric_on("gen", reduce),
        "weighted_sum_target": compute_metric_on("target", reduce),
        "weighted_sum_bias": _difference(reduce),
    }


def _integral_metadata(metadata: VariableMetadata) -> VariableMetadata:
    units = None if metadata.units is None else f"({metadata.units}) m**2"
    long_name = (
        None if metadata.long_name is None else f"area integral of {metadata.long_name}"
    )
    return VariableMetadata(units=units, long_name=long_name)


def _get_regional_weights(
    horizontal_coordinates: HorizontalCoordinates, region: Region
) -> torch.Tensor | None:
    if region == "global":
        return None
    return hemisphere_weights(horizontal_coordinates, region).to(get_device())


class RegionalMeanAggregator:
    """Per-step area-weighted mean or integral of selected variables per region.

    Regional means divide by the masked (e.g. ocean) area of the region.
    Variables absent from a batch are skipped.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        regional_weights: Mapping[str, torch.Tensor | None],
        variables: Sequence[str],
        reduction: Reduction,
        target: Literal["norm", "denorm"],
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._variables = list(variables)
        self._reduction = reduction
        self._target = target
        self._dist = Distributed.get_instance()
        self._variable_metadata = variable_metadata or {}
        device = get_device()
        self._region_metrics: dict[str, dict[str, MeanMetric]] = {
            region: {
                metric_name: AreaWeightedReducedMetric(
                    device=device, compute_metric=fn, n_timesteps=n_timesteps
                )
                for metric_name, fn in _get_metric_functions(
                    _get_reduce(gridded_operations, weights, reduction), reduction
                ).items()
            }
            for region, weights in regional_weights.items()
        }
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData):
        if self._target == "norm":
            target_data, gen_data = data.target_norm, data.prediction_norm
        else:
            target_data, gen_data = data.target, data.prediction
        names = [v for v in self._variables if v in gen_data and v in target_data]
        self._n_batches += 1
        if len(names) == 0:
            return
        target = {k: target_data[k] for k in names}
        gen = {k: gen_data[k] for k in names}
        for metrics in self._region_metrics.values():
            for metric in metrics.values():
                metric.record(target=target, gen=gen, i_time_start=data.i_time_start)

    def _get_series_data(
        self, step_slice: slice | None = None
    ) -> dict[str, list[SeriesData]]:
        if self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        return {
            region: get_series_data(metrics, self._dist, step_slice)
            for region, metrics in self._region_metrics.items()
        }

    @torch.no_grad()
    def get_logs(self, label: str, step_slice: slice | None = None):
        series: dict[str, np.ndarray] = {
            f"{region}/{datum.get_wandb_key()}": datum.data
            for region, data in self._get_series_data(step_slice).items()
            for datum in data
        }
        init_step = 0 if step_slice is None else step_slice.start
        return {f"{label}/series": data_to_table(series, init_step)}

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        series = [
            SeriesData(f"{datum.metric_name}-{region}", datum.var_name, datum.data)
            for region, data in self._get_series_data().items()
            for datum in data
        ]
        if self._reduction == "sum":
            metadata: Mapping[str, VariableMetadata] = {
                k: _integral_metadata(v) for k, v in self._variable_metadata.items()
            }
        else:
            metadata = self._variable_metadata
        return series_data_to_dataset(series, metadata)


@dataclasses.dataclass
class RegionalMeanMetricConfig:
    """
    Per-step area-weighted mean or integral of ``variables`` over regions.

    Hemispheres split at the equator (north: lat > 0, south: lat <= 0) and need
    a lat/lon grid. ``reduction="sum"`` integrates over cell area in m**2, so a
    field in kg/m**2 gives kg and a fraction gives m**2. Logged per step as
    ``<name>/<region>/<metric>/<variable>``.

    Example: hemispheric frozen mass (kg) and sea-ice area (m**2)::

        regional_means:
          - variables: [frozen_mass_total_area, sea_ice_fraction]
            reduction: sum

    Parameters:
        variables: Variables to reduce. Required, since each region repeats
            the reduction for every variable.
        regions: Any of "global", "north", "south".
        reduction: "mean" (masked-area mean) or "sum" (area integral).
        target: Data space; "sum" requires "denorm".
        name: Log label. Defaults to "regional_<reduction>", with "_norm"
            appended for target="norm".
        enabled: Whether the metric is computed.
        strict: If True, raise rather than skip when the metric is not
            supported (e.g. hemispheres on a non-lat/lon grid).
    """

    variables: list[str]
    regions: list[Region] = dataclasses.field(
        default_factory=lambda: ["global", "north", "south"]
    )
    reduction: Reduction = "mean"
    target: Literal["denorm", "norm"] = "denorm"
    name: str | None = None
    enabled: bool = True
    strict: bool = True

    def __post_init__(self):
        if len(self.variables) == 0:
            raise ValueError("RegionalMeanMetricConfig.variables must be non-empty.")
        unknown = set(self.regions) - set(get_args(Region))
        if unknown:
            raise ValueError(f"Unknown regions: {sorted(unknown)}")
        if len(self.regions) == 0 or len(set(self.regions)) != len(self.regions):
            raise ValueError(
                f"regions must be non-empty and unique, got {self.regions}"
            )
        if self.reduction == "sum" and self.target == "norm":
            raise ValueError(
                "reduction='sum' requires target='denorm'; "
                "area integrals of normalized data have no physical meaning."
            )
        if self.name is None:
            suffix = "_norm" if self.target == "norm" else ""
            self.name = f"regional_{self.reduction}{suffix}"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        regional_weights: dict[str, torch.Tensor | None] = {
            region: _get_regional_weights(ctx.horizontal_coordinates, region)
            for region in self.regions
        }
        agg = RegionalMeanAggregator(
            ctx.ops,
            regional_weights=regional_weights,
            variables=self.variables,
            reduction=self.reduction,
            target=self.target,
            n_timesteps=ctx.n_timesteps,
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=agg, time_series=agg)
