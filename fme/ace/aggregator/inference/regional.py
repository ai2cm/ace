import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.constants import EARTH_RADIUS
from fme.core.coordinates import HorizontalCoordinates, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .build_context import MetricBuildContext, MetricNotSupportedError
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
from .time_mean import TimeMeanAggregator
from .utils import Hemisphere, hemisphere_weights

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


class RegionalTimeMeanAggregator:
    """Time-mean maps cropped to polar caps, with hemispheric scalars.

    Maps show this rank's local rows with lat > lat_min (north) or
    lat < -lat_min (south). Scalars reduce the time-mean bias over the whole
    hemisphere (north: lat > 0, south: lat <= 0) with masked cell-area weights.
    Variables absent from a batch are skipped.
    """

    _captions = {
        "gen_map": (
            "{name} time-mean {where}; (top) generated and (bottom) target [{units}]"
        ),
        "bias_map": "{name} time-mean bias (generated - target) {where} [{units}]",
    }

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        lat: torch.Tensor,
        regional_weights: Mapping[Hemisphere, torch.Tensor],
        horizontal_dims: Sequence[str],
        variables: Sequence[str],
        lat_min: float,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = gridded_operations
        self._lat = lat.cpu()
        self._regional_weights = dict(regional_weights)
        self._horizontal_dims = list(horizontal_dims)
        self._variables = sorted(set(variables))
        self._lat_min = lat_min
        self._variable_metadata = variable_metadata or {}
        self._gen_agg = TimeMeanAggregator(gridded_operations)
        self._target_agg = TimeMeanAggregator(gridded_operations)
        self._recorded = False

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData):
        names = [
            v for v in self._variables if v in data.prediction and v in data.target
        ]
        if len(names) == 0:
            return
        gen = {k: data.prediction[k] for k in names}
        target = {k: data.target[k] for k in names}
        self._gen_agg.record_batch(data.replace(prediction=gen, prediction_norm=gen))
        self._target_agg.record_batch(
            data.replace(prediction=target, prediction_norm=target)
        )
        self._recorded = True

    def _crop_rows(self, region: Hemisphere) -> torch.Tensor:
        if region == "north":
            return self._lat > self._lat_min
        return self._lat < -self._lat_min

    def _caption(self, key: str, name: str, region: Hemisphere) -> str:
        metadata = self._variable_metadata.get(name)
        if metadata is None:
            caption_name, units = name, "unknown_units"
        else:
            caption_name = metadata.display_long_name(name)
            units = metadata.display_units()
        where = (
            f"north of {self._lat_min:g}N"
            if region == "north"
            else f"south of {self._lat_min:g}S"
        )
        return self._captions[key].format(name=caption_name, where=where, units=units)

    def _scalars(
        self, bias: torch.Tensor, weights: torch.Tensor, name: str
    ) -> tuple[float, float]:
        mean_bias = self._ops.regional_area_weighted_mean(bias, weights, name=name)
        mean_square = self._ops.regional_area_weighted_mean(bias**2, weights, name=name)
        return float(mean_bias), float(torch.sqrt(mean_square))

    def _get_maps(self) -> tuple[TensorDict, TensorDict]:
        return self._gen_agg.get_data(), self._target_agg.get_data()

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float | Image]:
        if not self._recorded:
            return {}
        gen, target = self._get_maps()
        logs: dict[str, float | Image] = {}
        for region, weights in self._regional_weights.items():
            rows = self._crop_rows(region)
            for name in sorted(gen):
                bias = gen[name] - target[name]
                mean_bias, rmse = self._scalars(bias, weights, name)
                logs[f"bias/{region}/{name}"] = mean_bias
                logs[f"rmse/{region}/{name}"] = rmse
                if not rows.any():  # this rank's chunk lies outside the cap
                    continue
                rows_device = rows.to(bias.device)
                logs[f"gen_map/{region}/{name}"] = plot_paneled_data(
                    [
                        [gen[name][rows_device].cpu().numpy()],
                        [target[name][rows_device].cpu().numpy()],
                    ],
                    diverging=False,
                    caption=self._caption("gen_map", name, region),
                )
                logs[f"bias_map/{region}/{name}"] = plot_paneled_data(
                    [[bias[rows_device].cpu().numpy()]],
                    diverging=True,
                    caption=self._caption("bias_map", name, region),
                )
                plt.close("all")
        return {f"{label}/{key}": value for key, value in logs.items()}

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        if not self._recorded:
            return xr.Dataset()
        gen, target = self._get_maps()
        lat_dim, lon_dim = self._horizontal_dims
        data: dict[str, xr.DataArray] = {}
        for region, weights in self._regional_weights.items():
            rows = self._crop_rows(region)
            region_lat_dim = f"{lat_dim}_{region}"
            for name in sorted(gen):
                attrs = self._variable_metadata.get(
                    name, VariableMetadata(long_name=name)
                ).as_attrs()
                bias = gen[name] - target[name]
                mean_bias, rmse = self._scalars(bias, weights, name)
                data[f"bias-{region}-{name}"] = xr.DataArray(mean_bias, attrs=attrs)
                data[f"rmse-{region}-{name}"] = xr.DataArray(rmse, attrs=attrs)
                if not rows.any():
                    continue
                rows_device = rows.to(bias.device)
                for key, value in [("gen_map", gen[name]), ("bias_map", bias)]:
                    data[f"{key}-{region}-{name}"] = xr.DataArray(
                        value[rows_device].cpu().numpy(),
                        dims=(region_lat_dim, lon_dim),
                        coords={region_lat_dim: self._lat[rows].numpy()},
                        attrs=attrs,
                    )
        return xr.Dataset(data)


@dataclasses.dataclass
class RegionalTimeMeanMetricConfig:
    """
    Time-mean generated and bias maps of ``variables`` cropped to polar caps,
    plus hemispheric bias and RMSE of the time mean. Needs a lat/lon grid.

    Maps are plate carree crops of lat > lat_min (north) or lat < -lat_min
    (south), logged as ``<name>/gen_map/<region>/<variable>`` (generated over
    target) and ``<name>/bias_map/<region>/<variable>``. Scalars
    ``<name>/bias/<region>/<variable>`` and ``<name>/rmse/<region>/<variable>``
    reduce over the whole hemisphere (north: lat > 0, south: lat <= 0) with
    masked cell-area weights.

    Example::

        regional_time_means:
          - variables: [sea_ice_fraction, frozen_mass_total_area]
            lat_min: 50

    Parameters:
        variables: Variables to map. Required, since each adds two time-mean
            accumulators.
        regions: Any of "north", "south".
        lat_min: Equatorward edge of the map crop in degrees.
        name: Log label.
        enabled: Whether the metric is computed.
        strict: If True, raise rather than skip on a non-lat/lon grid.
    """

    variables: list[str]
    regions: list[Hemisphere] = dataclasses.field(
        default_factory=lambda: ["north", "south"]
    )
    lat_min: float = 40.0
    name: str = "regional_time_mean"
    enabled: bool = True
    strict: bool = True

    def __post_init__(self):
        if len(self.variables) == 0:
            raise ValueError(
                "RegionalTimeMeanMetricConfig.variables must be non-empty."
            )
        unknown = set(self.regions) - set(get_args(Hemisphere))
        if unknown:
            raise ValueError(f"Unknown regions: {sorted(unknown)}")
        if len(self.regions) == 0 or len(set(self.regions)) != len(self.regions):
            raise ValueError(
                f"regions must be non-empty and unique, got {self.regions}"
            )
        if not 0 <= self.lat_min < 90:
            raise ValueError(f"lat_min must be in [0, 90), got {self.lat_min}")

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        coords = ctx.horizontal_coordinates
        if not isinstance(coords, LatLonCoordinates):
            raise MetricNotSupportedError(
                f"{self.name} metric requires LatLonCoordinates."
            )
        agg = RegionalTimeMeanAggregator(
            ctx.ops,
            lat=coords.localize().lat_1d,
            regional_weights={
                region: hemisphere_weights(coords, region).to(get_device())
                for region in self.regions
            },
            horizontal_dims=coords.dims,
            variables=self.variables,
            lat_min=self.lat_min,
            variable_metadata=ctx.variable_metadata,
        )
        return MetricBuildResult(aggregator=agg)
