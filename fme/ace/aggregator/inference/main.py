import dataclasses
import datetime
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import PairedData, PrognosticState
from fme.core.coordinates import HorizontalCoordinates, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.fill import SmoothFloodFill
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Table, WandB

from ..one_step.ensemble import (
    SelectStepEnsembleAggregator,
    get_one_step_ensemble_aggregator,
)
from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator
from .annual import GlobalMeanAnnualAggregator, PairedGlobalMeanAnnualAggregator
from .data import InferenceBatchData, SubAggregator, TimeSeriesLogs
from .enso import (
    EnsoCoefficientEvaluatorAggregator,
    LatLonRegion,
    PairedRegionalIndexAggregator,
    RegionalIndexAggregator,
)
from .histogram import HistogramAggregator
from .reduced import MeanAggregator, SingleTargetMeanAggregator
from .seasonal import SeasonalAggregator
from .spectrum import (
    PairedSphericalPowerSpectrumAggregator,
    SphericalPowerSpectrumAggregator,
)
from .time_mean import TimeMeanAggregator, TimeMeanEvaluatorAggregator
from .video import VideoAggregator
from .zonal_mean import ZonalMeanAggregator

wandb = WandB.get_instance()
APPROXIMATELY_TWO_YEARS = datetime.timedelta(days=730)
SLIGHTLY_LESS_THAN_FIVE_YEARS = datetime.timedelta(days=1800)
NINO34_LAT = (-5, 5)
NINO34_LON = (190, 240)


class _MetricNotSupportedError(Exception):
    """Raised when a metric cannot be built for the current grid type."""


class _OneStepMeanAdapter:
    """Adapts OneStepMeanAggregator to accept InferenceBatchData."""

    def __init__(self, inner: OneStepMeanAggregator):
        self._inner = inner

    def record_batch(self, data: InferenceBatchData) -> None:
        self._inner.record_batch(
            target_data=data.target,
            gen_data=data.prediction,
            target_data_norm=data.target_norm,
            gen_data_norm=data.prediction_norm,
            i_time_start=data.i_time_start,
        )

    def get_logs(self, label: str) -> dict[str, Any]:
        return self._inner.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


class _VariableFilterAdapter:
    """Wraps a sub-aggregator to filter InferenceBatchData to specified variables."""

    def __init__(self, inner: SubAggregator, variables: Sequence[str]):
        self._inner = inner
        self._variables = frozenset(variables)

    def record_batch(self, data: InferenceBatchData) -> None:
        vs = self._variables
        filtered = data.replace(
            prediction={k: v for k, v in data.prediction.items() if k in vs},
            prediction_norm={k: v for k, v in data.prediction_norm.items() if k in vs},
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
class _MetricBuildContext:
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


# ---------------------------------------------------------------------------
# Per-metric configuration dataclasses
# ---------------------------------------------------------------------------


def _maybe_filter(agg: SubAggregator, variables: list[str] | None) -> SubAggregator:
    if variables is not None:
        return _VariableFilterAdapter(agg, variables)
    return agg


@dataclasses.dataclass
class MeanMetricConfig:
    type: Literal["mean"] = "mean"
    variables: list[str] | None = None
    name: str | None = None
    target: Literal["denorm", "norm"] = "denorm"

    def __post_init__(self):
        if self.name is None:
            self.name = "mean_norm" if self.target == "norm" else "mean"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        agg: SubAggregator = MeanAggregator(
            ctx.ops,
            target=self.target,
            n_timesteps=ctx.n_timesteps,
            variable_metadata=ctx.variable_metadata,
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class StepMeanMetricConfig:
    step: int
    type: Literal["step_mean"] = "step_mean"
    variables: list[str] | None = None
    name: str | None = None
    target: Literal["denorm", "norm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            base = f"mean_step_{self.step}"
            self.name = f"{base}_norm" if self.target == "norm" else base

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        if self.step > ctx.n_forward_steps:
            raise ValueError(
                f"step_mean step {self.step} exceeds "
                f"n_forward_steps={ctx.n_forward_steps}"
            )
        target_time = self.step + ctx.n_ic_steps - 1
        is_norm = self.target == "norm"
        agg: SubAggregator = _OneStepMeanAdapter(
            OneStepMeanAggregator(
                ctx.ops,
                target_time=target_time,
                target=self.target,
                log_loss=False,
                include_bias=not is_norm,
                include_grad_mag_percent_diff=not is_norm,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class PowerSpectrumMetricConfig:
    type: Literal["power_spectrum"] = "power_spectrum"
    variables: list[str] | None = None
    name: str = "power_spectrum"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        try:
            agg: SubAggregator = PairedSphericalPowerSpectrumAggregator(
                gridded_operations=ctx.ops,
                nan_fill_fn=SmoothFloodFill(num_steps=4),
                report_plot=True,
                variable_metadata=ctx.variable_metadata,
            )
        except NotImplementedError as e:
            raise _MetricNotSupportedError(str(e)) from e
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class ZonalMeanMetricConfig:
    type: Literal["zonal_mean"] = "zonal_mean"
    variables: list[str] | None = None
    name: str = "zonal_mean"
    zonal_mean_max_size: int | bool = 4096

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        if ctx.ops.zonal_mean is None:
            raise _MetricNotSupportedError(
                "Zonal mean metric requires a grid type that supports zonal means."
            )
        agg: SubAggregator = ZonalMeanAggregator(
            zonal_mean=ctx.ops.zonal_mean,
            n_timesteps=ctx.n_timesteps,
            variable_metadata=ctx.variable_metadata,
            zonal_mean_max_size=self.zonal_mean_max_size,
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class VideoMetricConfig:
    type: Literal["video"] = "video"
    variables: list[str] | None = None
    name: str = "video"
    enable_extended_videos: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        if not isinstance(ctx.horizontal_coordinates, LatLonCoordinates):
            raise _MetricNotSupportedError("Video metric requires LatLonCoordinates.")
        agg: SubAggregator = VideoAggregator(
            n_timesteps=ctx.n_timesteps,
            enable_extended_videos=self.enable_extended_videos,
            variable_metadata=ctx.variable_metadata,
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class TimeMeanMetricConfig:
    type: Literal["time_mean"] = "time_mean"
    variables: list[str] | None = None
    name: str | None = None
    target: Literal["denorm", "norm"] = "denorm"
    reference_data: str | None = None
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = "time_mean_norm" if self.target == "norm" else "time_mean"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        is_norm = self.target == "norm"
        if self.reference_data is not None:
            ref = xr.open_dataset(self.reference_data, decode_timedelta=False)
        elif not is_norm:
            ref = ctx.time_mean_reference_data
        else:
            ref = None
        agg: SubAggregator = TimeMeanEvaluatorAggregator(
            ctx.ops,
            horizontal_dims=ctx.horizontal_coordinates.dims,
            target=self.target,
            variable_metadata=ctx.variable_metadata,
            reference_means=ref,
            channel_mean_names=(
                (self.channel_mean_names or ctx.channel_mean_names) if is_norm else None
            ),
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class HistogramMetricConfig:
    type: Literal["histogram"] = "histogram"
    variables: list[str] | None = None
    name: str = "histogram"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        agg: SubAggregator = HistogramAggregator()
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class SeasonalMetricConfig:
    type: Literal["seasonal"] = "seasonal"
    variables: list[str] | None = None
    name: str = "seasonal"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        agg: SubAggregator = SeasonalAggregator(
            ops=ctx.ops,
            variable_metadata=ctx.variable_metadata,
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class AnnualMetricConfig:
    type: Literal["annual"] = "annual"
    variables: list[str] | None = None
    name: str = "annual"
    reference_data: str | None = None

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> SubAggregator:
        if self.reference_data is not None:
            ref = xr.open_dataset(self.reference_data, decode_timedelta=False)
        else:
            ref = ctx.monthly_reference_data
        agg: SubAggregator = PairedGlobalMeanAnnualAggregator(
            ops=ctx.ops,
            timestep=ctx.timestep,
            variable_metadata=ctx.variable_metadata,
            monthly_reference_data=ref,
        )
        return _maybe_filter(agg, self.variables)


@dataclasses.dataclass
class EnsoIndexMetricConfig:
    type: Literal["enso_index"] = "enso_index"
    name: str = "enso_index"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> PairedRegionalIndexAggregator:
        if not isinstance(ctx.horizontal_coordinates, LatLonCoordinates):
            raise _MetricNotSupportedError(
                "enso_index metric requires LatLonCoordinates."
            )
        if not isinstance(ctx.ops, LatLonOperations):
            raise _MetricNotSupportedError(
                "enso_index metric requires LatLonOperations."
            )
        nino34_region = LatLonRegion(
            lat_bounds=NINO34_LAT,
            lon_bounds=NINO34_LON,
            lat=ctx.horizontal_coordinates.lat,
            lon=ctx.horizontal_coordinates.lon,
        )
        return PairedRegionalIndexAggregator(
            target_aggregator=RegionalIndexAggregator(
                regional_weights=nino34_region.regional_weights,
                regional_mean=ctx.ops.regional_area_weighted_mean,
            ),
            prediction_aggregator=RegionalIndexAggregator(
                regional_weights=nino34_region.regional_weights,
                regional_mean=ctx.ops.regional_area_weighted_mean,
            ),
        )


@dataclasses.dataclass
class EnsoCoefficientMetricConfig:
    type: Literal["enso_coefficient"] = "enso_coefficient"
    name: str = "enso_coefficient"

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: _MetricBuildContext) -> EnsoCoefficientEvaluatorAggregator:
        return EnsoCoefficientEvaluatorAggregator(
            ctx.initial_time,
            ctx.n_timesteps - 1,
            ctx.timestep,
            gridded_operations=ctx.ops,
            variable_metadata=ctx.variable_metadata,
        )


@dataclasses.dataclass
class EnsembleMetricConfig:
    step: int = 20
    type: Literal["ensemble"] = "ensemble"
    name: str | None = None
    log_mean_maps: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = f"ensemble_step_{self.step}"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: _MetricBuildContext) -> SelectStepEnsembleAggregator:
        return get_one_step_ensemble_aggregator(
            gridded_operations=ctx.ops,
            target_time=self.step,
            log_mean_maps=self.log_mean_maps,
            metadata=ctx.variable_metadata,
        )


MetricConfig = (
    MeanMetricConfig
    | StepMeanMetricConfig
    | PowerSpectrumMetricConfig
    | ZonalMeanMetricConfig
    | VideoMetricConfig
    | TimeMeanMetricConfig
    | HistogramMetricConfig
    | SeasonalMetricConfig
    | AnnualMetricConfig
    | EnsoIndexMetricConfig
    | EnsoCoefficientMetricConfig
    | EnsembleMetricConfig
)


@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    """
    Configuration for inference evaluator aggregator.

    Metrics can be configured explicitly via the ``metrics`` list, where each
    entry is a typed metric configuration (e.g. ``MeanMetricConfig``,
    ``StepMeanMetricConfig``).  When ``metrics`` is ``None``, a default set
    of metrics is computed at build time based on the grid type and run length.

    Parameters:
        metrics: Explicit list of metric configurations.  When ``None``, a
            default set is used.
        monthly_reference_data: Path to monthly reference data to compare against.
        time_mean_reference_data: Path to reference time means to compare against.
    """

    metrics: list[MetricConfig] | None = None
    monthly_reference_data: str | None = None
    time_mean_reference_data: str | None = None

    def __post_init__(self):
        if self.metrics is not None:
            names = [m.get_name() for m in self.metrics]
            seen: set[str] = set()
            duplicates: set[str] = set()
            for n in names:
                if n in seen:
                    duplicates.add(n)
                seen.add(n)
            if duplicates:
                raise ValueError(
                    f"Duplicate metric names: {sorted(duplicates)}. "
                    "Use the 'name' field to disambiguate."
                )

    @staticmethod
    def _default_metrics(
        ctx: _MetricBuildContext,
        n_ensemble_per_ic: int,
    ) -> list[MetricConfig]:
        """Compute default metrics based on runtime information."""
        metrics: list[MetricConfig] = [
            MeanMetricConfig(target="denorm"),
            MeanMetricConfig(target="norm"),
        ]

        if ctx.n_forward_steps >= 20:
            metrics.append(StepMeanMetricConfig(step=20, target="denorm"))
            metrics.append(StepMeanMetricConfig(step=20, target="norm"))

        metrics.extend(
            [
                PowerSpectrumMetricConfig(),
                ZonalMeanMetricConfig(),
                TimeMeanMetricConfig(target="denorm"),
                TimeMeanMetricConfig(target="norm"),
            ]
        )

        if n_ensemble_per_ic > 1:
            metrics.append(EnsembleMetricConfig(step=20))

        if ctx.n_timesteps * ctx.timestep > APPROXIMATELY_TWO_YEARS:
            metrics.append(AnnualMetricConfig())
            if isinstance(ctx.horizontal_coordinates, LatLonCoordinates) and isinstance(
                ctx.ops, LatLonOperations
            ):
                metrics.append(EnsoIndexMetricConfig())

        if ctx.n_timesteps * ctx.timestep > SLIGHTLY_LESS_THAN_FIVE_YEARS:
            metrics.append(EnsoCoefficientMetricConfig())

        return metrics

    def build(
        self,
        dataset_info: DatasetInfo,
        n_ic_steps: int,
        n_forward_steps: int,
        initial_time: xr.DataArray,
        normalize: Callable[[TensorMapping], TensorDict],
        output_dir: str | None = None,
        channel_mean_names: Sequence[str] | None = None,
        save_diagnostics: bool = True,
        n_ensemble_per_ic: int = 1,
        enable_time_series: bool = True,
    ) -> "InferenceEvaluatorAggregator":
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics.")
        if self.monthly_reference_data is None:
            monthly_reference_data = None
        else:
            monthly_reference_data = xr.open_dataset(
                self.monthly_reference_data, decode_timedelta=False
            )
        if self.time_mean_reference_data is None:
            time_mean_reference_data = None
        else:
            time_mean_reference_data = xr.open_dataset(
                self.time_mean_reference_data, decode_timedelta=False
            )

        n_timesteps = n_ic_steps + n_forward_steps
        ctx = _MetricBuildContext(
            ops=dataset_info.gridded_operations,
            horizontal_coordinates=dataset_info.horizontal_coordinates,
            n_timesteps=n_timesteps,
            n_ic_steps=n_ic_steps,
            timestep=dataset_info.timestep,
            variable_metadata=dataset_info.variable_metadata,
            channel_mean_names=channel_mean_names,
            monthly_reference_data=monthly_reference_data,
            time_mean_reference_data=time_mean_reference_data,
            initial_time=initial_time,
        )

        if self.metrics is not None:
            metrics = list(self.metrics)
        else:
            metrics = self._default_metrics(ctx, n_ensemble_per_ic)

        if not enable_time_series:
            metrics = [m for m in metrics if not isinstance(m, MeanMetricConfig)]

        is_explicit = self.metrics is not None
        aggregators: dict[str, SubAggregator] = {}
        time_series_aggregators: dict[str, TimeSeriesLogs] = {}
        ensemble_aggregators: dict[str, SelectStepEnsembleAggregator] = {}

        for metric in metrics:
            name = metric.get_name()
            try:
                if isinstance(metric, EnsembleMetricConfig):
                    ensemble_aggregators[name] = metric.build(ctx)
                    continue
                agg: SubAggregator = metric.build(ctx)
            except _MetricNotSupportedError:
                if is_explicit:
                    raise
                logging.warning(
                    f"{name} metric not supported for this grid type, " "omitting."
                )
                continue

            aggregators[name] = agg
            if isinstance(metric, MeanMetricConfig):
                time_series_aggregators[name] = agg  # type: ignore[assignment]

        return InferenceEvaluatorAggregator(
            aggregators=aggregators,
            time_series_aggregators=time_series_aggregators,
            coords=dataset_info.horizontal_coordinates.coords,
            n_ic_steps=n_ic_steps,
            normalize=normalize,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            n_ensemble_per_ic=n_ensemble_per_ic,
            ensemble_aggregators=ensemble_aggregators,
        )


@dataclasses.dataclass
class _LegacyStepMeanEntry:
    step: int
    name: str | None = None

    def get_name(self) -> str:
        return self.name or f"mean_step_{self.step}"


@dataclasses.dataclass
class LegacyFlagInferenceEvaluatorAggregatorConfig:
    """Backwards-compatible aggregator config that uses the old boolean-flag
    interface and translates it into the new typed-metric config at build time.
    """

    type: Literal["legacy"] = "legacy"
    log_histograms: bool = False
    log_video: bool = False
    log_extended_video: bool = False
    log_zonal_mean_images: bool | int = 4096
    log_seasonal_means: bool = False
    log_global_mean_time_series: bool = True
    log_global_mean_norm_time_series: bool = True
    monthly_reference_data: str | None = None
    time_mean_reference_data: str | None = None
    log_nino34_index: bool = True
    log_step_means: list[_LegacyStepMeanEntry] = dataclasses.field(
        default_factory=lambda: [_LegacyStepMeanEntry(step=20)]
    )

    def _build_metrics(
        self,
        n_forward_steps: int,
    ) -> list[MetricConfig]:
        metrics: list[MetricConfig] = []
        if self.log_global_mean_time_series:
            metrics.append(MeanMetricConfig(target="denorm"))
        if self.log_global_mean_norm_time_series:
            metrics.append(MeanMetricConfig(target="norm"))
        for entry in self.log_step_means:
            if entry.step <= n_forward_steps:
                metrics.append(
                    StepMeanMetricConfig(
                        step=entry.step, name=entry.get_name(), target="denorm"
                    )
                )
                metrics.append(
                    StepMeanMetricConfig(
                        step=entry.step,
                        name=f"{entry.get_name()}_norm",
                        target="norm",
                    )
                )
        metrics.append(PowerSpectrumMetricConfig())
        if self.log_zonal_mean_images:
            zonal_max = (
                self.log_zonal_mean_images
                if isinstance(self.log_zonal_mean_images, int)
                else 4096
            )
            metrics.append(ZonalMeanMetricConfig(zonal_mean_max_size=zonal_max))
        metrics.append(TimeMeanMetricConfig(target="denorm"))
        metrics.append(TimeMeanMetricConfig(target="norm"))
        if self.log_video:
            metrics.append(
                VideoMetricConfig(enable_extended_videos=self.log_extended_video)
            )
        if self.log_histograms:
            metrics.append(HistogramMetricConfig())
        if self.log_seasonal_means:
            metrics.append(SeasonalMetricConfig())
        return metrics

    def build(
        self,
        dataset_info: DatasetInfo,
        n_ic_steps: int,
        n_forward_steps: int,
        initial_time: xr.DataArray,
        normalize: Callable[[TensorMapping], TensorDict],
        output_dir: str | None = None,
        channel_mean_names: Sequence[str] | None = None,
        save_diagnostics: bool = True,
        n_ensemble_per_ic: int = 1,
        enable_time_series: bool = True,
    ) -> "InferenceEvaluatorAggregator":
        metrics = self._build_metrics(n_forward_steps)
        new_config = InferenceEvaluatorAggregatorConfig(
            metrics=metrics,
            monthly_reference_data=self.monthly_reference_data,
            time_mean_reference_data=self.time_mean_reference_data,
        )
        return new_config.build(
            dataset_info=dataset_info,
            n_ic_steps=n_ic_steps,
            n_forward_steps=n_forward_steps,
            initial_time=initial_time,
            normalize=normalize,
            output_dir=output_dir,
            channel_mean_names=channel_mean_names,
            save_diagnostics=save_diagnostics,
            n_ensemble_per_ic=n_ensemble_per_ic,
            enable_time_series=enable_time_series,
        )


class InferenceEvaluatorAggregator(
    InferenceAggregatorABC[PairedData | PrognosticState, PairedData]
):
    """
    Aggregates statistics for inference comparing a generated and target series.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        aggregators: dict[str, SubAggregator],
        time_series_aggregators: dict[str, TimeSeriesLogs],
        coords: Mapping[str, np.ndarray],
        n_ic_steps: int,
        normalize: Callable[[TensorMapping], TensorDict],
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        n_ensemble_per_ic: int = 1,
        ensemble_aggregators: dict[str, SelectStepEnsembleAggregator] | None = None,
    ):
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._aggregators = aggregators
        self._time_series_aggregators = time_series_aggregators
        self.n_ensemble_per_ic = n_ensemble_per_ic
        self._ensemble_aggregators = ensemble_aggregators or {}
        summary_aggregators: dict[str, SubAggregator | SelectStepEnsembleAggregator] = {
            name: agg
            for name, agg in aggregators.items()
            if name not in time_series_aggregators
        }
        if n_ensemble_per_ic > 1:
            summary_aggregators.update(self._ensemble_aggregators)
        self._summary_aggregators = summary_aggregators
        self._coords = coords
        self.n_ic_steps = n_ic_steps
        self._normalize = normalize
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        self._log_time_series = len(time_series_aggregators) > 0
        self._n_timesteps_seen = 0

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(
        self,
        data: PairedData,
    ) -> InferenceLogs:
        if len(data.prediction) == 0:
            raise ValueError("No prediction values in data")
        if len(data.target) == 0:
            raise ValueError("No target values in data")
        target_data = data.target
        batch = InferenceBatchData(
            prediction=data.prediction,
            prediction_norm=self._normalize(data.prediction),
            target=target_data,
            target_norm=self._normalize(target_data),
            time=data.time,
            i_time_start=self._n_timesteps_seen,
        )
        for aggregator in self._aggregators.values():
            aggregator.record_batch(batch)
        if self.n_ensemble_per_ic > 1:
            unfolded_target_data, unfolded_prediction_data = (
                data.as_ensemble_tensor_dicts(data.n_ensemble)
            )
            for ensemble_aggregator in self._ensemble_aggregators.values():
                ensemble_aggregator.record_batch(
                    target_data=unfolded_target_data,
                    gen_data=unfolded_prediction_data,
                    i_time_start=self._n_timesteps_seen,
                )
        n_times = data.time.shape[1]
        logs = self._get_inference_logs_slice(
            step_slice=slice(self._n_timesteps_seen, self._n_timesteps_seen + n_times),
        )
        self._n_timesteps_seen += n_times
        return logs

    def record_initial_condition(
        self,
        initial_condition: PairedData | PrognosticState,
    ) -> InferenceLogs:
        if self._n_timesteps_seen != 0:
            raise RuntimeError(
                "record_initial_condition may only be called once, "
                "before recording any batches"
            )
        if isinstance(initial_condition, PairedData):
            target_data = initial_condition.target
            gen_data = initial_condition.prediction
            time = initial_condition.time
        else:
            batch_data = initial_condition.as_batch_data()
            target_data = batch_data.data
            gen_data = target_data
            time = batch_data.time
        n_times = time.shape[1]
        if n_times != self.n_ic_steps:
            raise ValueError(
                f"Expected {self.n_ic_steps} initial condition steps, but got {n_times}"
            )
        batch = InferenceBatchData(
            prediction=gen_data,
            prediction_norm=self._normalize(gen_data),
            target=target_data,
            target_norm=self._normalize(target_data),
            time=time,
            i_time_start=0,
        )
        for name in self._time_series_aggregators:
            self._aggregators[name].record_batch(batch)
        logs = self._get_inference_logs_slice(
            step_slice=slice(self._n_timesteps_seen, self._n_timesteps_seen + n_times),
        )
        self._n_timesteps_seen = n_times
        return logs

    def get_summary_logs(self) -> InferenceLog:
        logs: InferenceLog = {}
        for name, aggregator in self._summary_aggregators.items():
            logging.info(f"Getting summary logs for {name} aggregator")
            logs.update(aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_logs(self):
        """Returns logs as can be reported to WandB."""
        logs: InferenceLog = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        if self.n_ensemble_per_ic > 1:
            for name, ensemble_aggregator in self._ensemble_aggregators.items():
                logs.update(ensemble_aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_inference_logs_slice(self, step_slice: slice):
        """
        Returns a subset of the time series for applicable metrics
        for a specific slice of as can be reported to WandB.

        Args:
            step_slice: Timestep slice to determine the time series subset.

        Returns:
            Tuple of start index and list of logs.
        """
        logs = {}
        for name, aggregator in self._time_series_aggregators.items():
            logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        return to_inference_logs(logs)

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        if self._save_diagnostics:
            reduced_diagnostics = get_reduced_diagnostics(
                sub_aggregators=self._aggregators,
                coords=self._coords,
            )
            if self._output_dir is not None:
                write_reduced_diagnostics(
                    reduced_diagnostics=reduced_diagnostics,
                    output_dir=self._output_dir,
                    subdir=subdir,
                )
            else:
                raise ValueError("Output directory not set.")


def to_inference_logs(
    log: Mapping[str, Table | float | int],
) -> list[dict[str, float | int]]:
    # We have a dictionary which contains WandB tables which we will convert
    # to a list of dictionaries, one for each row in the tables.
    # Any scalar values will be reported in the last dictionary.
    n_rows = 0
    for val in log.values():
        if isinstance(val, Table):
            n_rows = max(n_rows, len(val.data))
    logs: list[dict[str, float | int]] = []
    for i in range(max(1, n_rows)):
        logs.append({})
    for key, val in log.items():
        if isinstance(val, Table):
            for i, row in enumerate(val.data):
                for j, col in enumerate(val.columns):
                    key_without_table_name = key[: key.rfind("/")]
                    logs[i][f"{key_without_table_name}/{col}"] = row[j]
        else:
            logs[-1][key] = val
    return logs


def table_to_logs(table: Table) -> list[dict[str, float | int]]:
    """Converts a WandB table into a list of dictionaries."""
    logs = []
    for row in table.data:
        logs.append({table.columns[i]: row[i] for i in range(len(row))})
    return logs


@dataclasses.dataclass
class InferenceAggregatorConfig:
    """
    Configuration for inference aggregator.

    Parameters:
        time_mean_reference_data: Path to reference time means to compare against.
        log_global_mean_time_series: Whether to log global mean time series metrics.
    """

    time_mean_reference_data: str | None = None
    log_global_mean_time_series: bool = True

    def build(
        self,
        dataset_info: DatasetInfo,
        n_timesteps: int,
        output_dir: str | None = None,
        save_diagnostics: bool = True,
    ) -> "InferenceAggregator":
        if self.time_mean_reference_data is not None:
            time_means = xr.open_dataset(
                self.time_mean_reference_data,
                decode_timedelta=False,
            )
        else:
            time_means = None

        horizontal_coordinates = dataset_info.horizontal_coordinates
        gridded_operations = dataset_info.gridded_operations

        aggregators: dict[str, SubAggregator] = {}
        time_series_aggregators: dict[str, TimeSeriesLogs] = {}

        if self.log_global_mean_time_series:
            mean_agg = SingleTargetMeanAggregator(
                gridded_operations,
                n_timesteps=n_timesteps,
            )
            aggregators["mean"] = mean_agg
            time_series_aggregators["mean"] = mean_agg
        aggregators["time_mean"] = TimeMeanAggregator(
            gridded_operations=gridded_operations,
            variable_metadata=dataset_info.variable_metadata,
            reference_means=time_means,
        )
        aggregators["annual"] = GlobalMeanAnnualAggregator(
            gridded_operations,
            dataset_info.timestep,
            dataset_info.variable_metadata,
        )
        try:
            aggregators["power_spectrum"] = SphericalPowerSpectrumAggregator(
                gridded_operations=gridded_operations,
                nan_fill_fn=SmoothFloodFill(num_steps=4),
                report_plot=True,
                variable_metadata=dataset_info.variable_metadata,
            )
        except NotImplementedError:
            logging.warning(
                "Power spectrum aggregator not implemented for this grid type, "
                "omitting."
            )
        if (
            isinstance(horizontal_coordinates, LatLonCoordinates)
            and isinstance(gridded_operations, LatLonOperations)
            and n_timesteps * dataset_info.timestep > APPROXIMATELY_TWO_YEARS
        ):
            nino34_region = LatLonRegion(
                lat_bounds=NINO34_LAT,
                lon_bounds=NINO34_LON,
                lat=horizontal_coordinates.lat,
                lon=horizontal_coordinates.lon,
            )
            aggregators["enso_index"] = RegionalIndexAggregator(
                regional_weights=nino34_region.regional_weights,
                regional_mean=gridded_operations.regional_area_weighted_mean,
            )

        return InferenceAggregator(
            aggregators=aggregators,
            time_series_aggregators=time_series_aggregators,
            coords=horizontal_coordinates.coords,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
        )


class InferenceAggregator(
    InferenceAggregatorABC[
        PrognosticState,
        PairedData,
    ]
):
    """
    Aggregates statistics on a single timeseries of data.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        aggregators: dict[str, SubAggregator],
        time_series_aggregators: dict[str, TimeSeriesLogs],
        coords: Mapping[str, np.ndarray],
        save_diagnostics: bool = True,
        output_dir: str | None = None,
    ):
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._aggregators = aggregators
        self._time_series_aggregators = time_series_aggregators
        self._summary_aggregators = {
            name: agg
            for name, agg in aggregators.items()
            if name not in time_series_aggregators
        }
        self._coords = coords
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        self._log_time_series = len(time_series_aggregators) > 0
        self._n_timesteps_seen = 0

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(self, data: PairedData) -> InferenceLogs:
        """
        Record a batch of data.

        Args:
            data: Batch of data to record.
        """
        if len(data.prediction) == 0:
            raise ValueError("data is empty")
        batch = InferenceBatchData(
            prediction=data.prediction,
            time=data.time,
            i_time_start=self._n_timesteps_seen,
        )
        for aggregator in self._aggregators.values():
            aggregator.record_batch(batch)
        n_times = data.time.shape[1]
        logs = self._get_inference_logs_slice(
            step_slice=slice(self._n_timesteps_seen, self._n_timesteps_seen + n_times),
        )
        self._n_timesteps_seen += n_times
        return logs

    def record_initial_condition(
        self,
        initial_condition: PrognosticState,
    ) -> InferenceLogs:
        if self._n_timesteps_seen != 0:
            raise RuntimeError(
                "record_initial_condition may only be called once, "
                "before recording any batches"
            )
        batch_data = initial_condition.as_batch_data()
        batch = InferenceBatchData(
            prediction=batch_data.data,
            time=batch_data.time,
            i_time_start=0,
        )
        for name in self._time_series_aggregators:
            self._aggregators[name].record_batch(batch)
        n_times = batch_data.time.shape[1]
        logs = self._get_inference_logs_slice(
            step_slice=slice(self._n_timesteps_seen, self._n_timesteps_seen + n_times),
        )
        self._n_timesteps_seen = n_times
        return logs

    def get_summary_logs(self) -> InferenceLog:
        logs = {}
        for name, aggregator in self._summary_aggregators.items():
            logging.info(f"Getting summary logs for {name} aggregator")
            logs.update(aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_logs(self):
        """Returns logs as can be reported to WandB."""
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_inference_logs(self) -> list[dict[str, float | int]]:
        """
        Returns a list of logs to report to WandB.

        This is done because in inference, we use the wandb step
        as the time step, meaning we need to re-organize the logged data
        from tables into a list of dictionaries.
        """
        return to_inference_logs(self._get_logs())

    @torch.no_grad()
    def _get_inference_logs_slice(self, step_slice: slice):
        """
        Returns a subset of the time series for applicable metrics
        for a specific slice of as can be reported to WandB.

        Args:
            step_slice: Timestep slice to determine the time series subset.
        """
        logs = {}
        for name, aggregator in self._time_series_aggregators.items():
            logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        return to_inference_logs(logs)

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        if self._save_diagnostics:
            reduced_diagnostics = get_reduced_diagnostics(
                sub_aggregators=self._aggregators,
                coords=self._coords,
            )
            if self._output_dir is not None:
                write_reduced_diagnostics(
                    reduced_diagnostics=reduced_diagnostics,
                    output_dir=self._output_dir,
                    subdir=subdir,
                )
            else:
                raise ValueError("Output directory is not set.")
