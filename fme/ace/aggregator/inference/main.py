import dataclasses
import datetime
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import PairedData, PrognosticState
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.fill import SmoothFloodFill
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.gridded_ops import LatLonOperations
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

VALID_METRIC_TYPES = frozenset(
    {
        "mean",
        "step_mean",
        "power_spectrum",
        "zonal_mean",
        "video",
        "time_mean",
        "histogram",
        "seasonal",
        "annual",
        "enso_index",
        "enso_coefficient",
    }
)


class _VariableFilterAdapter:
    """Wraps a sub-aggregator to filter InferenceBatchData to specified variables."""

    def __init__(self, inner: Any, variables: Sequence[str]):
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
class StepMeanEntry:
    """
    Configuration for logging mean metrics at a particular step.

    Attributes:
        step: Number of forward steps after which to log mean metrics. For example,
            step=20 will log mean metrics at the 20th forward step
            (i.e. time index n_ic_steps + 19).
        name: Name to use for the logged metrics. If None, will use "mean_step_{step}".
    """

    step: int
    name: str | None = None

    def get_name(self):
        return self.name or f"mean_step_{self.step}"

    def validate(self, n_forward_steps: int):
        if self.step > n_forward_steps:
            raise ValueError(
                f"Step {self.step} is "
                f"greater than n_forward_steps {n_forward_steps}. "
                "Please ensure that all steps in log_step_means are less than or "
                "equal to "
                "n_forward_steps. If your run is less than 20 steps, you must pass "
                "a custom log_step_means configuration to override the default "
                "(e.g. log_step_means: [])."
            )


@dataclasses.dataclass
class MetricConfig:
    """Configuration for a single metric in the explicit metrics list.

    Attributes:
        type: Metric type. One of: "mean", "step_mean", "power_spectrum",
            "zonal_mean", "video", "time_mean", "histogram", "seasonal",
            "annual", "enso_index", "enso_coefficient".
        variables: Variables to include. None means all available variables.
        name: Custom metric name. Defaults vary by type (e.g. "mean",
            "power_spectrum", "mean_step_{step}").
        target: Whether to use denormalized ("denorm") or normalized ("norm")
            data. Only applies to "mean", "step_mean", and "time_mean" types.
        step: Forward step number. Required for "step_mean" type.
        enable_extended_videos: Whether to log extended statistical videos.
            Only applies to "video" type.
        reference_data: Path to reference data. For "time_mean" (reference
            means) and "annual" (monthly reference).
        channel_mean_names: Variable names for channel mean computation.
            Applies to norm-target "step_mean" and "time_mean" types.
        zonal_mean_max_size: Max time dimension for zonal mean images.
    """

    type: str
    variables: list[str] | None = None
    name: str | None = None
    target: Literal["denorm", "norm"] = "denorm"
    step: int | None = None
    enable_extended_videos: bool = False
    reference_data: str | None = None
    channel_mean_names: list[str] | None = None
    zonal_mean_max_size: int | bool = 4096

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        if self.type == "step_mean":
            if self.step is None:
                raise ValueError("step is required for step_mean metric")
            base = f"mean_step_{self.step}"
        elif self.type == "mean":
            base = "mean"
        elif self.type == "time_mean":
            base = "time_mean"
        else:
            return self.type
        if self.target == "norm":
            return f"{base}_norm"
        return base

    def __post_init__(self):
        if self.type not in VALID_METRIC_TYPES:
            raise ValueError(
                f"Unknown metric type: {self.type!r}. "
                f"Valid types: {sorted(VALID_METRIC_TYPES)}"
            )
        if self.type == "step_mean" and self.step is None:
            raise ValueError("step is required for step_mean metric type")


def _build_evaluator_metric(
    metric: MetricConfig,
    *,
    ops: Any,
    horizontal_coordinates: Any,
    n_timesteps: int,
    n_ic_steps: int,
    timestep: datetime.timedelta,
    variable_metadata: Any,
    channel_mean_names: Sequence[str] | None,
    monthly_reference_data: xr.Dataset | None,
    time_mean_reference_data: xr.Dataset | None,
    initial_time: xr.DataArray,
) -> Any:
    """Build a single evaluator sub-aggregator from a MetricConfig."""
    t = metric.type
    target = metric.target

    if t == "mean":
        return MeanAggregator(
            ops,
            target=target,
            n_timesteps=n_timesteps,
            variable_metadata=variable_metadata,
        )
    elif t == "step_mean":
        assert metric.step is not None
        target_time = metric.step + n_ic_steps - 1
        is_norm = target == "norm"
        return _OneStepMeanAdapter(
            OneStepMeanAggregator(
                ops,
                target_time=target_time,
                target=target,
                log_loss=False,
                include_bias=not is_norm,
                include_grad_mag_percent_diff=not is_norm,
                channel_mean_names=(
                    (metric.channel_mean_names or channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )
    elif t == "power_spectrum":
        return PairedSphericalPowerSpectrumAggregator(
            gridded_operations=ops,
            nan_fill_fn=SmoothFloodFill(num_steps=4),
            report_plot=True,
            variable_metadata=variable_metadata,
        )
    elif t == "zonal_mean":
        if ops.zonal_mean is None:
            raise ValueError(
                "Zonal mean metric requires a grid type that supports zonal means."
            )
        return ZonalMeanAggregator(
            zonal_mean=ops.zonal_mean,
            n_timesteps=n_timesteps,
            variable_metadata=variable_metadata,
            zonal_mean_max_size=metric.zonal_mean_max_size,
        )
    elif t == "video":
        if not isinstance(horizontal_coordinates, LatLonCoordinates):
            raise ValueError("Video metric requires LatLonCoordinates.")
        return VideoAggregator(
            n_timesteps=n_timesteps,
            enable_extended_videos=metric.enable_extended_videos,
            variable_metadata=variable_metadata,
        )
    elif t == "time_mean":
        is_norm = target == "norm"
        if metric.reference_data is not None:
            ref = xr.open_dataset(metric.reference_data, decode_timedelta=False)
        elif not is_norm:
            ref = time_mean_reference_data
        else:
            ref = None
        return TimeMeanEvaluatorAggregator(
            ops,
            horizontal_dims=horizontal_coordinates.dims,
            target=target,
            variable_metadata=variable_metadata,
            reference_means=ref,
            channel_mean_names=(
                (metric.channel_mean_names or channel_mean_names) if is_norm else None
            ),
        )
    elif t == "histogram":
        return HistogramAggregator()
    elif t == "seasonal":
        return SeasonalAggregator(
            ops=ops,
            variable_metadata=variable_metadata,
        )
    elif t == "annual":
        if metric.reference_data is not None:
            ref = xr.open_dataset(metric.reference_data, decode_timedelta=False)
        else:
            ref = monthly_reference_data
        return PairedGlobalMeanAnnualAggregator(
            ops=ops,
            timestep=timestep,
            variable_metadata=variable_metadata,
            monthly_reference_data=ref,
        )
    elif t == "enso_index":
        if not isinstance(horizontal_coordinates, LatLonCoordinates):
            raise ValueError("enso_index metric requires LatLonCoordinates.")
        if not isinstance(ops, LatLonOperations):
            raise ValueError("enso_index metric requires LatLonOperations.")
        nino34_region = LatLonRegion(
            lat_bounds=NINO34_LAT,
            lon_bounds=NINO34_LON,
            lat=horizontal_coordinates.lat,
            lon=horizontal_coordinates.lon,
        )
        return PairedRegionalIndexAggregator(
            target_aggregator=RegionalIndexAggregator(
                regional_weights=nino34_region.regional_weights,
                regional_mean=ops.regional_area_weighted_mean,
            ),
            prediction_aggregator=RegionalIndexAggregator(
                regional_weights=nino34_region.regional_weights,
                regional_mean=ops.regional_area_weighted_mean,
            ),
        )
    elif t == "enso_coefficient":
        return EnsoCoefficientEvaluatorAggregator(
            initial_time,
            n_timesteps - 1,
            timestep,
            gridded_operations=ops,
            variable_metadata=variable_metadata,
        )
    else:
        raise ValueError(f"Unknown metric type: {t!r}")


@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    """
    Configuration for inference evaluator aggregator.

    Parameters:
        log_histograms: Whether to log histograms of the targets and predictions.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension. If greater than 0 zonal-mean images will be logged. The
                value of log_zonal_mean_images is default to 4096 (2**12) and can be set
                with a maximum of 32768 (2**15) (limited by matplotlib).
        log_seasonal_means: Whether to log seasonal mean metrics and images.
        log_global_mean_time_series: Whether to log global mean time series metrics.
        log_global_mean_norm_time_series: Whether to log the normalized global mean
            time series metrics.
        monthly_reference_data: Path to monthly reference data to compare against.
        time_mean_reference_data: Path to reference time means to compare against.
        log_step_means: List of StepMeanEntry objects specifying steps at which
            to log mean metrics.
    """

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
    log_step_means: list[StepMeanEntry] = dataclasses.field(
        default_factory=lambda: [StepMeanEntry(step=20)]
    )
    metrics: list[MetricConfig] | None = None

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

        timestep = dataset_info.timestep
        horizontal_coordinates = dataset_info.horizontal_coordinates
        ops = dataset_info.gridded_operations
        n_timesteps = n_ic_steps + n_forward_steps

        aggregators: dict[str, SubAggregator] = {}
        time_series_aggregators: dict[str, TimeSeriesLogs] = {}
        ensemble_aggregators: dict[str, SelectStepEnsembleAggregator] = {}

        if self.metrics is not None:
            for metric in self.metrics:
                if (
                    metric.type == "step_mean"
                    and metric.step is not None
                    and metric.step > n_forward_steps
                ):
                    raise ValueError(
                        f"step_mean step {metric.step} exceeds "
                        f"n_forward_steps={n_forward_steps}"
                    )
                name = metric.get_name()
                agg = _build_evaluator_metric(
                    metric,
                    ops=ops,
                    horizontal_coordinates=horizontal_coordinates,
                    n_timesteps=n_timesteps,
                    n_ic_steps=n_ic_steps,
                    timestep=timestep,
                    variable_metadata=dataset_info.variable_metadata,
                    channel_mean_names=channel_mean_names,
                    monthly_reference_data=monthly_reference_data,
                    time_mean_reference_data=time_mean_reference_data,
                    initial_time=initial_time,
                )
                if metric.variables is not None:
                    agg = _VariableFilterAdapter(agg, metric.variables)
                aggregators[name] = agg
                if metric.type == "mean":
                    time_series_aggregators[name] = agg
        else:
            if self.log_global_mean_time_series:
                mean_agg = MeanAggregator(
                    ops,
                    target="denorm",
                    n_timesteps=n_timesteps,
                    variable_metadata=dataset_info.variable_metadata,
                )
                aggregators["mean"] = mean_agg
                time_series_aggregators["mean"] = mean_agg
            if self.log_global_mean_norm_time_series:
                mean_norm_agg = MeanAggregator(
                    ops,
                    target="norm",
                    n_timesteps=n_timesteps,
                    variable_metadata=dataset_info.variable_metadata,
                )
                aggregators["mean_norm"] = mean_norm_agg
                time_series_aggregators["mean_norm"] = mean_norm_agg
            for step_mean_entry in self.log_step_means:
                step_mean_entry.validate(n_forward_steps)
                step = step_mean_entry.step
                name = step_mean_entry.get_name()
                target_time = step + n_ic_steps - 1
                aggregators[name] = _OneStepMeanAdapter(
                    OneStepMeanAggregator(
                        ops,
                        target_time=target_time,
                        target="denorm",
                        log_loss=False,
                    )
                )
                aggregators[name + "_norm"] = _OneStepMeanAdapter(
                    OneStepMeanAggregator(
                        ops,
                        target_time=target_time,
                        target="norm",
                        log_loss=False,
                        include_bias=False,
                        include_grad_mag_percent_diff=False,
                        channel_mean_names=channel_mean_names,
                    )
                )
                if n_ensemble_per_ic > 1:
                    ensemble_aggregators["ensemble_step_20"] = (
                        get_one_step_ensemble_aggregator(
                            gridded_operations=ops,
                            target_time=20,
                            log_mean_maps=False,
                            metadata=dataset_info.variable_metadata,
                        )
                    )
            try:
                flood_fill = SmoothFloodFill(num_steps=4)
                aggregators["power_spectrum"] = PairedSphericalPowerSpectrumAggregator(
                    gridded_operations=ops,
                    nan_fill_fn=flood_fill,
                    report_plot=True,
                    variable_metadata=dataset_info.variable_metadata,
                )
            except NotImplementedError:
                logging.warning(
                    "Power spectrum aggregator not implemented for this grid "
                    "type, omitting."
                )
            if self.log_zonal_mean_images:
                if ops.zonal_mean is None:
                    logging.warning(
                        "Zonal mean aggregator not implemented for this grid "
                        "type, omitting."
                    )
                else:
                    aggregators["zonal_mean"] = ZonalMeanAggregator(
                        zonal_mean=ops.zonal_mean,
                        n_timesteps=n_timesteps,
                        variable_metadata=dataset_info.variable_metadata,
                        zonal_mean_max_size=self.log_zonal_mean_images,
                    )
            if isinstance(horizontal_coordinates, LatLonCoordinates):
                if self.log_video:
                    aggregators["video"] = VideoAggregator(
                        n_timesteps=n_timesteps,
                        enable_extended_videos=self.log_extended_video,
                        variable_metadata=dataset_info.variable_metadata,
                    )
            aggregators["time_mean"] = TimeMeanEvaluatorAggregator(
                ops,
                horizontal_dims=horizontal_coordinates.dims,
                variable_metadata=dataset_info.variable_metadata,
                reference_means=time_mean_reference_data,
            )
            aggregators["time_mean_norm"] = TimeMeanEvaluatorAggregator(
                ops,
                horizontal_dims=horizontal_coordinates.dims,
                target="norm",
                variable_metadata=dataset_info.variable_metadata,
                channel_mean_names=channel_mean_names,
            )
            if self.log_histograms:
                aggregators["histogram"] = HistogramAggregator()
            if self.log_seasonal_means:
                aggregators["seasonal"] = SeasonalAggregator(
                    ops=ops,
                    variable_metadata=dataset_info.variable_metadata,
                )
            if n_timesteps * timestep > APPROXIMATELY_TWO_YEARS:
                aggregators["annual"] = PairedGlobalMeanAnnualAggregator(
                    ops=ops,
                    timestep=timestep,
                    variable_metadata=dataset_info.variable_metadata,
                    monthly_reference_data=monthly_reference_data,
                )
                if (
                    isinstance(horizontal_coordinates, LatLonCoordinates)
                    and isinstance(ops, LatLonOperations)
                    and self.log_nino34_index
                ):
                    nino34_region = LatLonRegion(
                        lat_bounds=NINO34_LAT,
                        lon_bounds=NINO34_LON,
                        lat=horizontal_coordinates.lat,
                        lon=horizontal_coordinates.lon,
                    )
                    aggregators["enso_index"] = PairedRegionalIndexAggregator(
                        target_aggregator=RegionalIndexAggregator(
                            regional_weights=nino34_region.regional_weights,
                            regional_mean=ops.regional_area_weighted_mean,
                        ),
                        prediction_aggregator=RegionalIndexAggregator(
                            regional_weights=nino34_region.regional_weights,
                            regional_mean=ops.regional_area_weighted_mean,
                        ),
                    )
            if n_timesteps * timestep > SLIGHTLY_LESS_THAN_FIVE_YEARS:
                aggregators["enso_coefficient"] = EnsoCoefficientEvaluatorAggregator(
                    initial_time,
                    n_timesteps - 1,
                    timestep,
                    gridded_operations=ops,
                    variable_metadata=dataset_info.variable_metadata,
                )

        return InferenceEvaluatorAggregator(
            aggregators=aggregators,
            time_series_aggregators=time_series_aggregators,
            coords=horizontal_coordinates.coords,
            n_ic_steps=n_ic_steps,
            normalize=normalize,
            save_diagnostics=save_diagnostics,
            output_dir=output_dir,
            n_ensemble_per_ic=n_ensemble_per_ic,
            ensemble_aggregators=ensemble_aggregators,
        )


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
