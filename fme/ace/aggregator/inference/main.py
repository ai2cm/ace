import datetime
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

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

from ..one_step.ensemble import get_one_step_ensemble_aggregator
from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator
from .annual import GlobalMeanAnnualAggregator, PairedGlobalMeanAnnualAggregator
from .config import StepMeanEntry
from .data import InferenceBatchData
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
        dataset_info: DatasetInfo,
        n_ic_steps: int,
        n_forward_steps: int,
        initial_time: xr.DataArray,
        normalize: Callable[[TensorMapping], TensorDict],
        log_zonal_mean_images: bool | int,
        log_step_means: list[StepMeanEntry],
        output_dir: str | None = None,
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_seasonal_means: bool = False,
        log_global_mean_time_series: bool = True,
        log_global_mean_norm_time_series: bool = True,
        monthly_reference_data: xr.Dataset | None = None,
        log_histograms: bool = False,
        time_mean_reference_data: xr.Dataset | None = None,
        channel_mean_names: Sequence[str] | None = None,
        log_nino34_index: bool = True,
        save_diagnostics: bool = True,
        n_ensemble_per_ic: int = 1,
    ):
        """
        Args:
            dataset_info: Dataset coordinates and metadata.
            n_ic_steps: Number of initial condition steps in the data.
            n_forward_steps: Number of forward steps in the data.
            initial_time: Initial time for each sample.
            output_dir: Directory to save diagnostic output.
            normalize: Normalization function to use.
            log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension.
            log_step_means: List of StepMeanEntry objects specifying steps at which to
                log mean metrics.
            log_video: Whether to log videos of the state evolution.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            log_seasonal_means: Whether to log seasonal means metrics and images.
            log_global_mean_time_series: Whether to log global mean time series metrics.
            log_global_mean_norm_time_series: Whether to log the normalized global mean
                time series metrics.
            monthly_reference_data: Reference monthly data for computing target stats.
            log_histograms: Whether to aggregate histograms.
            time_mean_reference_data: Reference time means for computing bias stats.
            channel_mean_names: Names over which to compute channel means. If not
                provided, all available variables will be used.
            log_nino34_index: Whether to log the Nino34 index.
            save_diagnostics: Whether to save reduced diagnostics to disk.
            n_ensemble_per_ic: Number of ensemble members per initial condition.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._channel_mean_names = channel_mean_names
        self._aggregators: dict[str, Any] = {}
        self.n_ensemble_per_ic = n_ensemble_per_ic
        self._ensemble_aggregators: dict[str, Any] = {}
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        timestep = dataset_info.timestep
        horizontal_coordinates = dataset_info.horizontal_coordinates
        self._coords = horizontal_coordinates.coords
        ops = dataset_info.gridded_operations
        self._log_time_series = (
            log_global_mean_time_series or log_global_mean_norm_time_series
        )
        self.n_ic_steps = n_ic_steps
        n_timesteps = n_ic_steps + n_forward_steps
        self._streaming_names: list[str] = []
        if log_global_mean_time_series:
            self._aggregators["mean"] = MeanAggregator(
                ops,
                target="denorm",
                n_timesteps=n_timesteps,
                variable_metadata=dataset_info.variable_metadata,
            )
            self._streaming_names.append("mean")
        if log_global_mean_norm_time_series:
            self._aggregators["mean_norm"] = MeanAggregator(
                ops,
                target="norm",
                n_timesteps=n_timesteps,
                variable_metadata=dataset_info.variable_metadata,
            )
            self._streaming_names.append("mean_norm")
        for step_mean_entry in log_step_means:
            step_mean_entry.validate(n_forward_steps)
            step = step_mean_entry.step
            name = step_mean_entry.get_name()
            target_time = step + n_ic_steps - 1
            self._aggregators[name] = _OneStepMeanAdapter(
                OneStepMeanAggregator(
                    ops,
                    target_time=target_time,
                    target="denorm",
                    log_loss=False,
                )
            )
            self._aggregators[name + "_norm"] = _OneStepMeanAdapter(
                OneStepMeanAggregator(
                    ops,
                    target_time=target_time,
                    target="norm",
                    log_loss=False,
                    include_bias=False,
                    include_grad_mag_percent_diff=False,
                    channel_mean_names=self._channel_mean_names,
                )
            )
            if n_ensemble_per_ic > 1:
                self._ensemble_aggregators["ensemble_step_20"] = (
                    get_one_step_ensemble_aggregator(
                        gridded_operations=ops,
                        target_time=20,
                        log_mean_maps=False,
                        metadata=dataset_info.variable_metadata,
                    )
                )
        try:
            flood_fill = SmoothFloodFill(num_steps=4)
            self._aggregators["power_spectrum"] = (
                PairedSphericalPowerSpectrumAggregator(
                    gridded_operations=ops,
                    nan_fill_fn=flood_fill,
                    report_plot=True,
                    variable_metadata=dataset_info.variable_metadata,
                )
            )
        except NotImplementedError:
            logging.warning(
                "Power spectrum aggregator not implemented for this grid type, "
                "omitting."
            )
        if log_zonal_mean_images:
            if ops.zonal_mean is None:
                logging.warning(
                    "Zonal mean aggregator not implemented for this grid type, "
                    "omitting."
                )
            else:
                self._aggregators["zonal_mean"] = ZonalMeanAggregator(
                    zonal_mean=ops.zonal_mean,
                    n_timesteps=n_timesteps,
                    variable_metadata=dataset_info.variable_metadata,
                    zonal_mean_max_size=log_zonal_mean_images,
                )
        if isinstance(horizontal_coordinates, LatLonCoordinates):
            if log_video:
                self._aggregators["video"] = VideoAggregator(
                    n_timesteps=n_timesteps,
                    enable_extended_videos=enable_extended_videos,
                    variable_metadata=dataset_info.variable_metadata,
                )
        self._aggregators["time_mean"] = TimeMeanEvaluatorAggregator(
            ops,
            horizontal_dims=horizontal_coordinates.dims,
            variable_metadata=dataset_info.variable_metadata,
            reference_means=time_mean_reference_data,
        )
        self._aggregators["time_mean_norm"] = TimeMeanEvaluatorAggregator(
            ops,
            horizontal_dims=horizontal_coordinates.dims,
            target="norm",
            variable_metadata=dataset_info.variable_metadata,
            channel_mean_names=self._channel_mean_names,
        )
        if log_histograms:
            self._aggregators["histogram"] = HistogramAggregator()
        if log_seasonal_means:
            self._aggregators["seasonal"] = SeasonalAggregator(
                ops=ops,
                variable_metadata=dataset_info.variable_metadata,
            )
        if n_timesteps * timestep > APPROXIMATELY_TWO_YEARS:
            self._aggregators["annual"] = PairedGlobalMeanAnnualAggregator(
                ops=ops,
                timestep=timestep,
                variable_metadata=dataset_info.variable_metadata,
                monthly_reference_data=monthly_reference_data,
            )
            if (
                isinstance(horizontal_coordinates, LatLonCoordinates)
                and isinstance(ops, LatLonOperations)
                and log_nino34_index
            ):
                nino34_region = LatLonRegion(
                    lat_bounds=NINO34_LAT,
                    lon_bounds=NINO34_LON,
                    lat=horizontal_coordinates.lat,
                    lon=horizontal_coordinates.lon,
                )
                self._aggregators["enso_index"] = PairedRegionalIndexAggregator(
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
            self._aggregators["enso_coefficient"] = EnsoCoefficientEvaluatorAggregator(
                initial_time,
                n_timesteps - 1,
                timestep,
                gridded_operations=ops,
                variable_metadata=dataset_info.variable_metadata,
            )

        summary_aggregators_list = list(self._aggregators.items())
        if self.n_ensemble_per_ic > 1:
            summary_aggregators_list.extend(self._ensemble_aggregators.items())

        self._summary_aggregators = {
            name: agg
            for name, agg in summary_aggregators_list
            if name not in self._streaming_names
        }
        self._n_timesteps_seen = 0
        self._normalize = normalize

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
            n_times = initial_condition.time.shape[1]
        else:
            batch_data = initial_condition.as_batch_data()
            target_data = batch_data.data
            gen_data = target_data
            n_times = batch_data.time.shape[1]
        if n_times != self.n_ic_steps:
            raise ValueError(
                f"Expected {self.n_ic_steps} initial condition steps, but got {n_times}"
            )
        batch = InferenceBatchData(
            prediction=gen_data,
            prediction_norm=self._normalize(gen_data),
            target=target_data,
            target_norm=self._normalize(target_data),
            time=None,
            i_time_start=0,
        )
        for name in self._streaming_names:
            aggregator = self._aggregators.get(name)
            if aggregator is not None:
                aggregator.record_batch(batch)
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
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        if self.n_ensemble_per_ic > 1:
            for name, ensemble_aggregator in self._ensemble_aggregators.items():
                logs.update(ensemble_aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_inference_logs_slice(self, step_slice: slice):
        logs = {}
        for name in self._streaming_names:
            aggregator = self._aggregators.get(name)
            if aggregator is not None:
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
    logs = []
    for row in table.data:
        logs.append({table.columns[i]: row[i] for i in range(len(row))})
    return logs


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
        dataset_info: DatasetInfo,
        n_timesteps: int,
        save_diagnostics: bool = True,
        output_dir: str | None = None,
        time_mean_reference_data: xr.Dataset | None = None,
        log_global_mean_time_series: bool = True,
    ):
        """
        Args:
            dataset_info: The coordinates of the dataset.
            n_timesteps: Number of timesteps in the model.
            save_diagnostics: Whether to save diagnostics.
            output_dir: Directory to save diagnostic output.
            time_mean_reference_data: Reference time means for computing bias stats.
            log_global_mean_time_series: Whether to log global mean time series metrics.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._log_time_series = log_global_mean_time_series
        horizontal_coordinates = dataset_info.horizontal_coordinates
        self._coords = horizontal_coordinates.coords
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        aggregators: dict[str, Any] = {}
        gridded_operations = dataset_info.gridded_operations
        self._streaming_names: list[str] = []
        if log_global_mean_time_series:
            aggregators["mean"] = SingleTargetMeanAggregator(
                gridded_operations,
                n_timesteps=n_timesteps,
            )
            self._streaming_names.append("mean")
        aggregators["time_mean"] = TimeMeanAggregator(
            gridded_operations=gridded_operations,
            variable_metadata=dataset_info.variable_metadata,
            reference_means=time_mean_reference_data,
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
        self._aggregators = aggregators
        self._summary_aggregators = {
            name: aggregators[name]
            for name in ["time_mean", "annual", "enso_index", "power_spectrum"]
            if name in aggregators
        }
        self._n_timesteps_seen = 0

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(self, data: PairedData) -> InferenceLogs:
        if len(data.prediction) == 0:
            raise ValueError("data is empty")
        batch = InferenceBatchData(
            prediction=data.prediction,
            prediction_norm=data.prediction,
            target=None,
            target_norm=None,
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
        if "mean" in self._aggregators:
            batch = InferenceBatchData(
                prediction=batch_data.data,
                prediction_norm=batch_data.data,
                target=None,
                target_norm=None,
                time=None,
                i_time_start=0,
            )
            self._aggregators["mean"].record_batch(batch)
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
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        return logs

    @torch.no_grad()
    def _get_inference_logs(self) -> list[dict[str, float | int]]:
        return to_inference_logs(self._get_logs())

    @torch.no_grad()
    def _get_inference_logs_slice(self, step_slice: slice):
        logs = {}
        for name in self._streaming_names:
            aggregator = self._aggregators.get(name)
            if aggregator is not None:
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
