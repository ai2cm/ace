import dataclasses
import datetime
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import torch
import xarray as xr

from fme.ace.data_loading.batch_data import PairedData, PrognosticState
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.diagnostics import get_reduced_diagnostics, write_reduced_diagnostics
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Table, WandB
from fme.ace.utils import comm

from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator
from .annual import GlobalMeanAnnualAggregator, PairedGlobalMeanAnnualAggregator
from .enso import (
    EnsoCoefficientEvaluatorAggregator,
    LatLonRegion,
    PairedRegionalIndexAggregator,
    RegionalIndexAggregator,
)
from .histogram import HistogramAggregator
from .reduced import MeanAggregator, SingleTargetMeanAggregator
from .seasonal import SeasonalAggregator
from .spectrum import PairedSphericalPowerSpectrumAggregator
from .time_mean import TimeMeanAggregator, TimeMeanEvaluatorAggregator
from .video import VideoAggregator
from .zonal_mean import ZonalMeanAggregator
from physicsnemo.distributed.utils import compute_split_shapes

wandb = WandB.get_instance()
APPROXIMATELY_TWO_YEARS = datetime.timedelta(days=730)
SLIGHTLY_LESS_THAN_FIVE_YEARS = datetime.timedelta(days=1800)
NINO34_LAT = (-5, 5)
NINO34_LON = (190, 240)


class _Aggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        data: TensorMapping,
    ): ...

    @torch.no_grad()
    def get_logs(self, label: str): ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset: ...


class _EvaluatorAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ): ...

    @torch.no_grad()
    def get_logs(self, label: str): ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset: ...


class _TimeDependentAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        data: TensorMapping,
    ): ...

    @torch.no_grad()
    def get_logs(self, label: str): ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset: ...


class _TimeDependentEvaluatorAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ): ...

    @torch.no_grad()
    def get_logs(self, label: str): ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset: ...


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

    def build(
        self,
        dataset_info: DatasetInfo,
        n_timesteps: int,
        initial_time: xr.DataArray,
        normalize: Callable[[TensorMapping], TensorDict],
        output_dir: str | None = None,
        record_step_20: bool = False,
        channel_mean_names: Sequence[str] | None = None,
        save_diagnostics: bool = True,
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
            time_mean = None
        else:
            time_mean = xr.open_dataset(
                self.time_mean_reference_data, decode_timedelta=False
            )
        distributed = comm.is_distributed("spatial")
        if distributed:
          lat_length = len(monthly_reference_data.coords['lat'])
          lon_length = len(monthly_reference_data.coords['lon'])
          crop_shape = (lat_length, lon_length)
          crop_offset=(0, 0)

          if comm.get_size("h") > 1:
            shapes_h = compute_split_shapes(crop_shape[0], comm.get_size("h"))
            local_shape_h = shapes_h[comm.get_rank("h")]
            local_offset_h = crop_offset[0] + sum(shapes_h[: comm.get_rank("h")])
          else:
            local_shape_h = crop_shape[0]
            local_offset_h = crop_offset[0]

          if self.distributed and (comm.get_size("w") > 1):
            shapes_w = compute_split_shapes(crop_shape[1], comm.get_size("w"))
            local_shape_w = shapes_w[comm.get_rank("w")]
            local_offset_w = crop_offset[1] + sum(shapes_w[: comm.get_rank("w")])
          else:
            local_shape_w = crop_shape[1]
            local_offset_w = crop_offset[1]
          #CHECK that the array is split correctly.
          monthly_reference_data = monthly_reference_data.sel(lat=slice(local_offset_h, local_offset_h + local_shape_h-1), lon=slice(local_offset_w, local_offset_w + local_shape_w-1))
        return InferenceEvaluatorAggregator(
            dataset_info=dataset_info,
            n_timesteps=n_timesteps,
            initial_time=initial_time,
            output_dir=output_dir,
            log_histograms=self.log_histograms,
            log_video=self.log_video,
            enable_extended_videos=self.log_extended_video,
            log_zonal_mean_images=self.log_zonal_mean_images,
            log_seasonal_means=self.log_seasonal_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
            log_global_mean_norm_time_series=self.log_global_mean_norm_time_series,
            monthly_reference_data=monthly_reference_data,
            time_mean_reference_data=time_mean,
            record_step_20=record_step_20,
            channel_mean_names=channel_mean_names,
            normalize=normalize,
            save_diagnostics=save_diagnostics,
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
        dataset_info: DatasetInfo,
        n_timesteps: int,
        initial_time: xr.DataArray,
        normalize: Callable[[TensorMapping], TensorDict],
        log_zonal_mean_images: bool | int,
        output_dir: str | None = None,
        record_step_20: bool = False,
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
    ):
        """
        Args:
            dataset_info: Dataset coordinates and metadata.
            n_timesteps: Number of timesteps of inference that will be run.
            initial_time: Initial time for each sample.
            output_dir: Directory to save diagnostic output.
            normalize: Normalization function to use.
            log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension.
            record_step_20: Whether to record the mean of the 20th steps.
            log_video: Whether to log videos of the state evolution.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            log_seasonal_means: Whether to log seasonal means metrics and images.
            log_global_mean_time_series: Whether to log global mean time series metrics.
            log_global_mean_norm_time_series: Whether to log the normalized global mean
                time series metrics.
            monthly_reference_data: Reference monthly data for computing target stats.
            log_histograms: Whether to aggregate histograms.
            data_grid: The grid type of the data, used for spherical power spectrum.
            time_mean_reference_data: Reference time means for computing bias stats.
            channel_mean_names: Names over which to compute channel means. If not
                provided, all available variables will be used.
            log_nino34_index: Whether to log the Nino34 index.
            save_diagnostics: Whether to save reduced diagnostics to disk.
        """
        if save_diagnostics and output_dir is None:
            raise ValueError("Output directory must be set to save diagnostics")
        self._channel_mean_names = channel_mean_names
        self._aggregators: dict[str, _EvaluatorAggregator] = {}
        self._time_dependent_aggregators: dict[
            str, _TimeDependentEvaluatorAggregator
        ] = {}
        self._save_diagnostics = save_diagnostics
        self._output_dir = output_dir
        timestep = dataset_info.timestep
        horizontal_coordinates = dataset_info.horizontal_coordinates
        self._coords = horizontal_coordinates.coords
        ops = dataset_info.gridded_operations
        self._log_time_series = (
            log_global_mean_time_series or log_global_mean_norm_time_series
        )
        if log_global_mean_time_series:
            self._aggregators["mean"] = MeanAggregator(
                ops,
                target="denorm",
                n_timesteps=n_timesteps,
                variable_metadata=dataset_info.variable_metadata,
            )
        if log_global_mean_norm_time_series:
            self._aggregators["mean_norm"] = MeanAggregator(
                ops,
                target="norm",
                n_timesteps=n_timesteps,
                variable_metadata=dataset_info.variable_metadata,
            )
        self._record_step_20 = record_step_20
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(
                ops,
                target_time=20,
                target="denorm",
            )
            self._aggregators["mean_step_20_norm"] = OneStepMeanAggregator(
                ops,
                target_time=20,
                target="norm",
                channel_mean_names=self._channel_mean_names,
            )
        try:
            self._aggregators["power_spectrum"] = (
                PairedSphericalPowerSpectrumAggregator(
                    gridded_operations=ops,
                    report_plot=True,
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
            self._time_dependent_aggregators["seasonal"] = SeasonalAggregator(
                ops=ops,
                variable_metadata=dataset_info.variable_metadata,
            )
        if n_timesteps * timestep > APPROXIMATELY_TWO_YEARS:
            self._time_dependent_aggregators["annual"] = (
                PairedGlobalMeanAnnualAggregator(
                    ops=ops,
                    timestep=timestep,
                    variable_metadata=dataset_info.variable_metadata,
                    monthly_reference_data=monthly_reference_data,
                )
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
                self._time_dependent_aggregators["enso_index"] = (
                    PairedRegionalIndexAggregator(
                        target_aggregator=RegionalIndexAggregator(
                            regional_weights=nino34_region.regional_weights,
                            regional_mean=ops.regional_area_weighted_mean,
                        ),
                        prediction_aggregator=RegionalIndexAggregator(
                            regional_weights=nino34_region.regional_weights,
                            regional_mean=ops.regional_area_weighted_mean,
                        ),
                    )
                )
        if n_timesteps * timestep > SLIGHTLY_LESS_THAN_FIVE_YEARS:
            self._time_dependent_aggregators["enso_coefficient"] = (
                EnsoCoefficientEvaluatorAggregator(
                    initial_time,
                    n_timesteps - 1,
                    timestep,
                    gridded_operations=ops,
                    variable_metadata=dataset_info.variable_metadata,
                )
            )
        self._summary_aggregators = {
            name: agg
            for name, agg in list(self._aggregators.items())
            + list(self._time_dependent_aggregators.items())
            if name not in ["mean", "mean_norm"]
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
        target_data_norm = self._normalize(target_data)
        gen_data_norm = self._normalize(data.prediction)
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                target_data=target_data,
                gen_data=data.prediction,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                i_time_start=self._n_timesteps_seen,
            )
        for time_dependent_aggregator in self._time_dependent_aggregators.values():
            time_dependent_aggregator.record_batch(
                time=data.time,
                target_data=target_data,
                gen_data=data.prediction,
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
            target_data_norm = self._normalize(target_data)
            gen_data = initial_condition.prediction
            gen_data_norm = self._normalize(gen_data)
            n_times = initial_condition.time.shape[1]
        else:
            batch_data = initial_condition.as_batch_data()
            target_data = batch_data.data
            target_data_norm = self._normalize(target_data)
            gen_data = target_data
            gen_data_norm = target_data_norm
            n_times = batch_data.time.shape[1]
        for aggregator_name in ["mean", "mean_norm"]:
            aggregator = self._aggregators.get(aggregator_name)
            if aggregator is not None:
                aggregator.record_batch(
                    target_data=target_data,
                    gen_data=gen_data,
                    target_data_norm=target_data_norm,
                    gen_data_norm=gen_data_norm,
                    i_time_start=0,
                )
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
        if self._record_step_20:
            # we don't provide it so these are NaN always
            logs.pop("mean_step_20/loss")
            logs.pop("mean_step_20_norm/loss")
        return logs

    @torch.no_grad()
    def _get_logs(self):
        """
        Returns logs as can be reported to WandB.
        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        for name, time_dependent_aggregator in self._time_dependent_aggregators.items():
            logs.update(time_dependent_aggregator.get_logs(label=name))
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
        for name, aggregator in self._aggregators.items():
            if isinstance(aggregator, MeanAggregator):
                logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        return to_inference_logs(logs)

    @torch.no_grad()
    def flush_diagnostics(self, subdir: str | None = None):
        if self._save_diagnostics:
            reduced_diagnostics = get_reduced_diagnostics(
                sub_aggregators=(self._aggregators | self._time_dependent_aggregators),
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
    # we have a dictionary which contains WandB tables
    # which we will convert to a list of dictionaries, one for each
    # row in the tables. Any scalar values will be reported in the last
    # dictionary.
    n_rows = 0
    for val in log.values():
        if isinstance(val, Table):
            n_rows = max(n_rows, len(val.data))
    logs: list[dict[str, float | int]] = []
    for i in range(max(1, n_rows)):  # need at least one for non-series values
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
    """
    Converts a WandB table into a list of dictionaries.
    """
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
        output_dir: str,
    ) -> "InferenceAggregator":
        if self.time_mean_reference_data is not None:
            time_means = xr.open_dataset(
                self.time_mean_reference_data,
                decode_timedelta=False,
            )
        else:
            time_means = None
        return InferenceAggregator(
            dataset_info=dataset_info,
            n_timesteps=n_timesteps,
            output_dir=output_dir,
            time_mean_reference_data=time_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
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
        aggregators: dict[str, _Aggregator] = {}
        gridded_operations = dataset_info.gridded_operations
        if log_global_mean_time_series:
            aggregators["mean"] = SingleTargetMeanAggregator(
                gridded_operations,
                n_timesteps=n_timesteps,
            )
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
            for name in ["time_mean", "annual", "enso_index"]
            if name in aggregators
        }
        self._time_dependent_aggregator_names = ["annual", "enso_index"]
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
        for name in self._aggregators:
            if name in self._time_dependent_aggregator_names:
                self._aggregators[name].record_batch(
                    time=data.time, data=data.prediction
                )
            else:
                self._aggregators[name].record_batch(
                    data=data.prediction,
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
        initial_condition: PrognosticState,
    ) -> InferenceLogs:
        if self._n_timesteps_seen != 0:
            raise RuntimeError(
                "record_initial_condition may only be called once, "
                "before recording any batches"
            )
        batch_data = initial_condition.as_batch_data()
        if "mean" in self._aggregators:
            self._aggregators["mean"].record_batch(
                data=batch_data.data,
                i_time_start=0,
            )
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
        """
        Returns logs as can be reported to WandB.
        """
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
        for name, aggregator in self._aggregators.items():
            if isinstance(aggregator, SingleTargetMeanAggregator):
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
