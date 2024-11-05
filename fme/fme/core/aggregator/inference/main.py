import dataclasses
import datetime
import warnings
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
)

import torch
import xarray as xr

from fme.core.data_loading.batch_data import (
    BatchData,
    PairedData,
)
from fme.core.data_loading.data_typing import (
    HorizontalCoordinates,
    LatLonCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
)
from fme.core.generics.state import PrognosticStateABC
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Table, WandB

from ...gridded_ops import GriddedOperations
from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator
from .annual import GlobalMeanAnnualAggregator
from .enso import EnsoCoefficientEvaluatorAggregator
from .histogram import HistogramAggregator
from .reduced import MeanAggregator, SingleTargetMeanAggregator
from .seasonal import SeasonalAggregator
from .spectrum import PairedSphericalPowerSpectrumAggregator
from .time_mean import TimeMeanAggregator, TimeMeanEvaluatorAggregator
from .video import VideoAggregator
from .zonal_mean import ZonalMeanAggregator

wandb = WandB.get_instance()
APPROXIMATELY_TWO_YEARS = datetime.timedelta(days=730)
SLIGHTLY_LESS_THAN_FIVE_YEARS = datetime.timedelta(days=1800)


class _Aggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        data: TensorMapping,
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
        ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        ...


class _EvaluatorAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
        ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        ...


class _TimeDependentAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        data: TensorMapping,
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
        ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        ...


class _TimeDependentEvaluatorAggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
        ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        ...


@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    """
    Configuration for inference evaluator aggregator.

    Attributes:
        log_histograms: Whether to log histograms of the targets and predictions.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
            time dimension.
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
    log_zonal_mean_images: bool = True
    log_seasonal_means: bool = False
    log_global_mean_time_series: bool = True
    log_global_mean_norm_time_series: bool = True
    monthly_reference_data: Optional[str] = None
    time_mean_reference_data: Optional[str] = None

    def build(
        self,
        sigma_coordinates: SigmaCoordinates,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: datetime.timedelta,
        n_timesteps: int,
        initial_times: xr.DataArray,
        record_step_20: bool = False,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
        channel_mean_names: Optional[Sequence[str]] = None,
    ) -> "InferenceEvaluatorAggregator":
        if self.monthly_reference_data is None:
            monthly_reference_data = None
        else:
            monthly_reference_data = xr.open_dataset(self.monthly_reference_data)
        if self.time_mean_reference_data is None:
            time_mean = None
        else:
            time_mean = xr.open_dataset(self.time_mean_reference_data)

        if n_timesteps > 2**15 and self.log_zonal_mean_images:
            # matplotlib raises an error if image size is too large, and we plot
            # one pixel per timestep in the zonal mean images.
            warnings.warn(
                "Disabling zonal mean images logging due to large number of timesteps"
                f" (n_timesteps={n_timesteps}). Set log_zonal_mean_images=False or "
                "decrease n_timesteps to below 2**15 to avoid this warning."
            )
            log_zonal_mean_images = False
        else:
            log_zonal_mean_images = self.log_zonal_mean_images

        return InferenceEvaluatorAggregator(
            sigma_coordinates=sigma_coordinates,
            horizontal_coordinates=horizontal_coordinates,
            timestep=timestep,
            n_timesteps=n_timesteps,
            initial_times=initial_times,
            log_histograms=self.log_histograms,
            log_video=self.log_video,
            enable_extended_videos=self.log_extended_video,
            log_zonal_mean_images=log_zonal_mean_images,
            log_seasonal_means=self.log_seasonal_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
            log_global_mean_norm_time_series=self.log_global_mean_norm_time_series,
            monthly_reference_data=monthly_reference_data,
            time_mean_reference_data=time_mean,
            record_step_20=record_step_20,
            metadata=metadata,
            channel_mean_names=channel_mean_names,
        )


class InferenceEvaluatorAggregator(
    InferenceAggregatorABC[PrognosticStateABC[BatchData], PairedData]
):
    """
    Aggregates statistics for inference comparing a generated and target series.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        sigma_coordinates: SigmaCoordinates,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: datetime.timedelta,
        n_timesteps: int,
        initial_times: xr.DataArray,
        record_step_20: bool = False,
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_zonal_mean_images: bool = False,
        log_seasonal_means: bool = False,
        log_global_mean_time_series: bool = True,
        log_global_mean_norm_time_series: bool = True,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
        monthly_reference_data: Optional[xr.Dataset] = None,
        log_histograms: bool = False,
        time_mean_reference_data: Optional[xr.Dataset] = None,
        channel_mean_names: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            sigma_coordinates: Data sigma coordinates
            horizontal_coordinates: Data horizontal coordinates
            timestep: Timestep of the model.
            n_timesteps: Number of timesteps of inference that will be run.
            initial_times: Initial times for each sample.
            record_step_20: Whether to record the mean of the 20th steps.
            log_video: Whether to log videos of the state evolution.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension.
            log_seasonal_means: Whether to log seasonal means metrics and images.
            log_global_mean_time_series: Whether to log global mean time series metrics.
            log_global_mean_norm_time_series: Whether to log the normalized global mean
                time series metrics.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            monthly_reference_data: Reference monthly data for computing target stats.
            log_histograms: Whether to aggregate histograms.
            data_grid: The grid type of the data, used for spherical power spectrum.
            time_mean_reference_data: Reference time means for computing bias stats.
            channel_mean_names: Names over which to compute channel means. If not
                provided, all available variables will be used.
        """
        self._channel_mean_names = channel_mean_names
        self._aggregators: Dict[str, _EvaluatorAggregator] = {}
        self._time_dependent_aggregators: Dict[
            str, _TimeDependentEvaluatorAggregator
        ] = {}
        ops = horizontal_coordinates.gridded_operations
        self._log_time_series = (
            log_global_mean_time_series or log_global_mean_norm_time_series
        )
        if log_global_mean_time_series:
            self._aggregators["mean"] = MeanAggregator(
                ops,
                target="denorm",
                n_timesteps=n_timesteps,
                metadata=metadata,
            )
        if log_global_mean_norm_time_series:
            self._aggregators["mean_norm"] = MeanAggregator(
                ops,
                target="norm",
                n_timesteps=n_timesteps,
                metadata=metadata,
            )
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(
                ops, target_time=20
            )
        if isinstance(horizontal_coordinates, LatLonCoordinates):
            if log_zonal_mean_images:
                self._aggregators["zonal_mean"] = ZonalMeanAggregator(
                    n_timesteps=n_timesteps,
                    metadata=metadata,
                )
            self._aggregators[
                "spherical_power_spectrum"
            ] = PairedSphericalPowerSpectrumAggregator(
                horizontal_coordinates.area_weights.shape[-2],
                horizontal_coordinates.area_weights.shape[-1],
                horizontal_coordinates.grid,
            )
            if log_video:
                self._aggregators["video"] = VideoAggregator(
                    n_timesteps=n_timesteps,
                    enable_extended_videos=enable_extended_videos,
                    metadata=metadata,
                )
        self._aggregators["time_mean"] = TimeMeanEvaluatorAggregator(
            ops,
            horizontal_dims=horizontal_coordinates.dims,
            metadata=metadata,
            reference_means=time_mean_reference_data,
        )
        self._aggregators["time_mean_norm"] = TimeMeanEvaluatorAggregator(
            ops,
            horizontal_dims=horizontal_coordinates.dims,
            target="norm",
            metadata=metadata,
        )
        if log_histograms:
            self._aggregators["histogram"] = HistogramAggregator()
        if log_seasonal_means:
            self._time_dependent_aggregators["seasonal"] = SeasonalAggregator(
                ops=ops,
                metadata=metadata,
            )
        if n_timesteps * timestep > APPROXIMATELY_TWO_YEARS:
            self._time_dependent_aggregators["annual"] = GlobalMeanAnnualAggregator(
                ops=ops,
                timestep=timestep,
                metadata=metadata,
                monthly_reference_data=monthly_reference_data,
            )
        if n_timesteps * timestep > SLIGHTLY_LESS_THAN_FIVE_YEARS:
            self._time_dependent_aggregators[
                "enso_coefficient"
            ] = EnsoCoefficientEvaluatorAggregator(
                initial_times,
                n_timesteps - 1,
                timestep,
                gridded_operations=ops,
                metadata=metadata,
            )

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(
        self,
        data: PairedData,
        normalize: Callable[[TensorMapping], TensorDict],
        i_time_start: int,
    ):
        if len(data.prediction) == 0:
            raise ValueError("No prediction values in data")
        if len(data.target) == 0:
            raise ValueError("No target values in data")
        target_data = {k: v for k, v in data.target.items() if k in data.prediction}
        target_data_norm = normalize(target_data)
        gen_data_norm = normalize(data.prediction)
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                target_data=target_data,
                gen_data=data.prediction,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                i_time_start=i_time_start,
            )
        for time_dependent_aggregator in self._time_dependent_aggregators.values():
            time_dependent_aggregator.record_batch(
                time=data.times,
                target_data=target_data,
                gen_data=data.prediction,
            )

    def record_initial_condition(
        self,
        initial_condition: PrognosticStateABC[BatchData],
        normalize: Callable[[TensorMapping], TensorDict],
    ):
        data = initial_condition.as_state().device_data
        data_norm = normalize(data)
        for aggregator_name in ["mean", "mean_norm"]:
            aggregator = self._aggregators.get(aggregator_name)
            if aggregator is not None:
                aggregator.record_batch(
                    target_data=data,
                    gen_data=data,
                    target_data_norm=data_norm,
                    gen_data_norm=data_norm,
                    i_time_start=0,
                )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        for name, time_dependent_aggregator in self._time_dependent_aggregators.items():
            logs.update(time_dependent_aggregator.get_logs(label=name))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return logs

    @torch.no_grad()
    def get_inference_logs(self, label: str) -> List[Dict[str, Union[float, int]]]:
        """
        Returns a list of logs to report to WandB.

        This is done because in inference, we use the wandb step
        as the time step, meaning we need to re-organize the logged data
        from tables into a list of dictionaries.
        """
        return to_inference_logs(self.get_logs(label=label))

    @torch.no_grad()
    def get_inference_logs_slice(self, label: str, step_slice: slice):
        """
        Returns a subset of the time series for applicable metrics
        for a specific slice of as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            step_slice: Timestep slice to determine the time series subset.

        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            if isinstance(aggregator, MeanAggregator):
                logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return to_inference_logs(logs)

    @torch.no_grad()
    def get_datasets(
        self, excluded_aggregators: Optional[Iterable[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Returns datasets from combined aggregators.

        Args:
            excluded_aggregators: aggregator names for which `get_dataset`
                should not be called and no output should be returned.

        Returns:
            Dictionary of datasets from aggregators.
        """
        if excluded_aggregators is None:
            excluded_aggregators = []

        combined_aggregators = {
            **self._aggregators,
            **self._time_dependent_aggregators,
        }
        return {
            name: agg.get_dataset()
            for name, agg in combined_aggregators.items()
            if name not in excluded_aggregators
        }


def to_inference_logs(
    log: Mapping[str, Union[Table, float, int]],
) -> List[Dict[str, Union[float, int]]]:
    # we have a dictionary which contains WandB tables
    # which we will convert to a list of dictionaries, one for each
    # row in the tables. Any scalar values will be reported in the last
    # dictionary.
    n_rows = 0
    for val in log.values():
        if isinstance(val, Table):
            n_rows = max(n_rows, len(val.data))
    logs: List[Dict[str, Union[float, int]]] = []
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


def table_to_logs(table: Table) -> List[Dict[str, Union[float, int]]]:
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

    Attributes:
        time_mean_reference_data: Path to reference time means to compare against.
        log_global_mean_time_series: Whether to log global mean time series metrics.
    """

    time_mean_reference_data: Optional[str] = None
    log_global_mean_time_series: bool = True

    def build(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ) -> "InferenceAggregator":
        if self.time_mean_reference_data is not None:
            time_means = xr.open_dataset(self.time_mean_reference_data)
        else:
            time_means = None
        return InferenceAggregator(
            gridded_operations=gridded_operations,
            n_timesteps=n_timesteps,
            metadata=metadata,
            time_mean_reference_data=time_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
        )


class InferenceAggregator(
    InferenceAggregatorABC[PrognosticStateABC[BatchData], BatchData]
):
    """
    Aggregates statistics on a single timeseries of data.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
        time_mean_reference_data: Optional[xr.Dataset] = None,
        log_global_mean_time_series: bool = True,
    ):
        """
        Args:
            gridded_operations: Gridded operations for computing horizontal reductions.
            sigma_coordinates: Data sigma coordinates
            timestep: Timestep of the model.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            time_mean_reference_data: Reference time means for computing bias stats.
            log_global_mean_time_series: Whether to log global mean time series metrics.
        """
        self._log_time_series = log_global_mean_time_series
        aggregators: Dict[str, _Aggregator] = {}
        if log_global_mean_time_series:
            aggregators["mean"] = SingleTargetMeanAggregator(
                gridded_operations,
                n_timesteps=n_timesteps,
            )
        aggregators["time_mean"] = TimeMeanAggregator(
            gridded_operations=gridded_operations,
            metadata=metadata,
            reference_means=time_mean_reference_data,
        )
        self._aggregators = aggregators
        self._time_dependent_aggregators: Dict[str, _TimeDependentAggregator] = {}

    @property
    def log_time_series(self) -> bool:
        return self._log_time_series

    @torch.no_grad()
    def record_batch(
        self,
        data: BatchData,
        i_time_start: int,
        normalize: Optional[Callable[[TensorMapping], TensorDict]] = None,
    ):
        """
        Record a batch of data.

        Args:
            prediction: Batch of data to record.
            i_time_start: Start time index.
            normalize: Ignored, kept for API compatibility.
        """
        if len(data.data) == 0:
            raise ValueError("data is empty")
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                data=data.data,
                i_time_start=i_time_start,
            )
        for time_dependent_aggregator in self._time_dependent_aggregators.values():
            time_dependent_aggregator.record_batch(
                time=data.times,
                data=data.data,
            )

    def record_initial_condition(
        self,
        initial_condition: PrognosticStateABC[BatchData],
        normalize: Callable[[TensorMapping], TensorDict],
    ):
        data = initial_condition.as_state().device_data
        if "mean" in self._aggregators:
            self._aggregators["mean"].record_batch(
                data=data,
                i_time_start=0,
            )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            logs.update(aggregator.get_logs(label=name))
        for name, time_dependent_aggregator in self._time_dependent_aggregators.items():
            logs.update(time_dependent_aggregator.get_logs(label=name))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return logs

    @torch.no_grad()
    def get_inference_logs(self, label: str) -> List[Dict[str, Union[float, int]]]:
        """
        Returns a list of logs to report to WandB.

        This is done because in inference, we use the wandb step
        as the time step, meaning we need to re-organize the logged data
        from tables into a list of dictionaries.
        """
        return to_inference_logs(self.get_logs(label=label))

    @torch.no_grad()
    def get_inference_logs_slice(self, label: str, step_slice: slice):
        """
        Returns a subset of the time series for applicable metrics
        for a specific slice of as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            step_slice: Timestep slice to determine the time series subset.

        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            if isinstance(aggregator, SingleTargetMeanAggregator):
                logs.update(aggregator.get_logs(label=name, step_slice=step_slice))
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return to_inference_logs(logs)

    @torch.no_grad()
    def get_datasets(
        self, excluded_aggregators: Optional[Iterable[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Returns datasets from combined aggregators.

        Args:
            excluded_aggregators: aggregator names for which `get_dataset`
                should not be called and no output should be returned.

        Returns:
            Dictionary of datasets from aggregators.
        """
        if excluded_aggregators is None:
            excluded_aggregators = []

        combined_aggregators = {
            **self._aggregators,
            **self._time_dependent_aggregators,
        }
        return {
            name: agg.get_dataset()
            for name, agg in combined_aggregators.items()
            if name not in excluded_aggregators
        }
