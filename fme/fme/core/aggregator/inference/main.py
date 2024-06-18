import dataclasses
import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Union

import torch
import xarray as xr

from fme.core.data_loading.data_typing import SigmaCoordinates, VariableMetadata
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Table, WandB

from ..one_step.reduced import MeanAggregator as OneStepMeanAggregator
from .annual import GlobalMeanAnnualAggregator
from .histogram import HistogramAggregator
from .reduced import MeanAggregator
from .seasonal import SeasonalAggregator
from .time_mean import TimeMeanAggregator
from .video import VideoAggregator
from .zonal_mean import ZonalMeanAggregator

wandb = WandB.get_instance()
APPROXIMATELY_TWO_YEARS = datetime.timedelta(days=730)


class _Aggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        gen_data: TensorMapping,
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
        loss: float,
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
        gen_data: TensorMapping,
    ):
        ...

    @torch.no_grad()
    def get_logs(self, label: str):
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
        monthly_reference_data: Path to monthly reference data to compare against.
    """

    log_histograms: bool = False
    log_video: bool = False
    log_extended_video: bool = False
    log_zonal_mean_images: bool = True
    log_seasonal_means: bool = False
    monthly_reference_data: Optional[str] = None

    def build(self, **kwargs) -> "InferenceEvaluatorAggregator":
        if self.monthly_reference_data is None:
            monthly_reference_data = None
        else:
            monthly_reference_data = xr.open_dataset(self.monthly_reference_data)

        return InferenceEvaluatorAggregator(
            log_histograms=self.log_histograms,
            log_video=self.log_video,
            enable_extended_videos=self.log_extended_video,
            log_zonal_mean_images=self.log_zonal_mean_images,
            log_seasonal_means=self.log_seasonal_means,
            monthly_reference_data=monthly_reference_data,
            **kwargs,
        )


class InferenceEvaluatorAggregator:
    """
    Aggregates statistics for inference comparing a generated and target series.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
        n_timesteps: int,
        record_step_20: bool = False,
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_zonal_mean_images: bool = False,
        log_seasonal_means: bool = False,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
        monthly_reference_data: Optional[xr.Dataset] = None,
        log_histograms: bool = False,
    ):
        """
        Args:
            area_weights: Area weights for each grid cell.
            sigma_coordinates: Data sigma coordinates
            timestep: Timestep of the model.
            n_timesteps: Number of timesteps of inference that will be run.
            record_step_20: Whether to record the mean of the 20th steps.
            log_video: Whether to log videos of the state evolution.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension.
            log_seasonal_means: Whether to log seasonal means metrics and images.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            monthly_reference_data: Reference monthly data for computing target stats.
            log_histograms: Whether to aggregate histograms.
        """
        self._aggregators: Dict[str, _EvaluatorAggregator] = {
            "mean": MeanAggregator(
                area_weights,
                target="denorm",
                n_timesteps=n_timesteps,
                metadata=metadata,
            ),
            "mean_norm": MeanAggregator(
                area_weights,
                target="norm",
                n_timesteps=n_timesteps,
                metadata=metadata,
            ),
            "time_mean": TimeMeanAggregator(
                area_weights,
                metadata=metadata,
            ),
            "time_mean_norm": TimeMeanAggregator(
                area_weights,
                target="norm",
                metadata=metadata,
            ),
        }
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(
                area_weights, target_time=20
            )
        if log_video:
            self._aggregators["video"] = VideoAggregator(
                n_timesteps=n_timesteps,
                enable_extended_videos=enable_extended_videos,
                metadata=metadata,
            )
        if log_zonal_mean_images:
            self._aggregators["zonal_mean"] = ZonalMeanAggregator(
                n_timesteps=n_timesteps,
                metadata=metadata,
            )
        self._time_dependent_aggregators: Dict[
            str, _TimeDependentEvaluatorAggregator
        ] = {}
        if log_seasonal_means:
            self._time_dependent_aggregators["seasonal"] = SeasonalAggregator(
                area_weights=area_weights,
                metadata=metadata,
            )

        if n_timesteps * timestep > APPROXIMATELY_TWO_YEARS:
            self._time_dependent_aggregators["annual"] = GlobalMeanAnnualAggregator(
                area_weights=area_weights,
                timestep=timestep,
                metadata=metadata,
                monthly_reference_data=monthly_reference_data,
            )
        if log_histograms:
            self._aggregators["histogram"] = HistogramAggregator()

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")
        target_data = {k: v for k, v in target_data.items() if k in gen_data}
        target_data_norm = {k: v for k, v in target_data_norm.items() if k in gen_data}
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                loss=loss,
                target_data=target_data,
                gen_data=gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                i_time_start=i_time_start,
            )
        for time_dependent_aggregator in self._time_dependent_aggregators.values():
            time_dependent_aggregator.record_batch(
                time=time,
                target_data=target_data,
                gen_data=gen_data,
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
    def get_datasets(
        self, aggregator_whitelist: Optional[Iterable[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Args:
            aggregator_whitelist: aggregator names to include in the output. If
                None, return all the datasets associated with all aggregators.
        """
        datasets = (
            (name, agg.get_dataset()) for name, agg in self._aggregators.items()
        )
        if aggregator_whitelist is not None:
            filter_ = set(aggregator_whitelist)
            return {name: ds for name, ds in datasets if name in filter_}

        return {name: ds for name, ds in datasets}


def to_inference_logs(
    log: Mapping[str, Union[Table, float, int]]
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
    for i in range(n_rows):
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
    """

    def build(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ) -> "InferenceAggregator":
        return InferenceAggregator(
            area_weights=area_weights,
            sigma_coordinates=sigma_coordinates,
            timestep=timestep,
            metadata=metadata,
        )


class InferenceAggregator:
    """
    Aggregates statistics on a single timeseries of data.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        """
        Args:
            area_weights: Area weights for each grid cell.
            sigma_coordinates: Data sigma coordinates
            timestep: Timestep of the model.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._aggregators: Dict[str, _Aggregator] = {}
        self._time_dependent_aggregators: Dict[str, _TimeDependentAggregator] = {}

    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        data: TensorMapping,
    ):
        if len(data) == 0:
            raise ValueError("No data in gen_data")
        for aggregator in self._aggregators.values():
            aggregator.record_batch(
                gen_data=data,
            )
        for time_dependent_aggregator in self._time_dependent_aggregators.values():
            time_dependent_aggregator.record_batch(
                time=time,
                gen_data=data,
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
    def get_datasets(
        self, aggregator_whitelist: Optional[Iterable[str]] = None
    ) -> Dict[str, xr.Dataset]:
        """
        Args:
            aggregator_whitelist: aggregator names to include in the output. If
                None, return all the datasets associated with all aggregators.
        """
        datasets = (
            (name, agg.get_dataset()) for name, agg in self._aggregators.items()
        )
        if aggregator_whitelist is not None:
            filter_ = set(aggregator_whitelist)
            return {name: ds for name, ds in datasets if name in filter_}

        return {name: ds for name, ds in datasets}
