import dataclasses
import datetime
import warnings
from typing import Callable, Dict, Mapping, Optional, Union

import torch
import xarray as xr

from fme.ace.aggregator.inference.main import (
    InferenceEvaluatorAggregator as InferenceEvaluatorAggregator_,
)
from fme.ace.aggregator.one_step.main import OneStepAggregator as OneStepAggregator_
from fme.core.coordinates import HorizontalCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import (
    AggregatorABC,
    InferenceAggregatorABC,
    InferenceLog,
    InferenceLogs,
)
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.data_typing import CoupledVerticalCoordinate
from fme.coupled.stepper import CoupledTrainOutput


class TrainAggregator(AggregatorABC[CoupledTrainOutput]):
    def __init__(self):
        self._n_batches = 0
        self._loss = torch.tensor(0.0, device=get_device())

    @torch.no_grad()
    def record_batch(self, batch: CoupledTrainOutput):
        self._loss += batch.metrics["loss"]
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, torch.Tensor]:
        logs = {f"{label}/mean/loss": self._loss / self._n_batches}
        dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs


class OneStepAggregator(AggregatorABC[CoupledTrainOutput]):
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        ocean_loss_scaling: Optional[TensorMapping] = None,
        atmosphere_loss_scaling: Optional[TensorMapping] = None,
    ):
        """
        Args:
            gridded_operations: Operations for computing metrics on gridded data.
            variable_metadata: Metadata for each variable.
            ocean_loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation for the ocean stepper.
            atmosphere_loss_scaling: Dictionary of variables and their scaling factors
                used in loss computation for the atmosphere stepper.
        """
        self._dist = Distributed.get_instance()
        self._loss = torch.tensor(0.0, device=get_device())
        self._n_batches = 0
        self._aggregators = {
            "ocean": OneStepAggregator_(
                gridded_operations, variable_metadata, ocean_loss_scaling
            ),
            "atmosphere": OneStepAggregator_(
                gridded_operations, variable_metadata, atmosphere_loss_scaling
            ),
        }

    @torch.no_grad()
    def record_batch(
        self,
        batch: CoupledTrainOutput,
    ):
        self._loss += batch.metrics["loss"]
        self._aggregators["ocean"].record_batch(batch.ocean_data)
        self._aggregators["atmosphere"].record_batch(batch.atmosphere_data)
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        ocean_logs = self._aggregators["ocean"].get_logs(label)
        atmos_logs = self._aggregators["atmosphere"].get_logs(label)
        # loss is not included in component metrics so these are both nans
        ocean_logs.pop(f"{label}/mean/loss")
        atmos_logs.pop(f"{label}/mean/loss")
        duplicates = set(ocean_logs.keys()) & set(atmos_logs.keys())
        if len(duplicates) > 0:
            raise ValueError(
                "Duplicate keys found in ocean and atmosphere "
                f"{label} logs: {duplicates}."
            )
        logs = {**ocean_logs, **atmos_logs}
        loss = self._loss / self._n_batches
        logs[f"{label}/mean/loss"] = float(
            self._dist.reduce_mean(loss.detach()).cpu().numpy()
        )
        return logs


@dataclasses.dataclass
class InferenceEvaluatorAggregatorConfig:
    """
    Configuration for coupled inference evaluator aggregator.

    Parameters:
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
            This should include both ocean and atmosphere variables.
        time_mean_reference_data: Path to reference time means to compare against.
            This should include both ocean and atmosphere variables.
    """

    log_histograms: bool = False
    log_video: bool = False
    log_extended_video: bool = False
    log_zonal_mean_images: bool = False
    log_seasonal_means: bool = False
    log_global_mean_time_series: bool = True
    log_global_mean_norm_time_series: bool = True
    monthly_reference_data: Optional[str] = None
    time_mean_reference_data: Optional[str] = None

    def __post_init__(self):
        if self.log_global_mean_time_series or self.log_global_mean_norm_time_series:
            warnings.warn("Time series logging is not implemented for coupled models.")
            self.log_global_mean_time_series = False
            self.log_global_mean_norm_time_series = False

    def build(
        self,
        vertical_coordinate: CoupledVerticalCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        ocean_timestep: datetime.timedelta,
        atmosphere_timestep: datetime.timedelta,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        initial_time: xr.DataArray,
        ocean_normalize: Callable[[TensorMapping], TensorDict],
        atmosphere_normalize: Callable[[TensorMapping], TensorDict],
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ) -> "InferenceEvaluatorAggregator":
        if self.monthly_reference_data is None:
            monthly_reference_data = None
        else:
            monthly_reference_data = xr.open_dataset(self.monthly_reference_data)
        if self.time_mean_reference_data is None:
            time_mean = None
        else:
            time_mean = xr.open_dataset(self.time_mean_reference_data)

        if n_timesteps_atmosphere > 2**15 and self.log_zonal_mean_images:
            # matplotlib raises an error if image size is too large, and we plot
            # one pixel per timestep in the zonal mean images.
            warnings.warn(
                "Disabling zonal mean images logging due to large number of timesteps"
                f" (total atmosphere steps is {n_timesteps_atmosphere}). Set "
                "log_zonal_mean_images=False or "
                "decrease n_coupled_steps to avoid this warning."
            )
            log_zonal_mean_images = False
        else:
            log_zonal_mean_images = self.log_zonal_mean_images

        return InferenceEvaluatorAggregator(
            vertical_coordinate=vertical_coordinate,
            horizontal_coordinates=horizontal_coordinates,
            ocean_timestep=ocean_timestep,
            atmosphere_timestep=atmosphere_timestep,
            n_timesteps_ocean=n_timesteps_ocean,
            n_timesteps_atmosphere=n_timesteps_atmosphere,
            initial_time=initial_time,
            log_histograms=self.log_histograms,
            log_video=self.log_video,
            enable_extended_videos=self.log_extended_video,
            log_zonal_mean_images=log_zonal_mean_images,
            log_seasonal_means=self.log_seasonal_means,
            log_global_mean_time_series=self.log_global_mean_time_series,
            log_global_mean_norm_time_series=self.log_global_mean_norm_time_series,
            monthly_reference_data=monthly_reference_data,
            time_mean_reference_data=time_mean,
            variable_metadata=variable_metadata,
            ocean_normalize=ocean_normalize,
            atmosphere_normalize=atmosphere_normalize,
        )


class InferenceEvaluatorAggregator(
    InferenceAggregatorABC[
        Union[CoupledPairedData, CoupledPrognosticState],
        CoupledPairedData,
    ]
):
    def __init__(
        self,
        vertical_coordinate: CoupledVerticalCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        ocean_timestep: datetime.timedelta,
        atmosphere_timestep: datetime.timedelta,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        initial_time: xr.DataArray,
        ocean_normalize: Callable[[TensorMapping], TensorDict],
        atmosphere_normalize: Callable[[TensorMapping], TensorDict],
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_zonal_mean_images: bool = False,
        log_seasonal_means: bool = False,
        log_global_mean_time_series: bool = True,
        log_global_mean_norm_time_series: bool = True,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        monthly_reference_data: Optional[xr.Dataset] = None,
        log_histograms: bool = False,
        time_mean_reference_data: Optional[xr.Dataset] = None,
    ):
        self._record_ocean_step_20 = n_timesteps_ocean >= 20
        self._record_atmos_step_20 = n_timesteps_atmosphere >= 20
        self._aggregators = {
            "ocean": InferenceEvaluatorAggregator_(
                vertical_coordinate=vertical_coordinate.ocean,
                horizontal_coordinates=horizontal_coordinates,
                timestep=ocean_timestep,
                n_timesteps=n_timesteps_ocean,
                initial_time=initial_time,
                log_histograms=log_histograms,
                log_video=log_video,
                enable_extended_videos=enable_extended_videos,
                log_zonal_mean_images=log_zonal_mean_images,
                log_seasonal_means=log_seasonal_means,
                log_global_mean_time_series=log_global_mean_time_series,
                log_global_mean_norm_time_series=log_global_mean_norm_time_series,
                monthly_reference_data=monthly_reference_data,
                time_mean_reference_data=time_mean_reference_data,
                record_step_20=self._record_ocean_step_20,
                variable_metadata=variable_metadata,
                channel_mean_names=None,
                normalize=ocean_normalize,
            ),
            "atmosphere": InferenceEvaluatorAggregator_(
                vertical_coordinate=vertical_coordinate.atmosphere,
                horizontal_coordinates=horizontal_coordinates,
                timestep=atmosphere_timestep,
                n_timesteps=n_timesteps_atmosphere,
                initial_time=initial_time,
                log_histograms=log_histograms,
                log_video=log_video,
                enable_extended_videos=enable_extended_videos,
                log_zonal_mean_images=log_zonal_mean_images,
                log_seasonal_means=log_seasonal_means,
                log_global_mean_time_series=log_global_mean_time_series,
                log_global_mean_norm_time_series=log_global_mean_norm_time_series,
                monthly_reference_data=monthly_reference_data,
                time_mean_reference_data=time_mean_reference_data,
                record_step_20=self._record_atmos_step_20,
                variable_metadata=variable_metadata,
                channel_mean_names=None,
                normalize=atmosphere_normalize,
            ),
        }
        self._num_channels_ocean: Optional[int] = None
        self._num_channels_atmos: Optional[int] = None

    def _init_num_channels(self, data: CoupledPairedData):
        if self._num_channels_ocean is None:
            self._num_channels_ocean = len(data.ocean_data.prediction)
            self._num_channels_atmos = len(data.atmosphere_data.prediction)

    @torch.no_grad()
    def record_batch(self, data: CoupledPairedData) -> InferenceLogs:
        # TODO: combine and return log sequences
        self._init_num_channels(data)
        _ = self._aggregators["ocean"].record_batch(data.ocean_data)
        _ = self._aggregators["atmosphere"].record_batch(data.atmosphere_data)
        return []

    @torch.no_grad()
    def record_initial_condition(
        self,
        initial_condition: Union[CoupledPairedData, CoupledPrognosticState],
    ) -> InferenceLogs:
        """
        Record the initial condition.

        May only be recorded once, before any calls to record_batch.
        """
        # TODO: combine and return initial condition logs
        _ = self._aggregators["ocean"].record_initial_condition(
            initial_condition.ocean_data
        )
        _ = self._aggregators["atmosphere"].record_initial_condition(
            initial_condition.atmosphere_data
        )
        return [{}]

    @torch.no_grad()
    def get_summary_logs(self) -> InferenceLog:
        if self._num_channels_ocean is None or self._num_channels_atmos is None:
            raise ValueError("No data recorded.")
        ocean_logs = self._aggregators["ocean"].get_summary_logs()
        atmos_logs = self._aggregators["atmosphere"].get_summary_logs()
        ocean_channel_mean = ocean_logs.pop("time_mean_norm/rmse/channel_mean")
        atmos_channel_mean = atmos_logs.pop("time_mean_norm/rmse/channel_mean")
        ocean_logs["time_mean_norm/rmse/ocean_channel_mean"] = ocean_channel_mean
        atmos_logs["time_mean_norm/rmse/atmosphere_channel_mean"] = atmos_channel_mean
        # "mean_step_20/loss" is NaN since inference has no loss
        ocean_logs.pop("mean_step_20/loss", None)
        atmos_logs.pop("mean_step_20/loss", None)
        duplicates = set(ocean_logs.keys()) & set(atmos_logs.keys())
        if len(duplicates) > 0:
            raise ValueError(
                "Duplicate keys found in ocean and atmosphere "
                f"inference evaluator aggregator logs: {duplicates}."
            )
        num_channels = self._num_channels_ocean + self._num_channels_atmos
        channel_mean = (
            ocean_channel_mean * self._num_channels_ocean
            + atmos_channel_mean * self._num_channels_atmos
        ) / num_channels
        return {
            "time_mean_norm/rmse/channel_mean": channel_mean,
            **ocean_logs,
            **atmos_logs,
        }
