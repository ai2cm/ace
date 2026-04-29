from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import xarray as xr

from fme.core.typing_ import TensorDict, TensorMapping

if TYPE_CHECKING:
    from fme.core.dataset_info import DatasetInfo

    from .main import InferenceAggregator, InferenceEvaluatorAggregator


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
    ) -> InferenceEvaluatorAggregator:
        from .main import InferenceEvaluatorAggregator

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
        return InferenceEvaluatorAggregator(
            dataset_info=dataset_info,
            n_ic_steps=n_ic_steps,
            n_forward_steps=n_forward_steps,
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
            log_step_means=self.log_step_means,
            channel_mean_names=channel_mean_names,
            log_nino34_index=self.log_nino34_index,
            normalize=normalize,
            save_diagnostics=save_diagnostics,
            n_ensemble_per_ic=n_ensemble_per_ic,
        )


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
    ) -> InferenceAggregator:
        from .main import InferenceAggregator

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
