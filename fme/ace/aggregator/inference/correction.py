"""Aggregators for metrics of the corrector's correction during inference.

A stepper equipped with a corrector adjusts the network's raw output each step.
The "correction" is ``output - uncorrected`` for exactly the corrector-modified
variables. These aggregators quantify how much a model relies on its corrector,
computed in the same normalized space as the existing ``*_norm`` metrics:

    correction_norm = normalize(output) - normalize(uncorrected)

(Normalizing the raw delta would wrongly subtract the per-variable mean offset.)

The aggregators are silent when no corrector ran: the uncorrected prediction is
then empty, no batches are recorded, and ``get_logs`` returns nothing while
``get_dataset`` returns an empty dataset.
"""

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .reduced import AreaWeightedSingleTargetReducedMetric, _SeriesData, data_to_table


def compute_correction_norm(
    prediction: TensorMapping,
    uncorrected: TensorMapping,
    normalize: Callable[[TensorMapping], TensorDict],
) -> TensorDict:
    """Normalized-space correction for the corrector-modified variables.

    Computed as ``normalize(prediction) - normalize(uncorrected)`` over exactly
    the keys present in ``uncorrected`` (the sparse corrector-modified
    variables). Returns an empty dict when ``uncorrected`` is empty, so callers
    can use truthiness to detect whether a corrector was active.

    Args:
        prediction: Denormalized (corrected) prediction tensors.
        uncorrected: Denormalized pre-correction values of the corrector-modified
            variables. May be empty.
        normalize: The network normalizer's ``normalize`` callable.
    """
    if not uncorrected:
        return {}
    uncorrected_norm = normalize(uncorrected)
    prediction_norm = normalize({name: prediction[name] for name in uncorrected})
    return {
        name: prediction_norm[name] - uncorrected_norm[name]
        for name in uncorrected_norm
        if name in prediction_norm
    }


class CorrectionTimeMeanAggregator:
    """Time-mean maps and magnitudes of the normalized correction.

    Accumulates, per corrector-modified variable, the time-mean map of the
    signed normalized correction (logged as an image) and of its magnitude
    (whose area-weighted global mean is the ``correction_magnitude`` scalar).
    A ``channel_mean`` over the corrected variables only is also reported.
    """

    _MAGNITUDE_CAPTION = "{name} time-mean |normalized correction|"
    _MAP_CAPTION = "{name} time-mean normalized correction (output - uncorrected)"

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = gridded_operations
        self._variable_metadata: Mapping[str, VariableMetadata] = (
            variable_metadata or {}
        )
        self._signed_sum: TensorDict = {}
        self._magnitude_sum: TensorDict = {}
        self._n_timesteps = 0
        self._n_samples = 0

    @torch.no_grad()
    def record_batch(self, correction_norm: TensorMapping, i_time_start: int = 0):
        # The correction series has no initial-condition entry (uncorrected
        # values are produced only for forward steps), so every recorded
        # timestep contributes.
        if not correction_norm:
            return
        sample_dim, time_dim = 0, 1
        for name, tensor in correction_norm.items():
            signed = tensor.sum(dim=time_dim).sum(dim=sample_dim)
            magnitude = tensor.abs().sum(dim=time_dim).sum(dim=sample_dim)
            if name in self._signed_sum:
                self._signed_sum[name] += signed
                self._magnitude_sum[name] += magnitude
            else:
                self._signed_sum[name] = signed
                self._magnitude_sum[name] = magnitude
        first = next(iter(correction_norm.values()))
        self._n_samples = first.size(sample_dim)
        self._n_timesteps += first.size(time_dim)

    @property
    def _has_data(self) -> bool:
        return self._n_timesteps > 0 and len(self._signed_sum) > 0

    def _get_time_means(self) -> tuple[TensorDict, TensorDict]:
        dist = Distributed.get_instance()
        denom = self._n_timesteps * self._n_samples
        signed, magnitude = {}, {}
        for name in sorted(self._signed_sum.keys()):
            signed[name] = dist.reduce_mean(self._signed_sum[name] / denom)
            magnitude[name] = dist.reduce_mean(self._magnitude_sum[name] / denom)
        return signed, magnitude

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, float | Image]:
        if not self._has_data:
            return {}
        signed, magnitude = self._get_time_means()
        logs: dict[str, float | Image] = {}
        magnitudes: list[float] = []
        for name in signed:
            scalar = float(
                self._ops.area_weighted_mean(magnitude[name], name=name).cpu().numpy()
            )
            magnitudes.append(scalar)
            logs[f"correction_magnitude/{name}"] = scalar
            logs[f"correction_map/{name}"] = plot_paneled_data(
                [[signed[name].cpu().numpy()]],
                diverging=True,
                caption=self._caption(self._MAP_CAPTION, name),
            )
        if magnitudes:
            logs["correction_magnitude/channel_mean"] = sum(magnitudes) / len(
                magnitudes
            )
        if len(label) != 0:
            return {f"{label}/{key}": value for key, value in logs.items()}
        return logs

    def _caption(self, template: str, name: str) -> str:
        if name in self._variable_metadata:
            display_name = self._variable_metadata[name].display_long_name(name)
        else:
            display_name = name
        return template.format(name=display_name)

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        if not self._has_data:
            return xr.Dataset()
        signed, _ = self._get_time_means()
        dims = ("lat", "lon")
        data = {}
        for name, value in signed.items():
            data[f"correction_map-{name}"] = xr.DataArray(value.cpu(), dims=dims)
        return xr.Dataset(data)


class CorrectionMeanAggregator:
    """Per-step area-weighted time series of the normalized correction.

    Mirrors the ``mean_norm`` aggregator's per-step structure: for each
    corrector-modified variable it tracks the area-weighted global mean of the
    correction magnitude (``weighted_correction_magnitude``) and the
    area-weighted spatial standard deviation of the signed correction
    (``weighted_correction_std``, mirroring ``weighted_std_gen``).
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._ops = gridded_operations
        self._dist = Distributed.get_instance()
        self._variable_metadata: Mapping[str, VariableMetadata] = (
            variable_metadata or {}
        )
        device = get_device()
        self._variable_metrics = {
            "weighted_correction_magnitude": AreaWeightedSingleTargetReducedMetric(
                device=device,
                compute_metric=lambda tensors: self._ops.area_weighted_mean_dict(
                    {name: tensors[name].abs() for name in tensors}
                ),
                n_timesteps=n_timesteps,
            ),
            "weighted_correction_std": AreaWeightedSingleTargetReducedMetric(
                device=device,
                compute_metric=lambda tensors: self._ops.area_weighted_std_dict(
                    tensors
                ),
                n_timesteps=n_timesteps,
            ),
        }
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(self, correction_norm: TensorMapping, i_time_start: int):
        if not correction_norm:
            return
        for metric in self._variable_metrics.values():
            metric.record(tensors=correction_norm, i_time_start=i_time_start)
        self._n_batches += 1

    def _get_series_data(self, step_slice: slice | None = None) -> list[_SeriesData]:
        data: list[_SeriesData] = []
        for metric_name, metric in self._variable_metrics.items():
            metric_results = metric.get()
            for var_name in sorted(metric_results.keys()):
                arr = metric_results[var_name].detach()
                if step_slice is not None:
                    arr = arr[step_slice]
                data.append(
                    _SeriesData(
                        metric_name=metric_name,
                        var_name=var_name,
                        data=self._dist.reduce_mean(arr).cpu().numpy(),
                    )
                )
        return data

    @torch.no_grad()
    def get_logs(self, label: str, step_slice: slice | None = None):
        if self._n_batches == 0:
            return {}
        series_data: dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data
            for datum in self._get_series_data(step_slice)
        }
        init_step = 0 if step_slice is None else step_slice.start
        # A distinct table key (vs the ``{label}/series`` of the main aggregator
        # sharing this label) avoids a collision while still resolving to the
        # ``{label}/...`` per-step keys, since ``to_inference_logs`` strips
        # everything after the final "/" to form the column prefix.
        return {f"{label}/correction_series": data_to_table(series_data, init_step)}

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        if self._n_batches == 0:
            return xr.Dataset()
        data_vars = {}
        for datum in self._get_series_data():
            metadata = self._variable_metadata.get(
                datum.var_name, VariableMetadata("unknown_units", datum.var_name)
            )
            data_vars[datum.get_xarray_key()] = xr.DataArray(
                datum.data, dims=["forecast_step"], attrs=metadata.as_attrs()
            )
        n_forecast_steps = len(next(iter(data_vars.values())))
        coords = {"forecast_step": np.arange(n_forecast_steps)}
        return xr.Dataset(data_vars=data_vars, coords=coords)


@dataclasses.dataclass
class _CorrectionAggregators:
    """The correction aggregators built for one inference aggregator.

    ``time_mean`` is always built when correction metrics are enabled;
    ``time_series`` is built only when per-step time series are enabled (it is
    dropped for inline training-time inference, matching the existing
    ``mean``/``mean_norm`` time-series disable path).
    """

    time_mean: CorrectionTimeMeanAggregator
    time_series: CorrectionMeanAggregator | None


def build_correction_aggregators(
    gridded_operations: GriddedOperations,
    n_timesteps: int,
    variable_metadata: Mapping[str, VariableMetadata] | None,
    enable_time_series: bool,
) -> _CorrectionAggregators:
    return _CorrectionAggregators(
        time_mean=CorrectionTimeMeanAggregator(
            gridded_operations=gridded_operations,
            variable_metadata=variable_metadata,
        ),
        time_series=(
            CorrectionMeanAggregator(
                gridded_operations=gridded_operations,
                n_timesteps=n_timesteps,
                variable_metadata=variable_metadata,
            )
            if enable_time_series
            else None
        ),
    )


class CorrectionRecorder:
    """Records and reports correction metrics for an inference aggregator.

    Shared by the evaluator and no-target inference aggregators. Correction
    metrics live under the ``time_mean_norm`` and ``mean_norm`` label groups: in
    the evaluator they merge with the existing norm metrics under those labels;
    in the no-target aggregator they are the only members of those groups.

    Stays silent until a non-empty correction is recorded, so a stepper with no
    active corrector produces no correction logs and no diagnostics files.
    """

    TIME_MEAN_LABEL = "time_mean_norm"
    TIME_SERIES_LABEL = "mean_norm"

    def __init__(
        self,
        normalize: Callable[[TensorMapping], TensorDict],
        time_mean: CorrectionTimeMeanAggregator | None,
        time_series: CorrectionMeanAggregator | None,
    ):
        self._normalize = normalize
        self._time_mean = time_mean
        self._time_series = time_series
        self._recorded = False

    @property
    def enabled(self) -> bool:
        return self._time_mean is not None

    def record(
        self,
        prediction: TensorMapping,
        uncorrected: TensorMapping | None,
        i_time_start: int,
    ):
        if self._time_mean is None or not uncorrected:
            return
        correction_norm = compute_correction_norm(
            prediction, uncorrected, self._normalize
        )
        if not correction_norm:
            return
        self._time_mean.record_batch(correction_norm, i_time_start)
        if self._time_series is not None:
            self._time_series.record_batch(correction_norm, i_time_start)
        self._recorded = True

    def summary_logs(self) -> dict:
        if not self._recorded or self._time_mean is None:
            return {}
        return self._time_mean.get_logs(label=self.TIME_MEAN_LABEL)

    def time_series_logs(self, step_slice: slice) -> dict:
        if not self._recorded or self._time_series is None:
            return {}
        return self._time_series.get_logs(
            label=self.TIME_SERIES_LABEL, step_slice=step_slice
        )

    def diagnostics(self) -> dict:
        out: dict = {}
        if self._time_mean is not None:
            out["time_mean_norm_correction"] = self._time_mean
        if self._time_series is not None:
            out["mean_norm_correction"] = self._time_series
        return out
