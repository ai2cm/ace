import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.normalizer import NormalizeFn
from fme.core.step.step_diagnostics import StepDiagnostics
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import Image

from ..plotting import plot_paneled_data
from .reduced import (
    AreaWeightedSingleTargetReducedMetric,
    SeriesData,
    data_to_table,
    get_series_data,
    series_data_to_dataset,
)


class StepDiagnosticsSubAggregator(Protocol):
    """A per-concern aggregator of metrics from the StepDiagnostics carried
    on prediction data. Entries in the StepDiagnosticsAggregator's dict
    satisfy this; the wrapper delegates record/log/flush calls without
    knowing what any entry measures.
    """

    @property
    def log_time_series(self) -> bool: ...

    def record_batch(
        self,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None: ...

    def summary_logs(self) -> dict[str, Any]: ...

    def time_series_logs(self, step_slice: slice) -> dict[str, Any]: ...

    def diagnostics(self) -> dict[str, xr.Dataset]: ...


class StepDiagnosticsAggregator:
    """The one aggregator the hosts see: wraps the dict of per-concern
    sub-aggregators and fans record_batch / summary_logs / time_series_logs /
    diagnostics out over it, merging the results. Owns the
    combine-the-sub-aggregators logic so the hosts don't. The dict keys name
    the sub-aggregators' diagnostics files.
    """

    def __init__(self, aggregators: dict[str, StepDiagnosticsSubAggregator]):
        self._aggregators = aggregators

    @property
    def log_time_series(self) -> bool:
        return any(agg.log_time_series for agg in self._aggregators.values())

    def record_batch(
        self,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None:
        for aggregator in self._aggregators.values():
            aggregator.record_batch(step_diagnostics, i_time_start)

    def summary_logs(self) -> dict[str, Any]:
        logs: dict[str, Any] = {}
        for aggregator in self._aggregators.values():
            logs.update(aggregator.summary_logs())
        return logs

    def time_series_logs(self, step_slice: slice) -> dict[str, Any]:
        logs: dict[str, Any] = {}
        for aggregator in self._aggregators.values():
            logs.update(aggregator.time_series_logs(step_slice))
        return logs

    def diagnostics(self) -> dict[str, xr.Dataset]:
        datasets: dict[str, xr.Dataset] = {}
        for name, aggregator in self._aggregators.items():
            for suffix, ds in aggregator.diagnostics().items():
                datasets[f"{suffix}_{name}"] = ds
        return datasets


@dataclasses.dataclass
class StepDiagnosticsMetricConfig:
    """Granularity of StepDiagnostics-derived metrics.

    Controls the correction-delta metrics, which are uniform over all
    corrector-modified variables and merge into the existing
    ``time_mean_norm`` / ``mean_norm`` metric groups.

    Parameters:
        correction_scalars: Whether to log the correction scalar metrics:
            the time-mean ``correction_magnitude`` per variable, and (where
            per-step time series are enabled) the per-step
            ``weighted_correction_magnitude`` and ``weighted_correction_std``
            series.
        correction_maps: Whether to log the signed time-mean normalized
            correction map image per variable.
    """

    correction_scalars: bool = True
    correction_maps: bool = False

    def build(
        self,
        gridded_operations: GriddedOperations,
        n_timesteps: int,
        variable_metadata: Mapping[str, VariableMetadata] | None,
        enable_time_series: bool,
        normalize: NormalizeFn | None,
    ) -> StepDiagnosticsAggregator | None:
        """Build the aggregator, or return None when nothing is enabled or
        no normalizer is available (the correction metrics are
        normalized-space, so callers that supply no normalizer and keep the
        default configuration keep their current behavior). Raises ValueError
        when the configuration was explicitly changed from the defaults but
        no normalizer is supplied, rather than silently dropping an explicit
        opt-in.
        """
        if not (self.correction_scalars or self.correction_maps):
            return None
        if normalize is None:
            if self != StepDiagnosticsMetricConfig():
                raise ValueError(
                    "step_diagnostics metrics were explicitly configured, but "
                    "the aggregator was built without a normalizer, which the "
                    "normalized-space correction metrics require. Supply a "
                    "normalizer or leave the step_diagnostics configuration "
                    "at its defaults."
                )
            return None
        time_mean = CorrectionDeltaTimeMeanAggregator(
            gridded_operations=gridded_operations,
            variable_metadata=variable_metadata,
            record_scalars=self.correction_scalars,
            record_maps=self.correction_maps,
        )
        if self.correction_scalars and enable_time_series:
            time_series = CorrectionDeltaMeanAggregator(
                gridded_operations=gridded_operations,
                n_timesteps=n_timesteps,
                variable_metadata=variable_metadata,
            )
        else:
            time_series = None
        return StepDiagnosticsAggregator(
            {
                "correction": CorrectionDeltaAggregator(
                    normalize=normalize,
                    time_mean=time_mean,
                    time_series=time_series,
                )
            }
        )


class CorrectionDeltaAggregator:
    """StepDiagnosticsSubAggregator for the corrector's correction delta,
    quantifying how much a model relies on its corrector in the same
    normalized space as the existing ``*_norm`` metrics. Computes the
    normalized correction once per batch as ``normalize(delta,
    apply_mean=False)``, i.e. ``delta / std``, and dispatches it to the
    granular sub-aggregators, which the hosts never see. The mean
    subtraction is skipped because the delta is a difference quantity;
    centering would wrongly subtract the per-variable mean offset. Off-mask
    cells stay NaN, consistent with the masked prediction; the mask-aware
    gridded reductions handle them.

    Silent when no corrector ran: ``step_diagnostics`` is then ``None`` (or
    its delta empty), no batches are recorded, and the log and dataset
    methods return nothing.

    Not a registered sub-aggregator of the hosts' one-name-per-aggregator
    registry because (a) the metrics consume the step diagnostics, which
    InferenceBatchData does not carry, and (b) the logs merge into the
    existing time_mean_norm / mean_norm label groups, which the registry
    cannot express.
    """

    TIME_MEAN_LABEL = "time_mean_norm"
    TIME_SERIES_LABEL = "mean_norm"

    def __init__(
        self,
        normalize: NormalizeFn,
        time_mean: "CorrectionDeltaTimeMeanAggregator | None",
        time_series: "CorrectionDeltaMeanAggregator | None",
    ):
        self._normalize = normalize
        self._time_mean = time_mean
        self._time_series = time_series

    @property
    def log_time_series(self) -> bool:
        return self._time_series is not None

    @torch.no_grad()
    def record_batch(
        self,
        step_diagnostics: StepDiagnostics | None,
        i_time_start: int,
    ) -> None:
        if step_diagnostics is None:
            # no corrector ran, or the prediction pipeline does not attach
            # diagnostics
            return
        delta = step_diagnostics.delta
        if not delta:
            return
        correction_norm = self._normalize(delta, apply_mean=False)
        missing = set(delta) - set(correction_norm)
        if missing:
            raise ValueError(
                "The normalizer has no normalization constants for the "
                f"correction delta variables {sorted(missing)}; their "
                "correction metrics cannot be computed."
            )
        if self._time_mean is not None:
            self._time_mean.record_batch(correction_norm)
        if self._time_series is not None:
            self._time_series.record_batch(correction_norm, i_time_start)

    def summary_logs(self) -> dict[str, Any]:
        if self._time_mean is None:
            return {}
        return self._time_mean.get_logs(label=self.TIME_MEAN_LABEL)

    def time_series_logs(self, step_slice: slice) -> dict[str, Any]:
        if self._time_series is None:
            return {}
        return self._time_series.get_logs(
            label=self.TIME_SERIES_LABEL, step_slice=step_slice
        )

    def diagnostics(self) -> dict[str, xr.Dataset]:
        # Suffixed dataset names: the plain "time_mean_norm"/"mean_norm"
        # diagnostics files belong to the existing aggregators sharing those
        # label groups.
        datasets: dict[str, xr.Dataset] = {}
        if self._time_mean is not None:
            datasets[self.TIME_MEAN_LABEL] = self._time_mean.get_dataset()
        if self._time_series is not None:
            datasets[self.TIME_SERIES_LABEL] = self._time_series.get_dataset()
        return datasets


class CorrectionDeltaTimeMeanAggregator:
    """Granular sub-aggregator: time-mean maps and scalars of the normalized
    correction. Silent (empty logs and dataset) until non-empty data is
    recorded.
    """

    _MAGNITUDE_CAPTION = "{name} time-mean |normalized correction|"
    _MAP_CAPTION = "{name} time-mean normalized correction (corrected - uncorrected)"

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        record_scalars: bool = True,
        record_maps: bool = False,
    ):
        self._ops = gridded_operations
        self._variable_metadata: Mapping[str, VariableMetadata] = (
            variable_metadata or {}
        )
        self._record_scalars = record_scalars
        self._record_maps = record_maps
        self._signed_sum: TensorDict = {}
        self._magnitude_sum: TensorDict = {}
        self._n_timesteps = 0
        self._n_samples = 0

    @torch.no_grad()
    def record_batch(self, correction_norm: TensorMapping) -> None:
        """Accumulate one time window. The time-mean arithmetic assumes each
        call carries the same sample count and the same variable set (the
        inference loop's contiguous-windows contract); violations raise
        rather than silently biasing the means.
        """
        if not correction_norm:
            return
        sample_dim, time_dim = 0, 1
        first = next(iter(correction_norm.values()))
        if self._signed_sum:
            if set(correction_norm) != set(self._signed_sum):
                raise ValueError(
                    "The correction delta variable set must be constant "
                    f"across batches; previously recorded "
                    f"{sorted(self._signed_sum)}, got "
                    f"{sorted(correction_norm)}."
                )
            if first.size(sample_dim) != self._n_samples:
                raise ValueError(
                    "The correction delta sample count must be constant "
                    f"across batches; previously recorded {self._n_samples}, "
                    f"got {first.size(sample_dim)}."
                )
        for name, tensor in correction_norm.items():
            signed = tensor.sum(dim=time_dim).sum(dim=sample_dim)
            magnitude = tensor.abs().sum(dim=time_dim).sum(dim=sample_dim)
            if name in self._signed_sum:
                self._signed_sum[name] += signed
                self._magnitude_sum[name] += magnitude
            else:
                self._signed_sum[name] = signed
                self._magnitude_sum[name] = magnitude
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
        for name in signed:
            if self._record_scalars:
                # mask-aware reduction: the per-name spatial mask folds into
                # the area weights, so NaN off-mask cells drop out
                logs[f"correction_magnitude/{name}"] = float(
                    self._ops.area_weighted_mean(magnitude[name], name=name)
                    .cpu()
                    .numpy()
                )
            if self._record_maps:
                logs[f"correction_map/{name}"] = plot_paneled_data(
                    [[signed[name].cpu().numpy()]],
                    diverging=True,
                    caption=self._caption(self._MAP_CAPTION, name),
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
        if not self._has_data or not self._record_maps:
            return xr.Dataset()
        signed, _ = self._get_time_means()
        dims = ("lat", "lon")
        data = {}
        for name, value in signed.items():
            data[f"correction_map-{name}"] = xr.DataArray(value.cpu(), dims=dims)
        return xr.Dataset(data)


class CorrectionDeltaMeanAggregator:
    """Granular sub-aggregator: per-forecast-step area-weighted series of the
    normalized correction, mirroring the ``mean_norm`` per-step structure:
    for each corrector-modified variable it tracks the area-weighted global
    mean of the correction magnitude (``weighted_correction_magnitude``) and
    the area-weighted spatial standard deviation of the signed correction
    (``weighted_correction_std``, mirroring ``weighted_std_gen``). Silent
    (empty logs and dataset) until non-empty data is recorded.
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
    def record_batch(self, correction_norm: TensorMapping, i_time_start: int) -> None:
        if not correction_norm:
            return
        for metric in self._variable_metrics.values():
            metric.record(tensors=correction_norm, i_time_start=i_time_start)
        self._n_batches += 1

    def _get_series_data(self, step_slice: slice | None = None) -> list[SeriesData]:
        return get_series_data(self._variable_metrics, self._dist, step_slice)

    @torch.no_grad()
    def get_logs(self, label: str, step_slice: slice | None = None) -> dict[str, Any]:
        if self._n_batches == 0:
            return {}
        series_data: dict[str, np.ndarray] = {
            datum.get_wandb_key(): datum.data
            for datum in self._get_series_data(step_slice)
        }
        init_step = 0 if step_slice is None else step_slice.start
        # A distinct table key (vs the ``{label}/series`` of the main
        # aggregator sharing this label) avoids a collision while still
        # resolving to the ``{label}/...`` per-step keys, since
        # ``to_inference_logs`` strips everything after the final "/" to form
        # the column prefix.
        return {f"{label}/correction_series": data_to_table(series_data, init_step)}

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        if self._n_batches == 0:
            return xr.Dataset()
        return series_data_to_dataset(self._get_series_data(), self._variable_metadata)
