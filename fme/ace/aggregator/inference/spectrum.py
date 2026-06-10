import dataclasses
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.fill import SmoothFloodFill
from fme.core.gridded_ops import GriddedOperations
from fme.core.metrics import spherical_power_spectrum
from fme.core.typing_ import TensorMapping

from .build_context import MetricBuildContext, MetricNotSupportedError, maybe_filter
from .data import InferenceBatchData, MetricBuildResult, SubAggregator


class SphericalPowerSpectrumAggregator:
    """Average the power spectrum over batch and time dimensions."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        nan_fill_fn: Callable[[torch.Tensor, str], torch.Tensor] = lambda x, _: x,
        report_plot: bool = True,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        Distributed.get_instance().require_no_spatial_parallelism(
            "SphericalPowerSpectrumAggregator does not support spatial parallelism."
        )
        self._real_sht = gridded_operations.get_real_sht()
        self._power_spectrum: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._nan_fill_fn = nan_fill_fn
        self._report_plot = report_plot
        self._variable_metadata: Mapping[str, VariableMetadata] = (
            variable_metadata or {}
        )

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData):
        self.record_data(data.prediction)

    @torch.no_grad()
    def record_data(self, prediction: TensorMapping):
        for name in prediction:
            batch_size = prediction[name].shape[0]
            time_size = prediction[name].shape[1]
            tensor = self._nan_fill_fn(prediction[name], name)
            power_spectrum = spherical_power_spectrum(tensor, self._real_sht)
            mean_power_spectrum = torch.mean(power_spectrum, dim=(0, 1))
            new_count = batch_size * time_size
            if name not in self._power_spectrum:
                self._power_spectrum[name] = mean_power_spectrum
            else:
                weighted_average = (
                    new_count * mean_power_spectrum
                    + self._counts[name] * self._power_spectrum[name]
                ) / (new_count + self._counts[name])
                self._power_spectrum[name] = weighted_average
            self._counts[name] += new_count

    def get_mean(self) -> dict[str, torch.Tensor]:
        dist = Distributed.get_instance()
        logs = {}
        sorted_names = sorted(list(self._power_spectrum))
        for name in sorted_names:
            _mean_spectrum = self._power_spectrum[name]
            if dist.world_size > 1:
                # assuming same count on all workers
                _mean_spectrum = dist.reduce_mean(_mean_spectrum)
            logs[name] = _mean_spectrum
        return logs

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, plt.Figure]:
        logs: dict[str, plt.Figure] = {}
        if self._report_plot:
            for name, spectrum in self.get_mean().items():
                fig = _plot_spectrum_pair(spectrum.cpu(), target=None)
                logs[f"{label}/{name}"] = fig
                plt.close(fig)
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        data_vars = {}
        for name, spectrum in self.get_mean().items():
            spectrum_np = spectrum.cpu().numpy()
            wavenumber = np.arange(len(spectrum_np))
            metadata = self._variable_metadata.get(name)
            if metadata is not None and metadata.long_name is not None:
                long_name = f"spherical power spectrum of {metadata.long_name}"
                units = f"({metadata.units or 'unknown_units'})^2"
            else:
                long_name = f"spherical power spectrum of {name}"
                units = "unknown_units"
            attrs = {"long_name": long_name, "units": units}
            data_vars[name] = xr.DataArray(
                spectrum_np,
                dims=["wavenumber"],
                coords={"wavenumber": wavenumber},
                attrs=attrs,
            )
        return xr.Dataset(data_vars)


class PairedSphericalPowerSpectrumAggregator:
    """Record batches and return plots for paired prediction and target data.

    Parameters:
        gridded_operations: lat/lon grid helpers used to build the SHT.
        report_plot: master plot toggle. When ``False``, no per-variable
            spectrum-pair PNGs are emitted regardless of ``plot_variables``.
        nan_fill_fn: per-variable NaN filler applied before the SHT.
        variable_metadata: optional CF metadata for spectrum-dataset attrs.
        report_directional_bias: when ``True`` (default), emit
            ``positive_norm_bias`` and ``negative_norm_bias`` scalar metrics
            per variable. When ``False``, only ``mean_abs_norm_bias`` and
            ``smallest_scale_norm_bias`` are emitted — useful when the
            directional split is redundant clutter in the W&B run.
        plot_variables: when ``None`` (default), plot spectrum-pair figures
            for every variable (current behaviour). When a list, restrict
            plotting to those variable names — scalar metrics are still
            emitted for every variable regardless. Lets callers keep the
            cheap scalar comparisons cohort-wide while limiting the
            expensive per-variable plot output to a small reference set
            (e.g. upper-level + h500 + precipitation).
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        report_plot: bool,
        nan_fill_fn: Callable[[torch.Tensor, str], torch.Tensor] = lambda x, _: x,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        report_directional_bias: bool = True,
        plot_variables: list[str] | None = None,
    ):
        self._gen_aggregator = SphericalPowerSpectrumAggregator(
            gridded_operations,
            nan_fill_fn,
            report_plot=False,
            variable_metadata=variable_metadata,
        )
        self._target_aggregator = SphericalPowerSpectrumAggregator(
            gridded_operations,
            nan_fill_fn,
            report_plot=False,
            variable_metadata=variable_metadata,
        )
        self._report_plot = report_plot
        self._report_directional_bias = report_directional_bias
        self._plot_variables: set[str] | None = (
            None if plot_variables is None else set(plot_variables)
        )

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData):
        target = data.target if data.has_target else None
        self.record_paired_data(prediction=data.prediction, target=target)

    @torch.no_grad()
    def record_paired_data(
        self,
        prediction: TensorMapping,
        target: TensorMapping | None,
    ):
        self._gen_aggregator.record_data(prediction)
        if target is not None:
            self._target_aggregator.record_data(target)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, plt.Figure | float]:
        logs: dict[str, plt.Figure | float] = {}
        gen_spectrum = self._gen_aggregator.get_mean()
        target_spectrum = self._target_aggregator.get_mean()
        if self._report_plot:
            for name in gen_spectrum:
                if (
                    self._plot_variables is not None
                    and name not in self._plot_variables
                ):
                    continue
                gen_spectrum_cpu = gen_spectrum[name].cpu()
                if name not in target_spectrum:
                    warnings.warn(f"Missing power spectrum target data for {name}")
                    target_spectrum_cpu = None
                else:
                    target_spectrum_cpu = target_spectrum[name].cpu()
                fig = _plot_spectrum_pair(gen_spectrum_cpu, target_spectrum_cpu)
                logs[f"{label}/{name}"] = fig
                plt.close(fig)
        metrics = _get_spectrum_metrics(
            gen_spectrum,
            target_spectrum,
            report_directional_bias=self._report_directional_bias,
        )
        for name, value in metrics.items():
            logs[f"{label}/{name}"] = value
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        gen_ds = self._gen_aggregator.get_dataset()
        target_ds = self._target_aggregator.get_dataset()
        if not gen_ds or not target_ds:
            return xr.Dataset()
        source = ["prediction", "target"]
        ds = xr.concat([gen_ds, target_ds], dim="source")
        return ds.assign_coords(source=source)


def _get_spectrum_metrics(
    gen_spectrum: dict[str, torch.Tensor],
    target_spectrum: dict[str, torch.Tensor],
    report_directional_bias: bool = True,
) -> dict[str, float]:
    """
    Compute metrics for the spectrum.

    Args:
        gen_spectrum: Dictionary of 1-dimensional generated mean power spectra.
        target_spectrum: Dictionary of 1-dimensional target mean power spectra.
        report_directional_bias: when True (default), emit per-variable
            ``positive_norm_bias`` and ``negative_norm_bias`` keys. When
            False, only ``mean_abs_norm_bias`` and ``smallest_scale_norm_bias``
            are emitted — the directional pair is redundant with
            ``mean_abs_norm_bias``.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    for name in gen_spectrum:
        if len(gen_spectrum[name].shape) != 1:
            raise ValueError(
                f"Expected 1-dimensional power spectrum for {name}, "
                f"got {gen_spectrum[name].shape}"
            )
        metrics[f"smallest_scale_norm_bias/{name}"] = get_smallest_scale_power_bias(
            gen_spectrum[name], target_spectrum[name]
        )
        positive_bias, negative_bias = get_positive_and_negative_power_bias(
            gen_spectrum[name], target_spectrum[name]
        )
        if report_directional_bias:
            metrics[f"positive_norm_bias/{name}"] = positive_bias
            metrics[f"negative_norm_bias/{name}"] = negative_bias
        metrics[f"mean_abs_norm_bias/{name}"] = abs(positive_bias) + abs(negative_bias)
    return metrics


def get_smallest_scale_power_bias(
    gen_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
) -> float:
    return float((gen_spectrum[-1] / target_spectrum[-1] - 1).mean().cpu())


def get_positive_and_negative_power_bias(
    gen_spectrum: torch.Tensor,
    target_spectrum: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute the positive and negative power bias for the spectrum,
    normalized by the target spectrum.
    """
    ratio = gen_spectrum / target_spectrum - 1
    positive_bias = ratio[ratio > 0].sum() / target_spectrum.shape[0]
    negative_bias = ratio[ratio < 0].sum() / target_spectrum.shape[0]
    return float(positive_bias.cpu()), float(negative_bias.cpu())


def _plot_spectrum_pair(
    prediction: torch.Tensor, target: torch.Tensor | None
) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(prediction, "--", label="prediction", color="C1")
    if target is not None:
        ax.plot(target, "-", label="target", color="C0")
    ax.set(yscale="log")
    ax.set(xlabel="wavenumber")
    ax.legend()
    plt.tight_layout()
    return fig


@dataclasses.dataclass
class PowerSpectrumMetricConfig:
    """
    Parameters:
        variables: when set, filter the aggregator to these variables only
            (affects every output — scalar metrics and the per-variable
            spectrum-pair plots).
        name: log prefix and wandb key prefix.
        enabled: master toggle for the metric.
        strict: raise if the metric can't be built (e.g. wrong grid).
        report_directional_bias: when False, drop the
            ``positive_norm_bias`` and ``negative_norm_bias`` scalar
            metrics. ``mean_abs_norm_bias`` is unaffected (and is the
            directional pair's redundant summary). Defaults to True for
            backwards compatibility.
        plot_variables: when set, restrict the per-variable spectrum-pair
            plot to these variable names — scalar metrics are still
            emitted for every variable that passed ``variables``. Use to
            keep the cheap scalar comparisons cohort-wide while limiting
            the expensive per-variable plot output to a small reference
            list. Defaults to None (plot everything that passed
            ``variables``, current behaviour).
    """

    variables: list[str] | None = None
    name: str = "power_spectrum"
    enabled: bool = True
    strict: bool = False
    report_directional_bias: bool = True
    plot_variables: list[str] | None = None

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        try:
            agg: SubAggregator = PairedSphericalPowerSpectrumAggregator(
                gridded_operations=ctx.ops,
                nan_fill_fn=SmoothFloodFill(num_steps=4),
                report_plot=True,
                variable_metadata=ctx.variable_metadata,
                report_directional_bias=self.report_directional_bias,
                plot_variables=self.plot_variables,
            )
        except NotImplementedError as e:
            raise MetricNotSupportedError(str(e)) from e
        return MetricBuildResult(aggregator=maybe_filter(agg, self.variables))
