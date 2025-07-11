import logging
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.metrics import spherical_power_spectrum
from fme.core.typing_ import TensorMapping


class SphericalPowerSpectrumAggregator:
    """Average the power spectrum over batch and time dimensions."""

    def __init__(self, gridded_operations: GriddedOperations):
        self._real_sht = gridded_operations.get_real_sht()
        self._power_spectrum: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = defaultdict(int)

    @torch.no_grad()
    def record_batch(self, data: TensorMapping):
        for name in data:
            batch_size = data[name].shape[0]
            time_size = data[name].shape[1]
            power_spectrum = spherical_power_spectrum(data[name], self._real_sht)
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


class PairedSphericalPowerSpectrumAggregator:
    """Record batches and return plots for paired prediction and target data."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        report_plot: bool,
    ):
        self._gen_aggregator = SphericalPowerSpectrumAggregator(gridded_operations)
        self._target_aggregator = SphericalPowerSpectrumAggregator(gridded_operations)
        self._report_plot = report_plot

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        self._gen_aggregator.record_batch(gen_data)
        self._target_aggregator.record_batch(target_data)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, plt.Figure]:
        logs = {}
        gen_spectrum = self._gen_aggregator.get_mean()
        target_spectrum = self._target_aggregator.get_mean()
        if self._report_plot:
            for name in gen_spectrum:
                gen_spectrum_cpu = gen_spectrum[name].cpu()
                if name not in target_spectrum:
                    warnings.warn(f"Missing power spectrum target data for {name}")
                    target_spectrum_cpu = None
                else:
                    target_spectrum_cpu = target_spectrum[name].cpu()
                fig = _plot_spectrum_pair(gen_spectrum_cpu, target_spectrum_cpu)
                logs[f"{label}/{name}"] = fig
                plt.close(fig)
        metrics = _get_spectrum_metrics(gen_spectrum, target_spectrum)
        for name, value in metrics.items():
            logs[f"{label}/{name}"] = value
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        logging.debug(
            "get_dataset not implemented for PairedSphericalPowerSpectrumAggregator. "
            "Returning an empty dataset."
        )
        return xr.Dataset()


def _get_spectrum_metrics(
    gen_spectrum: dict[str, torch.Tensor],
    target_spectrum: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compute metrics for the spectrum.

    Args:
        gen_spectrum: Dictionary of 1-dimensional generated mean power spectra.
        target_spectrum: Dictionary of 1-dimensional target mean power spectra.

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
