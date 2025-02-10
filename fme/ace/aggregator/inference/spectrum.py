import logging
import warnings
from collections import defaultdict
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch_harmonics
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.metrics import spherical_power_spectrum
from fme.core.typing_ import TensorMapping


class SphericalPowerSpectrumAggregator:
    """Average the power spectrum over batch and time dimensions."""

    def __init__(self, nlat: int, nlon: int, grid: str = "legendre-gauss"):
        self._real_sht = torch_harmonics.RealSHT(nlat, nlon, grid=grid)
        self._power_spectrum: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = defaultdict(int)

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

    def get_mean(self) -> Dict[str, torch.Tensor]:
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

    def __init__(self, nlat: int, nlon: int, grid: str = "legendre-gauss"):
        self._gen_aggregator = SphericalPowerSpectrumAggregator(nlat, nlon, grid)
        self._target_aggregator = SphericalPowerSpectrumAggregator(nlat, nlon, grid)

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
    def get_logs(self, label: str) -> Dict[str, plt.Figure]:
        logs = {}
        gen_spectrum = self._gen_aggregator.get_mean()
        target_spectrum = self._target_aggregator.get_mean()

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
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        logging.debug(
            "get_dataset not implemented for PairedSphericalPowerSpectrumAggregator. "
            "Returning an empty dataset."
        )
        return xr.Dataset()


def _plot_spectrum_pair(
    prediction: torch.tensor, target: Optional[torch.tensor]
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
