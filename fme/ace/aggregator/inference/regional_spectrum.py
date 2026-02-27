import logging
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import xarray as xr

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping

from .spectrum import _get_spectrum_metrics, _plot_spectrum_pair

LAT_BOUNDS = (-40, 35)
LON_BOUNDS = (180, 243)


def _detrend_linear(data):
    """
    Removes a linear plane of best fit from 4D (B, C, H, W) data.
    """
    B, C, H, W = data.shape
    device = data.device
    dtype = data.dtype

    y_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    A = torch.stack([X.flatten(), Y.flatten(), torch.ones_like(X).flatten()], dim=1)

    B_prime = B * C
    data_flat = data.reshape(B_prime, H * W)

    # Solve A * coeffs = data_flat.T
    # A is (H*W, 3), data_flat.T is (H*W, B_prime)
    # coeffs will be shape (3, B_prime)
    coeffs, _, _, _ = torch.linalg.lstsq(A, data_flat.T)

    # Reconstruct the plane
    # (H*W, 3) @ (B_prime, 3, 1) -> (B_prime, H*W, 1) -> (B_prime, H, W)
    plane = (A @ coeffs.permute(1, 0).unsqueeze(-1)).reshape(B_prime, H, W)

    detrended_data = data.reshape(B_prime, H, W) - plane

    return detrended_data.reshape(B, C, H, W)


def compute_isotropic_spectrum(
    data,
    dx=1.0,
    dy=1.0,
    num_bins=None,
    n_factor=4,
    remove_mean=True,
    detrend=None,
    window="Hann",
    truncate=True,
    cutoff_before_bins: bool = True,
    weights=None,
):
    """
    Compute the isotropic 1D power spectrum from 2D, 3D, or 4D data.

    Matches `xrft.isotropic_power_spectrum(scaling="density")`.
    The output spectrum is computed for each batch and channel element.

    Args:
        data: Input data tensor with shape (H, W), (B, H, W), or (B, C, H, W).
        dx: Grid spacing in the x-dimension.
        dy: Grid spacing in the y-dimension.
        num_bins: Number of bins. If None, defaults to min(H, W) // n_factor.
        n_factor: Factor to determine number of bins when num_bins is None.
        remove_mean: If True, removes spatial mean. Overridden by `detrend`.
        detrend: Detrending method, either 'linear' or 'constant'.
        window: Window function to apply, currently only 'hann' is supported.
        truncate: If True, truncates spectrum at the smallest Nyquist frequency.
        cutoff_before_bins: If True, truncates the spectrum after computing
            the bin locations. Matches xrft implementation.
        weights: Regional weights for masking with shape (H, W). Will be
            broadcast to match data shape. Regions with weight=0 will not
            contribute to spectrum.

    Returns:
        A tuple of (k_bins_centers, iso_spectrum) where:
        - k_bins_centers: 1D tensor of bin center wavenumbers with shape (num_bins,).
        - iso_spectrum: The (k * P(k)) spectrum with shape matching input
          dimensionality: (num_bins,), (B, num_bins), or (B, C, num_bins).
    """
    # --- 1. Input Validation and Setup ---
    device = data.device
    dtype = data.dtype
    orig_dim = data.dim()

    # Unify input shape to 4D (B, C, H, W)
    if orig_dim == 2:
        data = data.reshape(1, 1, *data.shape)  # (H, W) -> (1, 1, H, W)
    elif orig_dim == 3:
        data = data.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    elif orig_dim != 4:
        raise ValueError("Input data must be 2D, 3D, or 4D (B, C, H, W)")

    B, C, H, W = data.shape
    B_prime = B * C
    Lx = W * dx
    Ly = H * dy

    if num_bins is None:
        num_bins = min(H, W) // n_factor

    # Apply regional weights if provided
    if weights is not None:
        # Ensure weights are on the same device and have the right shape
        if weights.shape != (H, W):
            raise ValueError(
                f"Weights shape {weights.shape} "
                f"does not match data spatial shape ({H}, {W})"
            )
        weights = weights.to(device=device, dtype=dtype)
        # Broadcast weights to match data shape (B, C, H, W)
        weights_broadcast = weights.unsqueeze(0).unsqueeze(0)
        data = data * weights_broadcast

    if detrend == "linear":
        data = _detrend_linear(data)
    elif detrend == "constant" or remove_mean:
        data = data - torch.mean(data, dim=(-2, -1), keepdim=True)

    if window and window.lower() == "hann":
        win_y = torch.hann_window(H, device=device, dtype=dtype).unsqueeze(1)
        win_x = torch.hann_window(W, device=device, dtype=dtype).unsqueeze(0)
        win_2d = (win_y * win_x).reshape(1, 1, H, W)
        window_correction = torch.mean(win_2d**2).item()
        data = data * win_2d
    else:
        window_correction = 1.0

    fft_2d = torch.fft.rfft2(data, norm="forward")
    power_2d = torch.abs(fft_2d) ** 2
    power_2d = power_2d / window_correction
    psd_2d = power_2d * (Lx * Ly)

    k_x = torch.fft.rfftfreq(W, d=dx, device=device, dtype=dtype)
    k_y = torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)
    k_x_nyq = 1.0 / (2.0 * dx)
    k_y_nyq = 1.0 / (2.0 * dy)
    k_Y, k_X = torch.meshgrid(k_y, k_x, indexing="ij")
    k_mag = torch.sqrt(k_X**2 + k_Y**2)
    k_max_domain = k_mag.max()

    if truncate and cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
    else:
        k_max = k_max_domain

    k_bins = torch.linspace(0, k_max, num_bins + 1, device=device, dtype=dtype)

    if truncate and not cutoff_before_bins:
        k_max_cutoff = min(k_x_nyq, k_y_nyq)
        k_max = min(k_max_domain, k_max_cutoff)
        k_bins = k_bins[k_bins < k_max_cutoff]
        num_bins = k_bins.numel() - 1

    k_bins_centers = (k_bins[:-1] + k_bins[1:]) / 2
    k_mag_flat = k_mag.flatten()
    bin_edges = k_bins[1:-1]
    bin_indices = torch.bucketize(k_mag_flat, bin_edges, right=True)

    N_flat = k_mag_flat.shape[0]
    psd_flat_batched = psd_2d.reshape(B_prime, N_flat)
    bin_indices_batched = bin_indices.expand(B_prime, -1)

    binned_psd_sum = torch.zeros(B_prime, num_bins, device=device, dtype=dtype)
    binned_psd_sum.scatter_add_(dim=1, index=bin_indices_batched, src=psd_flat_batched)

    binned_counts = torch.bincount(bin_indices, minlength=num_bins)
    binned_counts_safe = binned_counts.float()
    binned_counts_safe[binned_counts_safe == 0] = torch.nan

    iso_psd_binned = binned_psd_sum / binned_counts_safe.unsqueeze(0)
    iso_spectrum = iso_psd_binned * k_bins_centers.unsqueeze(0)
    iso_spectrum = iso_spectrum.reshape(B, C, num_bins)
    iso_spectrum[..., 0] = torch.nan

    if orig_dim == 2:
        iso_spectrum = iso_spectrum.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        iso_spectrum = iso_spectrum.squeeze(1)

    return k_bins_centers, iso_spectrum


class RegionalSpectrumAggregator:
    """Average the regional power spectrum over batch and time dimensions."""

    def __init__(
        self,
        regional_weights: torch.Tensor,
    ):
        self._regional_weights = regional_weights
        self._power_spectrum: dict[str, torch.Tensor] = {}
        self._wavenumbers: torch.Tensor | None = None
        self._counts: dict[str, int] = defaultdict(int)

    @torch.no_grad()
    def record_batch(self, data: TensorMapping):
        for name in data:
            # Expecting data of shape (batch, time, lat, lon)
            batch_size = data[name].shape[0]
            time_size = data[name].shape[1]

            # Reshape to (batch*time, lat, lon) for spectrum computation
            data_reshaped = data[name].reshape(
                batch_size * time_size, data[name].shape[2], data[name].shape[3]
            )

            # Compute spectrum with regional weights
            wavenumbers, power_spectrum = compute_isotropic_spectrum(
                data_reshaped, weights=self._regional_weights
            )

            # Store wavenumbers (same for all variables)
            if self._wavenumbers is None:
                self._wavenumbers = wavenumbers

            # Average over batch*time dimension
            mean_power_spectrum = torch.mean(power_spectrum, dim=0)

            new_count = batch_size * time_size
            if name not in self._power_spectrum:
                self._power_spectrum[name] = mean_power_spectrum
            else:
                # Weighted average with previous values
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


class PairedRegionalSpectrumAggregator:
    """Record batches and return plots for paired prediction and target data."""

    def __init__(
        self,
        regional_weights: torch.Tensor,
        report_plot: bool,
    ):
        self._gen_aggregator = RegionalSpectrumAggregator(regional_weights)
        self._target_aggregator = RegionalSpectrumAggregator(regional_weights)
        self._report_plot = report_plot

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping | None = None,
        gen_data_norm: TensorMapping | None = None,
        i_time_start: int = 0,
    ):
        self._gen_aggregator.record_batch(gen_data)
        self._target_aggregator.record_batch(target_data)

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, plt.Figure | float]:
        logs: dict[str, plt.Figure | float] = {}
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
            "get_dataset not implemented for PairedRegionalSpectrumAggregator. "
            "Returning an empty dataset."
        )
        return xr.Dataset()
