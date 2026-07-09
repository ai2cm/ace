"""Spectral matching loss for distillation.

A band-weighted, per-variable loss on the difference between the zonal power
spectra of two fields. Intended as an auxiliary generator loss in distillation:
matching the student's spectrum to the teacher's directly restores the
high-wavenumber energy that few-step sampling blurs, without the freedom a GAN
has to invent spatial texture (the spectrum constrains amplitude per wavenumber
only; spatial placement is pinned by the distribution-matching term).

See ``fme/downscaling/distillation/specs/11-spectral-matching-loss.md``.
"""

import dataclasses

import torch

from fme.downscaling.metrics_and_maths import compute_zonal_power_spectrum


@dataclasses.dataclass
class SpectralMatchingLossConfig:
    """Configuration for the spectral matching loss.

    The overall scale of the term is *not* set here; it is applied by the caller
    (the distillation auxiliary-loss weight) so there is a single overall knob.
    This config controls only the *shape* of the loss.

    Parameters:
        band_gamma: High-wavenumber emphasis exponent. Band weights are
            proportional to ``(k / k_max) ** band_gamma`` over wavenumber ``k``,
            then normalized to mean 1 over the included wavenumbers. ``0.0``
            weights all wavenumbers equally; larger values concentrate the loss
            on the small-scale tail where the student degrades.
        min_wavenumber: Wavenumbers below this index are excluded (weight 0), so
            the term does not spend its budget on the large scales the
            distribution-matching loss already reproduces.
        variable_weights: Per-variable multiplicative weights keyed by output
            variable name. Variables not listed default to 1.0.
        eps: Additive constant inside the log for numerical stability.
        log: If True, take the L1 difference of log-power; otherwise of raw
            power.
    """

    band_gamma: float = 0.0
    min_wavenumber: int = 0
    variable_weights: dict[str, float] = dataclasses.field(default_factory=dict)
    eps: float = 1e-12
    log: bool = True

    def __post_init__(self):
        if self.band_gamma < 0.0:
            raise ValueError(f"band_gamma must be >= 0, got {self.band_gamma}")
        if self.min_wavenumber < 0:
            raise ValueError(f"min_wavenumber must be >= 0, got {self.min_wavenumber}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def build(self, out_names: list[str]) -> "SpectralMatchingLoss":
        unknown = set(self.variable_weights) - set(out_names)
        if unknown:
            raise ValueError(
                f"variable_weights has keys not in out_names {out_names}: "
                f"{sorted(unknown)}"
            )
        variable_weights = torch.tensor(
            [self.variable_weights.get(name, 1.0) for name in out_names],
            dtype=torch.float32,
        )
        return SpectralMatchingLoss(
            band_gamma=self.band_gamma,
            min_wavenumber=self.min_wavenumber,
            variable_weights=variable_weights,
            eps=self.eps,
            log=self.log,
        )


class SpectralMatchingLoss(torch.nn.Module):
    """Band-weighted, per-variable L1 loss between zonal power spectra.

    ``forward(prediction, target)`` returns a scalar. The target is detached, so
    gradient flows only into ``prediction``. Both tensors must share the layout
    ``(..., channel, latitude, longitude)`` with the channel axis at
    ``CHANNEL_DIM`` and channels ordered to match ``out_names``.
    """

    def __init__(
        self,
        band_gamma: float,
        min_wavenumber: int,
        variable_weights: torch.Tensor,
        eps: float,
        log: bool,
    ):
        super().__init__()
        self.band_gamma = band_gamma
        self.min_wavenumber = min_wavenumber
        self.eps = eps
        self.log = log
        # (C,) reshaped to broadcast against (..., C, K). Non-persistent: it is
        # derived from config, so it should not enter checkpoints.
        self.register_buffer(
            "variable_weights", variable_weights.reshape(-1, 1), persistent=False
        )

    def _band_weights(self, n_wavenumbers: int, device, dtype) -> torch.Tensor:
        k = torch.arange(n_wavenumbers, device=device, dtype=dtype)
        k_max = max(n_wavenumbers - 1, 1)
        weights = (k / k_max) ** self.band_gamma
        included = k >= self.min_wavenumber
        weights = torch.where(included, weights, torch.zeros_like(weights))
        n_included = int(included.sum())
        if n_included == 0:
            raise ValueError(
                f"min_wavenumber={self.min_wavenumber} excludes all "
                f"{n_wavenumbers} wavenumbers"
            )
        # Normalize to mean 1 over included wavenumbers so the overall scale is
        # comparable across band_gamma / min_wavenumber settings.
        weights = weights / (weights.sum() / n_included)
        return weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Spectrum-then-average: compute each sample's power spectrum, then
        # average the spectra over the batch to estimate the *ensemble* power
        # E[|FFT|^2]. This preserves the incoherent high-wavenumber energy that
        # averaging the fields first (a conditional mean) would cancel, and it
        # matches how the eval spec_mae aggregator computes spectra. Prediction
        # and target are independent samples sharing conditioning, so the match
        # is distributional (mean spectrum vs mean spectrum), not pointwise.
        power_pred = compute_zonal_power_spectrum(prediction)
        power_target = compute_zonal_power_spectrum(target).detach()

        # Average over all leading (sample/batch) dims -> (C, K).
        lead = tuple(range(power_pred.ndim - 2))
        if lead:
            power_pred = power_pred.mean(dim=lead)
            power_target = power_target.mean(dim=lead)

        if self.log:
            diff = (
                torch.log(power_pred + self.eps) - torch.log(power_target + self.eps)
            ).abs()
        else:
            diff = (power_pred - power_target).abs()

        # diff: (C, K)
        band_weights = self._band_weights(diff.shape[-1], diff.device, diff.dtype)
        var_weights = self.variable_weights.to(dtype=diff.dtype)
        weighted = diff * band_weights * var_weights
        return weighted.mean()
