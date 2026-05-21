"""Idealized clear-sky model for CO2-induced temperature change.

This is an experimental knob used by the network normalizer to subtract a
physically-motivated greenhouse forcing signal from temperature fields
before normalization (and add it back on denormalization). The model
itself is independent of the normalizer: it just maps a CO2 mixing ratio
and a pressure level to a temperature change.
"""

import dataclasses
import math

import torch


@dataclasses.dataclass
class CO2TemperatureProfileConfig:
    """Per-doubling ΔT(p) profile for CO2-induced temperature change.

    Total response is the per-doubling profile scaled by
    ``log2(co2 / co2_reference_vmr)``. The per-doubling profile is:

      - ``delta_t_surface_per_doubling`` for ``p ≥ tropopause_pressure_pa``
      - log-linearly interpolated for the stratosphere
      - ``delta_t_stratosphere_per_doubling`` for ``p ≤ stratosphere_top_pressure_pa``

    Parameters:
        co2_reference_vmr: Reference CO2 volume mixing ratio (mol/mol). The
            data's ``global_mean_co2`` is stored as a VMR (e.g. 280 ppmv =
            ``2.8e-4``), so keep this in the same units.
        delta_t_surface_per_doubling: K per CO2 doubling in the troposphere.
        delta_t_stratosphere_per_doubling: K per CO2 doubling at and above
            ``stratosphere_top_pressure_pa`` (typically negative).
        tropopause_pressure_pa: Pressure dividing tropospheric (uniform
            warming) from stratospheric (log-linearly transitioning) regions.
        stratosphere_top_pressure_pa: Pressure at and above which the
            stratospheric cooling magnitude saturates.
    """

    co2_reference_vmr: float = 280e-6
    delta_t_surface_per_doubling: float = 3.0
    delta_t_stratosphere_per_doubling: float = -10.0
    tropopause_pressure_pa: float = 2.0e4
    stratosphere_top_pressure_pa: float = 1.0e2

    def __post_init__(self):
        if self.co2_reference_vmr <= 0:
            raise ValueError("co2_reference_vmr must be positive")
        if self.tropopause_pressure_pa <= self.stratosphere_top_pressure_pa:
            raise ValueError(
                "tropopause_pressure_pa must be greater than "
                "stratosphere_top_pressure_pa"
            )

    def per_doubling_at(self, pressure_pa: float) -> float:
        """Per-doubling ΔT (K) at a single pressure level."""
        if pressure_pa >= self.tropopause_pressure_pa:
            return self.delta_t_surface_per_doubling
        if pressure_pa <= self.stratosphere_top_pressure_pa:
            return self.delta_t_stratosphere_per_doubling
        log_p = math.log(pressure_pa)
        log_pt = math.log(self.tropopause_pressure_pa)
        log_ptop = math.log(self.stratosphere_top_pressure_pa)
        frac = (log_pt - log_p) / (log_pt - log_ptop)
        return (
            self.delta_t_surface_per_doubling
            + (
                self.delta_t_stratosphere_per_doubling
                - self.delta_t_surface_per_doubling
            )
            * frac
        )

    def delta_t(self, co2_vmr: torch.Tensor, pressure_pa: float) -> torch.Tensor:
        """ΔT at a single pressure level for a batch of CO2 values.

        Args:
            co2_vmr: CO2 volume mixing ratio. Any shape; output has the same
                shape.
            pressure_pa: Pressure level in Pa.

        Returns:
            ΔT in K, same shape as ``co2_vmr``.
        """
        log2_ratio = torch.log2(co2_vmr / self.co2_reference_vmr)
        return log2_ratio * self.per_doubling_at(pressure_pa)
