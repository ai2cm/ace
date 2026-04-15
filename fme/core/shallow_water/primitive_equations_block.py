"""Multi-level isobaric primitive equations via ``HybridCoordinateBlockStepper``.

This is a thin wrapper that converts isobaric pressure levels to hybrid
coefficients (b=0) and delegates to ``HybridCoordinateBlockStepper`` with
``isobaric=True``. The isobaric flag disables vertical advection of V and q,
PGF correction, and surface pressure evolution — matching the simplified
physics of ``PrimitiveEquationsStepper``.
"""

import math

import torch
import torch.nn as nn

from fme.core.shallow_water.hybrid_equations_block import (
    HybridCoordinateBlockStepper,
    isobaric_coefficients,
)


class PrimitiveEquationsBlockStepper(nn.Module):
    """Multi-level isobaric primitive equations using a VectorDiscoBlock.

    Composes a ``HybridCoordinateBlockStepper`` with ``isobaric=True``
    and wraps its interface to hide the constant surface pressure.

    State: ``(uv, T, q)`` — same as ``PrimitiveEquationsStepper``.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        n_levels: int = 3,
        pressure_levels: list[float] | None = None,
        R: float = 287.0,
        omega: float = 7.292e-5,
        phi_surface: float = 0.0,
        kernel_shape: int = 5,
        theta_cutoff: float | None = None,
        diffusion_coeff: float | None = None,
    ):
        super().__init__()
        K = n_levels
        if pressure_levels is None:
            pressure_levels = [
                100000.0 * math.exp(-k * math.log(10.0) / max(1, K - 1))
                for k in range(K)
            ]
        a_mid, b_mid, a_int, b_int = isobaric_coefficients(pressure_levels)
        self.stepper = HybridCoordinateBlockStepper(
            shape=shape,
            a_mid=a_mid,
            b_mid=b_mid,
            a_interface=a_int,
            b_interface=b_int,
            R=R,
            omega=omega,
            phi_surface=phi_surface,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            diffusion_coeff=diffusion_coeff,
            isobaric=True,
        )
        self._p_surface = a_int[0]

    def _constant_ps(self, uv: torch.Tensor) -> torch.Tensor:
        return uv.new_full(
            (uv.shape[0], self.stepper.nlat, self.stepper.nlon), self._p_surface
        )

    def compute_tendencies(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        duv, dT, dq, _ = self.stepper.compute_tendencies(
            uv, T, q, self._constant_ps(uv)
        )
        return duv, dT, dq

    def step(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_s = self._constant_ps(uv)
        uv2, T2, q2, _ = self.stepper.step(uv, T, q, p_s, dt)
        return uv2, T2, q2

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        return self.stepper.integrate_area(field)

    def total_kinetic_energy(self, uv: torch.Tensor) -> torch.Tensor:
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)
        return self.integrate_area(ke).sum(dim=1)
