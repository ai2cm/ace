"""Linearized shallow water model on the sphere.

Uses VectorDiscoConvS2 with fixed weights to approximate the spatial
differential operators (gradient, divergence) in the linearized shallow
water equations:

    ∂h'/∂t = -H₀ ∇·V
    ∂V/∂t  = -g ∇h' - f k×V

where h' is the perturbation depth (h - H₀), V = (u, v) is the velocity
vector in the local meridian frame, g is gravity, H₀ is the mean depth,
and f = 2Ω sin(lat) is the Coriolis parameter.
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2


class ShallowWaterStepper(nn.Module):
    """Linearized shallow water stepper on the sphere.

    Uses a VectorDiscoConvS2 convolution with frozen weights to compute
    the spatial tendencies, plus pointwise Coriolis forcing. Time
    integration uses RK4.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        g: float = 1.0,
        mean_depth: float = 1.0,
        omega: float = 0.5,
        kernel_shape: int = 5,
        theta_cutoff: float | None = None,
    ):
        """
        Args:
            shape: (nlat, nlon) grid dimensions.
            g: gravity (nondimensional, default 1.0).
            mean_depth: mean fluid depth H₀ (nondimensional, default 1.0).
            omega: rotation rate (nondimensional, default 0.5 so f=sin(lat)).
            kernel_shape: radial filter kernel shape for VectorDiscoConvS2.
            theta_cutoff: filter support radius (radians). Defaults to
                2 grid spacings.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.g = g
        self.mean_depth = mean_depth

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        # Coriolis parameter f = 2Ω sin(lat)
        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats  # colatitude → latitude
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        # Shape (1, 1, nlat, 1) for broadcasting with (B, C, H, W)
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))

        # Quadrature weights for area integrals, shape (nlat,)
        # quad_weights integrate over cos(colat) d(colat),
        # multiply by 2π/nlon for longitude to get area element
        area = (quad_weights * 2.0 * math.pi / self.nlon).float()
        self.register_buffer("area_weights", area)

        # Convolution for spatial operators
        self.conv = VectorDiscoConvS2(
            in_channels_scalar=1,
            in_channels_vector=1,
            out_channels_scalar=1,
            out_channels_vector=1,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # Freeze all parameters
        for p in self.conv.parameters():
            p.requires_grad = False

        # Set weights for the linearized equations:
        #   dh/dt = -H₀ div(V)   →  W_vs divergence component
        #   dV/dt = -g grad(h)   →  W_sv gradient component
        with torch.no_grad():
            self.conv.W_ss.zero_()
            self.conv.W_vv.zero_()
            self.conv.W_sv.zero_()
            self.conv.W_vs.zero_()
            # Divergence: vs path component 0
            self.conv.W_vs[0, 0, :, 0] = -mean_depth
            # Gradient: sv path component 0
            self.conv.W_sv[0, 0, :, 0] = -g

    def compute_tendencies(
        self, h: torch.Tensor, uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute time tendencies from the linearized equations.

        Args:
            h: perturbation depth h', shape (B, 1, nlat, nlon).
            uv: velocity (u, v), shape (B, 1, nlat, nlon, 2).

        Returns:
            dh_dt: shape (B, 1, nlat, nlon)
            duv_dt: shape (B, 1, nlat, nlon, 2)
        """
        # Spatial tendencies from convolution
        dh_dt, duv_dt = self.conv(h, uv)

        # Coriolis: f × V = f * (-v, u)
        f = self.f_coriolis  # (1, 1, nlat, 1)
        u = uv[..., 0]  # (B, 1, nlat, nlon)
        v = uv[..., 1]
        coriolis = torch.stack([-f * v, f * u], dim=-1)
        duv_dt = duv_dt + coriolis

        return dh_dt, duv_dt

    def step(
        self, h: torch.Tensor, uv: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one time step using RK4.

        Args:
            h: perturbation depth, shape (B, 1, nlat, nlon).
            uv: velocity, shape (B, 1, nlat, nlon, 2).
            dt: time step in seconds.

        Returns:
            h_new, uv_new with the same shapes.
        """
        k1_h, k1_uv = self.compute_tendencies(h, uv)
        k2_h, k2_uv = self.compute_tendencies(
            h + 0.5 * dt * k1_h, uv + 0.5 * dt * k1_uv
        )
        k3_h, k3_uv = self.compute_tendencies(
            h + 0.5 * dt * k2_h, uv + 0.5 * dt * k2_uv
        )
        k4_h, k4_uv = self.compute_tendencies(h + dt * k3_h, uv + dt * k3_uv)

        h_new = h + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)
        uv_new = uv + (dt / 6.0) * (k1_uv + 2 * k2_uv + 2 * k3_uv + k4_uv)
        return h_new, uv_new

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate a scalar field over the sphere.

        Args:
            field: shape (..., nlat, nlon)

        Returns:
            Integral value, shape (...)
        """
        # area_weights has shape (nlat,), sum over lat and lon
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))

    def total_energy(self, h: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Compute total energy E = ½g h'² + ½H₀(u² + v²).

        Args:
            h: perturbation depth, shape (B, 1, nlat, nlon).
            uv: velocity, shape (B, 1, nlat, nlon, 2).

        Returns:
            Total energy, shape (B,)
        """
        pe = 0.5 * self.g * h[:, 0] ** 2
        ke = 0.5 * self.mean_depth * (uv[:, 0, ..., 0] ** 2 + uv[:, 0, ..., 1] ** 2)
        return self.integrate_area(pe + ke)

    def total_mass(self, h: torch.Tensor) -> torch.Tensor:
        """Compute total mass ∫ h' dA.

        Args:
            h: perturbation depth, shape (B, 1, nlat, nlon).

        Returns:
            Total mass, shape (B,)
        """
        return self.integrate_area(h[:, 0])
