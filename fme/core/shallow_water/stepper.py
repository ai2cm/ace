"""Linearized shallow water model on the sphere.

Uses a VectorDiscoBlock with fixed weights to approximate the spatial
differential operators (gradient, divergence) and Coriolis forcing in
the linearized shallow water equations:

    dh'/dt = -H0 div(V)
    dV/dt  = -g grad(h') - f x V

where h' is the perturbation depth (h - H0), V = (u, v) is the velocity
vector in the local meridian frame, g is gravity, H0 is the mean depth,
and f = 2*omega*sin(lat) is the Coriolis parameter.
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.shallow_water.block import VectorDiscoBlock


class ShallowWaterStepper(nn.Module):
    """Linearized shallow water stepper on the sphere.

    Uses a single VectorDiscoBlock with frozen weights. The convolution
    computes gradient and divergence, and the ScalarVectorProduct handles
    Coriolis forcing. Time integration uses RK4.

    The Coriolis parameter f(lat) is provided as a second scalar channel.
    The block's convolution produces tendencies from h (channel 0) and V,
    while the ScalarVectorProduct uses f (channel 1) to rotate V.
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
            mean_depth: mean fluid depth H0 (nondimensional, default 1.0).
            omega: rotation rate (nondimensional, default 0.5 so f=sin(lat)).
            kernel_shape: radial filter kernel shape.
            theta_cutoff: filter support radius (radians). Defaults to
                2 grid spacings.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.g = g
        self.mean_depth = mean_depth

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        # Coriolis parameter f = 2*omega*sin(lat), shape (1, 1, nlat, 1)
        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))

        # Quadrature weights for area integrals, shape (nlat,)
        area = (quad_weights * 2.0 * math.pi / self.nlon).float()
        self.register_buffer("area_weights", area)

        # Block: 2 scalar channels (h, f), 1 vector channel (V)
        # activation="none" preserves sign of tendencies for conservation
        self.block = VectorDiscoBlock(
            n_scalar=2,
            n_vector=1,
            shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            activation="none",
        )

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

        # Set convolution weights for the linearized equations:
        #   dh/dt = -H0 div(V)  → W_vs[0, 0, :, 1] (divergence, channel 0)
        #   dV/dt = -g grad(h)  → W_sv[0, 0, :, 1] (gradient from channel 0)
        # f channel (channel 1) should not contribute to convolution output.
        with torch.no_grad():
            self.block.conv.W_ss.zero_()
            # Pass through f via isotropic smoothing so sv_product sees it
            self.block.conv.W_ss[1, 1, 0] = 1.0
            self.block.conv.W_vv.zero_()
            self.block.conv.W_sv.zero_()
            self.block.conv.W_vs.zero_()
            self.block.conv.W_vs[0, 0, :, 1] = mean_depth
            self.block.conv.W_sv[0, 0, :, 1] = g
            if self.block.conv.bias_scalar is not None:
                self.block.conv.bias_scalar.zero_()

            # Zero out scalar MLP (residual MLP, so zero = identity)
            for module in self.block.scalar_mlp:
                if hasattr(module, "weight"):
                    module.weight.zero_()
                if hasattr(module, "bias"):
                    module.bias.zero_()

            # ScalarVectorProduct: f (channel 1) rotates V (Coriolis)
            assert self.block.sv_product is not None
            self.block.sv_product.weight.zero_()
            self.block.sv_product.weight[1, 0, 1] = 1.0

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
        B = h.shape[0]
        # Build 2-channel scalar input: [h, f_coriolis]
        f = self.f_coriolis.expand(B, 1, self.nlat, self.nlon)
        x_scalar = torch.cat([h, f], dim=1)  # (B, 2, nlat, nlon)

        # Block computes conv + activation + MLP + ScalarVectorProduct
        # and adds residual. We subtract the residual to get the tendency.
        y_scalar, y_vector = self.block(x_scalar, uv)
        dh_dt = y_scalar[:, 0:1] - h  # remove residual for h channel
        duv_dt = y_vector - uv  # remove residual for V

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
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))

    def total_energy(self, h: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Compute total energy E = 1/2 g h'^2 + 1/2 H0 (u^2 + v^2).

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
        """Compute total mass: integral of h' over sphere.

        Args:
            h: perturbation depth, shape (B, 1, nlat, nlon).

        Returns:
            Total mass, shape (B,)
        """
        return self.integrate_area(h[:, 0])
