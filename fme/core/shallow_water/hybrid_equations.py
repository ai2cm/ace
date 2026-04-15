"""Multi-level hydrostatic primitive equations in hybrid sigma-pressure coordinates.

The pressure at each model level is

    p_k(λ, φ, t)  =  a_k  +  b_k · p_s(λ, φ, t)

where ``a_k`` (Pa) and ``b_k`` (dimensionless) are prescribed level-dependent
constants, and ``p_s`` is the prognostic surface pressure.

Boundary conditions on the interface coefficients:
    top (k = 0 half-level):     a_interface[0] ≥ 0,  b_interface[0] = 0
    surface (k = K half-level): a_interface[K] = 0,  b_interface[K] = 1

Special cases
-------------
* **Pure sigma**:  ``a_k = 0``, ``b_k = σ_k``  →  ``p_k = σ_k p_s``.
* **Pure pressure**: ``a_k = p_k`` (const), ``b_k = 0``
  →  ``p_k`` independent of ``p_s``.

Equations
---------
Layer thickness::

    Δp_k = Δa_k + Δb_k · p_s      (B, K, H, W — spatially varying)

Surface pressure tendency::

    ∂p_s/∂t = −∑_k (Δp_k div_k + Δb_k V_k·∇p_s)

Vertical-velocity diagnostic (C = (∂p/∂η)η̇ at interfaces)::

    C_{k+1/2} = C_{k-1/2}
                − Δb_k ∂p_s/∂t
                − Δp_k div_k
                − Δb_k V_k·∇p_s

    C_k = ½(C_{k-1/2} + C_{k+1/2})   [mid-level]

Hydrostatic geopotential::

    φ_0 = φ_surface
    φ_k = φ_{k-1} + R T_{k-1} ln(p_{k-1}/p_k)     (spatially varying)

Pressure gradient force::

    PGF_k = −∇φ_k  −  (R T_k b_k / p_k) ∇p_s

Pressure velocity::

    ω_k = b_k (∂p_s/∂t + V_k·∇p_s) + C_k

Momentum tendency (Lamb form)::

    ∂V/∂t = PGF − f×V − ∇KE + ζ×V − C_k ∂V/∂p

Temperature tendency::

    ∂T/∂t = −V·∇T + (κT/p)ω − C_k ∂T/∂p

Humidity::

    ∂q/∂t = −V·∇q − C_k ∂q/∂p

All vertical derivatives ``∂X/∂p`` use central finite differences in pressure
space (one-sided at the boundaries) so they are spatially varying fields.

The sigma coordinate is recovered exactly when ``a_k = 0``, ``b_k = σ_k``.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct

R_EARTH = 6.371e6  # m
C_P = 1004.0  # J/kg/K


class HybridCoordinateStepper(nn.Module):
    """Multi-level primitive equations in hybrid σ-p coordinates.

    State: ``(uv, T, q, p_s)``

    Parameters
    ----------
    shape:
        ``(nlat, nlon)`` grid dimensions.
    a_mid:
        Level mid-point ``a`` coefficients (Pa), length K, surface-to-top.
        ``a_mid[k] + b_mid[k] * p_s`` gives the mid-level pressure at level k.
    b_mid:
        Level mid-point ``b`` coefficients (dimensionless), length K.
    a_interface:
        Interface ``a`` coefficients (Pa), length K+1.
        ``a_interface[0]`` is the top; ``a_interface[K]`` is the surface.
    b_interface:
        Interface ``b`` coefficients (dimensionless), length K+1.
        Must satisfy ``b_interface[0] = 0`` (top) and ``b_interface[K] = 1``
        (surface).
    R:
        Gas constant for dry air (J/kg/K).
    omega:
        Earth rotation rate (rad/s).
    phi_surface:
        Surface geopotential φ_s (m²/s²).
    kernel_shape:
        DISCO kernel shape for differential operators.
    theta_cutoff:
        Filter radius (rad). Defaults to 2 grid spacings.
    diffusion_coeff:
        ν for ∇^(2n) diffusion (m²ⁿ/s). ``None`` = off.
    diffusion_order:
        Diffusion order n; 1 → ∇², 2 → ∇⁴.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        a_mid: list[float],
        b_mid: list[float],
        a_interface: list[float],
        b_interface: list[float],
        R: float = 287.0,
        omega: float = 7.292e-5,
        phi_surface: float = 0.0,
        kernel_shape: int = 5,
        theta_cutoff: float | None = None,
        diffusion_coeff: float | None = None,
        diffusion_order: int = 1,
    ):
        super().__init__()
        self.nlat, self.nlon = shape
        K = len(a_mid)
        if len(b_mid) != K:
            raise ValueError(f"len(b_mid)={len(b_mid)} != len(a_mid)={K}")
        if len(a_interface) != K + 1:
            raise ValueError(
                f"len(a_interface)={len(a_interface)} must be len(a_mid)+1={K+1}"
            )
        if len(b_interface) != K + 1:
            raise ValueError(
                f"len(b_interface)={len(b_interface)} must be len(a_mid)+1={K+1}"
            )
        self.n_levels = K
        self.R = R
        self.phi_surface = phi_surface
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_order = diffusion_order

        a_m = torch.tensor(a_mid, dtype=torch.float32)
        b_m = torch.tensor(b_mid, dtype=torch.float32)
        a_i = torch.tensor(a_interface, dtype=torch.float32)
        b_i = torch.tensor(b_interface, dtype=torch.float32)

        self.register_buffer("a_mid", a_m)  # (K,)
        self.register_buffer("b_mid", b_m)  # (K,)
        self.register_buffer("a_interface", a_i)  # (K+1,)
        self.register_buffer("b_interface", b_i)  # (K+1,)

        # Layer thicknesses in a and b: Δa_k = a_{k+1/2} − a_{k-1/2}
        # With index 0 = surface interface, K = top interface
        delta_a = a_i[1:] - a_i[:-1]  # (K,)  (negative: pressure decreases upward)
        delta_b = b_i[1:] - b_i[:-1]  # (K,)  (negative)
        self.register_buffer("delta_a", delta_a)
        self.register_buffer("delta_b", delta_b)

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))

        area = (quad_weights * 2.0 * math.pi / self.nlon).float()
        self.register_buffer("area_weights", area)

        disco_kw: dict[str, Any] = dict(
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )
        self.grad_conv = VectorDiscoConvS2(
            in_channels_scalar=1,
            in_channels_vector=0,
            out_channels_scalar=0,
            out_channels_vector=1,
            **disco_kw,
        )
        self.vort_conv = VectorDiscoConvS2(
            in_channels_scalar=0,
            in_channels_vector=1,
            out_channels_scalar=1,
            out_channels_vector=0,
            **disco_kw,
        )
        self.div_conv = VectorDiscoConvS2(
            in_channels_scalar=0,
            in_channels_vector=1,
            out_channels_scalar=1,
            out_channels_vector=0,
            **disco_kw,
        )
        self.coriolis_product = ScalarVectorProduct(n_scalar=1, n_vector=1)
        self.vort_product = ScalarVectorProduct(n_scalar=1, n_vector=1)

        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self.grad_conv.W_sv.zero_()
            self.grad_conv.W_sv[0, 0, :, 1] = 1.0 / R_EARTH

            self.vort_conv.W_vs.zero_()
            self.vort_conv.W_vs[0, 0, :, 0] = 1.0 / R_EARTH

            self.div_conv.W_vs.zero_()
            self.div_conv.W_vs[0, 0, :, 1] = 1.0 / R_EARTH

            self.coriolis_product.weight.zero_()
            self.coriolis_product.weight[0, 0, 1] = 1.0

            self.vort_product.weight.zero_()
            self.vort_product.weight[0, 0, 1] = 1.0

    # ── Differential operators ─────────────────────────────────────────────

    def _gradient(self, s: torch.Tensor) -> torch.Tensor:
        """∇s for (B, K, H, W) → (B, K, H, W, 2)."""
        B, K, H, W = s.shape
        s_flat = s.reshape(B * K, 1, H, W)
        dummy_v = s.new_zeros(B * K, 0, H, W, 2)
        _, g = self.grad_conv(s_flat, dummy_v)
        return g.reshape(B, K, H, W, 2)

    def _gradient_2d(self, s: torch.Tensor) -> torch.Tensor:
        """∇s for (B, H, W) → (B, H, W, 2)."""
        return self._gradient(s.unsqueeze(1)).squeeze(1)

    def _divergence(self, uv: torch.Tensor) -> torch.Tensor:
        """∇·V for (B, K, H, W, 2) → (B, K, H, W)."""
        B, K, H, W, _ = uv.shape
        uv_flat = uv.reshape(B * K, 1, H, W, 2)
        dummy_s = uv.new_zeros(B * K, 0, H, W)
        d_flat, _ = self.div_conv(dummy_s, uv_flat)
        return d_flat.reshape(B, K, H, W)

    def _vorticity(self, uv: torch.Tensor) -> torch.Tensor:
        """curl(V) for (B, K, H, W, 2) → (B, K, H, W)."""
        B, K, H, W, _ = uv.shape
        uv_flat = uv.reshape(B * K, 1, H, W, 2)
        dummy_s = uv.new_zeros(B * K, 0, H, W)
        z_flat, _ = self.vort_conv(dummy_s, uv_flat)
        return z_flat.reshape(B, K, H, W)

    def _laplacian(self, s: torch.Tensor) -> torch.Tensor:
        return self._divergence(self._gradient(s))

    def _apply_diffusion(self, s: torch.Tensor) -> torch.Tensor:
        lap = self._laplacian(s)
        for _ in range(self.diffusion_order - 1):
            lap = self._laplacian(lap)
        return (-1) ** (self.diffusion_order + 1) * self.diffusion_coeff * lap  # type: ignore[operator]

    # ── Physics helpers ────────────────────────────────────────────────────

    def _level_pressures(self, p_s: torch.Tensor) -> torch.Tensor:
        """Mid-level pressures p_k = a_k + b_k · p_s.

        Args:
            p_s: (B, H, W)

        Returns:
            p_k: (B, K, H, W)
        """
        a = self.a_mid.view(1, -1, 1, 1)  # (1, K, 1, 1)
        b = self.b_mid.view(1, -1, 1, 1)
        return a + b * p_s.unsqueeze(1)

    def _layer_dp(self, p_s: torch.Tensor) -> torch.Tensor:
        """Layer thicknesses Δp_k = Δa_k + Δb_k · p_s.

        Args:
            p_s: (B, H, W)

        Returns:
            dp: (B, K, H, W)   (negative: pressure decreases upward)
        """
        da = self.delta_a.view(1, -1, 1, 1)
        db = self.delta_b.view(1, -1, 1, 1)
        return da + db * p_s.unsqueeze(1)

    def _hydrostatic(self, T: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        """Geopotential from T via hydrostatic balance.

        φ_0 = φ_surface
        φ_k = φ_{k-1} + R T_{k-1} ln(p_{k-1}/p_k)   k ≥ 1

        Unlike the sigma case, ln(p_{k-1}/p_k) is a spatially varying field
        because a ≠ 0.

        Args:
            T:   (B, K, H, W)
            p_k: (B, K, H, W) — mid-level pressures
        Returns:
            phi: (B, K, H, W)
        """
        B, K, H, W = T.shape
        phi = T.new_zeros(B, K, H, W)
        phi[:, 0] = self.phi_surface
        for k in range(1, K):
            log_ratio = torch.log(p_k[:, k - 1] / p_k[:, k])  # (B, H, W)
            phi[:, k] = phi[:, k - 1] + self.R * T[:, k - 1] * log_ratio
        return phi

    def _C_interfaces(
        self,
        div: torch.Tensor,
        uv: torch.Tensor,
        grad_ps: torch.Tensor,
        dp_k: torch.Tensor,
        dp_s_dt: torch.Tensor,
    ) -> torch.Tensor:
        """C = (∂p/∂η)η̇ at K+1 interfaces, (B, K+1, H, W).

        Recurrence (top-to-bottom, but indexed bottom-to-top here):

            C_{k+1/2} = C_{k-1/2}
                         − Δb_k ∂p_s/∂t
                         − Δp_k div_k
                         − Δb_k V_k·∇p_s

        Bottom BC: C_{-1/2} = 0 (surface, index 0 in the array).
        With the correct recurrence, C_{K+1/2} = 0 (top BC) is automatic.

        Args:
            div:      (B, K, H, W)
            uv:       (B, K, H, W, 2)
            grad_ps:  (B, H, W, 2)    — ∇p_s
            dp_k:     (B, K, H, W)    — Δp_k
            dp_s_dt:  (B, H, W)       — ∂p_s/∂t
        Returns:
            C_half: (B, K+1, H, W)
        """
        B, K, H, W = div.shape
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
        db = self.delta_b.view(1, K, 1, 1)  # (1, K, 1, 1)

        C_half = div.new_zeros(B, K + 1, H, W)
        # C_half[:, 0] = 0  (surface BC)
        for k in range(K):
            C_half[:, k + 1] = (
                C_half[:, k]
                - db[:, k] * dp_s_dt
                - dp_k[:, k] * div[:, k]
                - db[:, k] * v_dot_gps[:, k]
            )
        return C_half

    def _dX_dp(self, X: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        """∂X/∂p by central finite differences in pressure space.

        One-sided differences at top (k=0) and bottom (k=K-1).

        Args:
            X:   (B, K, H, W)
            p_k: (B, K, H, W) — mid-level pressures
        Returns:
            dX_dp: (B, K, H, W)
        """
        B, K, H, W = X.shape
        dX = X.new_zeros(B, K, H, W)
        if K == 1:
            return dX
        # Interior: central differences
        for k in range(1, K - 1):
            dp = p_k[:, k + 1] - p_k[:, k - 1]
            dX[:, k] = (X[:, k + 1] - X[:, k - 1]) / dp
        # Boundaries: one-sided
        dX[:, 0] = (X[:, 1] - X[:, 0]) / (p_k[:, 1] - p_k[:, 0])
        dX[:, K - 1] = (X[:, K - 1] - X[:, K - 2]) / (p_k[:, K - 1] - p_k[:, K - 2])
        return dX

    # ── Main tendency computation ──────────────────────────────────────────

    def compute_tendencies(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        p_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time tendencies for all prognostic variables.

        Args:
            uv:  (B, K, H, W, 2)
            T:   (B, K, H, W)
            q:   (B, K, H, W)
            p_s: (B, H, W)

        Returns:
            ``(duv_dt, dT_dt, dq_dt, dp_s_dt)``
        """
        B, K, H, W = T.shape

        # ── Pressure fields ───────────────────────────────────────────────
        p_k = self._level_pressures(p_s)  # (B, K, H, W)
        dp_k = self._layer_dp(p_s)  # (B, K, H, W)

        # ── Geopotential ──────────────────────────────────────────────────
        # Subtract the spatial mean before differentiating: only ∇φ matters,
        # and mean-subtraction eliminates the O(|φ|·ε) error from discrete
        # gradient of a constant, where ε is the discretization error.
        phi = self._hydrostatic(T, p_k)  # (B, K, H, W)
        phi = phi - phi.mean(dim=(-2, -1), keepdim=True)

        # ── ∇p_s ──────────────────────────────────────────────────────────
        # Same mean-subtraction for p_s: the large constant part
        # (~100 kPa) would otherwise create O(|p_s|·ε) error in the
        # discrete gradient.
        ps_anom = p_s - p_s.mean(dim=(-2, -1), keepdim=True)
        grad_ps = self._gradient_2d(ps_anom)  # (B, H, W, 2)

        # ── Divergence ────────────────────────────────────────────────────
        div = self._divergence(uv)  # (B, K, H, W)

        # ── Surface pressure tendency (before C, which needs it) ──────────
        # ∂p_s/∂t = −∑_k (Δp_k_physical div_k + Δb_k_physical V_k·∇p_s)
        # delta_b = b_interface[k+1] − b_interface[k] < 0 (b decreases upward),
        # so the physical layer Δb = −delta_b > 0, and similarly Δp = −dp_k > 0.
        # Substituting gives ∂p_s/∂t = +∑_k (dp_k div_k + delta_b V_k·∇p_s).
        db = self.delta_b.view(1, K, 1, 1)
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
        dp_s_dt = (dp_k * div + db * v_dot_gps).sum(dim=1)  # (B, H, W)

        # ── C interfaces and mid-level C ──────────────────────────────────
        C_half = self._C_interfaces(div, uv, grad_ps, dp_k, dp_s_dt)  # (B, K+1, H, W)
        C_k = 0.5 * (C_half[:, :-1] + C_half[:, 1:])  # (B, K,   H, W)

        # ── Pressure gradient force ───────────────────────────────────────
        # PGF_k = −∇φ_k − (R T_k b_k / p_k) ∇p_s
        grad_phi = self._gradient(phi)  # (B, K, H, W, 2)
        b_k = self.b_mid.view(1, K, 1, 1)
        pgf_coeff = (self.R * T * b_k / p_k).unsqueeze(-1)  # (B, K, H, W, 1)
        pgf = -grad_phi - pgf_coeff * grad_ps.unsqueeze(1)  # (B, K, H, W, 2)

        # ── Momentum: Lamb form + Coriolis + vertical advection ───────────
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)
        grad_ke = self._gradient(ke)  # (B, K, H, W, 2)

        vort = self._vorticity(uv)  # (B, K, H, W)
        BK = B * K
        uv_flat = uv.reshape(BK, 1, H, W, 2)
        vort_flat = vort.reshape(BK, 1, H, W)
        f_flat = self.f_coriolis.expand(BK, 1, H, W)

        vort_adv = self.vort_product(vort_flat, uv_flat).reshape(B, K, H, W, 2)
        coriolis = self.coriolis_product(f_flat, uv_flat).reshape(B, K, H, W, 2)

        # Vertical advection: −C_k ∂V/∂p
        dV0_dp = self._dX_dp(uv[..., 0], p_k)
        dV1_dp = self._dX_dp(uv[..., 1], p_k)
        vert_adv_uv = torch.stack([-C_k * dV0_dp, -C_k * dV1_dp], dim=-1)

        duv_dt = pgf - coriolis - grad_ke + vort_adv + vert_adv_uv

        # ── Temperature ───────────────────────────────────────────────────
        # ∂T/∂t = −V·∇T + (κT/p)ω − C_k ∂T/∂p
        # where ω_k = b_k(∂p_s/∂t + V_k·∇p_s) + C_k
        grad_T = self._gradient(T)
        horiz_adv_T = -(uv[..., 0] * grad_T[..., 0] + uv[..., 1] * grad_T[..., 1])

        kappa = self.R / C_P
        # V_k · ∇p_s (already computed as v_dot_gps)
        omega = b_k * (dp_s_dt.unsqueeze(1) + v_dot_gps) + C_k  # (B, K, H, W)
        dT_dp = self._dX_dp(T, p_k)
        dT_dt = horiz_adv_T + (kappa * T / p_k) * omega - C_k * dT_dp

        # ── Humidity ─────────────────────────────────────────────────────
        grad_q = self._gradient(q)
        dq_dt = -(
            uv[..., 0] * grad_q[..., 0] + uv[..., 1] * grad_q[..., 1]
        ) - C_k * self._dX_dp(q, p_k)

        # ── Optional diffusion ────────────────────────────────────────────
        if self.diffusion_coeff is not None:
            dT_dt = dT_dt + self._apply_diffusion(T)
            dq_dt = dq_dt + self._apply_diffusion(q)
            duv_dt = duv_dt + torch.stack(
                [self._apply_diffusion(uv[..., 0]), self._apply_diffusion(uv[..., 1])],
                dim=-1,
            )

        return duv_dt, dT_dt, dq_dt, dp_s_dt

    def step(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        p_s: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one time step with RK4."""

        def _tend(uv_, T_, q_, ps_):
            return self.compute_tendencies(uv_, T_, q_, ps_)

        k1 = _tend(uv, T, q, p_s)
        k2 = _tend(
            uv + 0.5 * dt * k1[0],
            T + 0.5 * dt * k1[1],
            q + 0.5 * dt * k1[2],
            p_s + 0.5 * dt * k1[3],
        )
        k3 = _tend(
            uv + 0.5 * dt * k2[0],
            T + 0.5 * dt * k2[1],
            q + 0.5 * dt * k2[2],
            p_s + 0.5 * dt * k2[3],
        )
        k4 = _tend(uv + dt * k3[0], T + dt * k3[1], q + dt * k3[2], p_s + dt * k3[3])

        c = dt / 6.0
        return (
            uv + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            T + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            q + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
            p_s + c * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        )

    def geopotential(self, T: torch.Tensor, p_s: torch.Tensor) -> torch.Tensor:
        """Geopotential at each level from temperature and surface pressure."""
        return self._hydrostatic(T, self._level_pressures(p_s))

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate scalar field over the sphere, shape (..., nlat, nlon)."""
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))


# ── Convenience constructors ───────────────────────────────────────────────


def sigma_coefficients(
    sigma_levels: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return hybrid coefficients that reproduce pure sigma coordinates.

    ``p_k = b_k * p_s``, so ``a_k = 0`` and ``b_k = σ_k``.  Interface
    values are the midpoints between adjacent sigma levels, with the surface
    interface fixed at σ=1 and the top interface at σ=0.

    Args:
        sigma_levels: σ_k values, surface-to-top (σ[0] = 1.0).

    Returns:
        ``(a_mid, b_mid, a_interface, b_interface)``
    """
    K = len(sigma_levels)
    sig = sigma_levels
    a_mid = [0.0] * K
    b_mid = list(sig)

    # Interface σ: surface=σ[0], top=0; intermediate = midpoints
    b_int = [sig[0]]
    for k in range(K - 1):
        b_int.append(0.5 * (sig[k] + sig[k + 1]))
    b_int.append(0.0)
    a_int = [0.0] * (K + 1)

    return a_mid, b_mid, a_int, b_int


def cam_like_coefficients(
    n_levels: int,
    p_top: float = 1000.0,
    p_ref: float = 101325.0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return hybrid coefficients similar to CAM/CESM vertical levels.

    Levels transition smoothly from sigma-like near the surface to
    pressure-like near the top::

        b_k = σ_k²   (falls to zero quadratically)
        a_k = σ_k p_ref (1 − b_k)  (where σ_k is log-spaced)

    The top interface pressure is ``p_top`` Pa.

    Args:
        n_levels: number of vertical levels K.
        p_top: pressure at the model top (Pa).
        p_ref: reference surface pressure (Pa).

    Returns:
        ``(a_mid, b_mid, a_interface, b_interface)``
    """
    # Log-spaced sigma from 1 (surface) to p_top/p_ref (top)
    import math as _math

    sigma_surface = 1.0
    sigma_top = p_top / p_ref

    sigma_mid = [
        _math.exp(
            _math.log(sigma_surface)
            + k * (_math.log(sigma_top) - _math.log(sigma_surface)) / (n_levels - 1)
        )
        for k in range(n_levels)
    ]

    b_mid = [s**2 for s in sigma_mid]
    a_mid = [s * p_ref * (1.0 - b) for s, b in zip(sigma_mid, b_mid)]

    # Interfaces: midpoints between adjacent mid-levels, with BCs
    b_int = [1.0]
    a_int = [0.0]
    for k in range(n_levels - 1):
        b_avg = 0.5 * (b_mid[k] + b_mid[k + 1])
        a_avg = 0.5 * (a_mid[k] + a_mid[k + 1])
        b_int.append(b_avg)
        a_int.append(a_avg)
    b_int.append(0.0)
    a_int.append(p_top)

    return a_mid, b_mid, a_int, b_int
