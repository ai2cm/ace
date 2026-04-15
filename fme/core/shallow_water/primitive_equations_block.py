"""Multi-level isobaric primitive equations encoded in a single VectorDiscoBlock.

Equivalent to ``PrimitiveEquationsStepper`` but encodes all horizontal
operations (dynamics, advection, diffusion) in one DISCO forward pass.

Block  (n_s = 6K + 1,  n_v = K,  residual = False)
----------------------------------------------------
Per-level scalar channels — 6 kinds each:

    kind 0 — T_anom  mean-subtracted temperature (drives PGF via W_sv)
    kind 1 — KE_k    kinetic energy (drives -∇KE via W_sv)
    kind 2 — ζ_k     vorticity (zero input; filled by W_vs curl)
    kind 3 — δ_k     divergence (zero input; filled by W_vs div)
    kind 4 — T_k     temperature (advected by W_vs2, diffused by W_ss)
    kind 5 — q_k     specific humidity (advected by W_vs2, diffused by W_ss)

Channel 6K — f   Coriolis parameter (passes through pointwise_ss)

Vector channels 0..K-1 — V_k  horizontal velocity at each level.

Encodings
---------
W_sv:   lower-triangular T_anom → -∇φ_k (hydrostatic PGF),
        diagonal KE → -∇KE_k (Lamb term)
W_vs:   V_k → ζ_k (curl) and δ_k (divergence)
W_vs2:  -V_k·∇T_k, -V_k·∇q_k (horizontal advection)
W_ss:   ν∇²T_k, ν∇²q_k (scalar diffusion, approximate Laplacian)
W_vv:   ν∇²V_k (velocity diffusion, approximate Laplacian)
sv_product: ζ_k×V_k (vorticity advection), -f×V_k (Coriolis)

Block output (residual=False → raw tendencies)
----------------------------------------------
    y_v[:, :K]  =  PGF + Coriolis + ζ×V - ∇KE + ν∇²V  (momentum tendency)
    y_s T_k     =  -V·∇T + ν∇²T                        (partial T tendency)
    y_s q_k     =  -V·∇q + ν∇²q                        (partial q tendency)
    y_s ζ_k/δ_k =  vorticity / divergence

Physics computed externally
---------------------------
* KE precomputation (pointwise, before block)
* ω from vertically-integrated δ_k (from block output)
* T vertical coupling  (ω (κT/p − ∂T/∂p))
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.shallow_water.block import VectorDiscoBlock
from fme.core.shallow_water.hybrid_equations_block import _compute_laplacian_weights
from fme.core.shallow_water.scalar_vector_product import VectorDotProduct

R_EARTH = 6.371e6  # m
C_P = 1004.0  # J/kg/K

# Per-level scalar kind indices
_T_ANOM_KIND = 0
_KE_KIND = 1
_ZETA_KIND = 2
_DIV_KIND = 3
_T_KIND = 4
_Q_KIND = 5
_N_KINDS = 6


def _ch(level: int, kind: int) -> int:
    """Channel index for a given (level, kind) pair (interleaved layout)."""
    return level * _N_KINDS + kind


class PrimitiveEquationsBlockStepper(nn.Module):
    """Multi-level isobaric primitive equations using a VectorDiscoBlock.

    Encodes all horizontal dynamics in a single DISCO forward pass.
    See module docstring for channel layout and encoding details.

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
        """
        Args:
            shape: ``(nlat, nlon)`` grid dimensions.
            n_levels: number of pressure levels K.
            pressure_levels: pressure at each level (Pa), surface-to-top.
                Defaults to K levels log-spaced from 100 000 Pa to 10 000 Pa.
            R: dry-air gas constant (J/kg/K).
            omega: Earth rotation rate (rad/s).
            phi_surface: surface geopotential (m²/s²).
            kernel_shape: DISCO filter kernel shape.
            theta_cutoff: filter support radius (rad).
            diffusion_coeff: ν for ∇² diffusion (m²/s). None disables it.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.n_levels = n_levels
        self.R = R
        self.phi_surface = phi_surface
        self.diffusion_coeff = diffusion_coeff

        K = n_levels
        if pressure_levels is None:
            pressure_levels = [
                100000.0 * math.exp(-k * math.log(10.0) / max(1, K - 1))
                for k in range(K)
            ]
        if len(pressure_levels) != K:
            raise ValueError(
                f"len(pressure_levels)={len(pressure_levels)} != n_levels={K}"
            )

        p = torch.tensor(pressure_levels, dtype=torch.float32)
        self.register_buffer("pressure_levels", p)

        log_p_ratio = torch.log(p[:-1] / p[1:]) if K > 1 else torch.zeros(0)
        self.register_buffer("log_p_ratio", log_p_ratio)

        # Layer thicknesses for ω integration
        if K > 1:
            p_lower = torch.cat([p[:1], 0.5 * (p[:-1] + p[1:])])
            p_upper = torch.cat([0.5 * (p[:-1] + p[1:]), p.new_zeros(1)])
        else:
            p_lower, p_upper = p[:1], p.new_zeros(1)
        self.register_buffer("delta_p", (p_lower - p_upper).float())

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        self.register_buffer(
            "f_coriolis",
            (2.0 * omega * torch.sin(lats)).float().reshape(1, 1, -1, 1),
        )
        self.register_buffer(
            "area_weights",
            (quad_weights * 2.0 * math.pi / self.nlon).float(),
        )

        # ── Block (n_s = 6K+1, n_v = K) ─────────────────────────────────────
        n_s = K * _N_KINDS + 1  # 6 kinds per level + f
        n_v = K
        self.block = VectorDiscoBlock(
            n_scalar=n_s,
            n_vector=n_v,
            shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            activation="none",
            residual=False,
        )

        # ── ke_product: KE_k = ½|V_k|² via diagonal VectorDotProduct ────────
        self.ke_product = VectorDotProduct(n_scalar=K, n_vector=K)

        # Freeze all parameters; physics weights are fixed
        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self._init_physics_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_physics_weights(self) -> None:
        """Set all block weights to encode the primitive equations."""
        K = self.n_levels
        R = self.R
        log_p_ratio = self.log_p_ratio  # (K-1,)
        f_ch = K * _N_KINDS  # channel index of the Coriolis parameter

        # ── Zero all block conv parameters ───────────────────────────────────
        self.block.conv.W_ss.zero_()
        self.block.conv.W_vs.zero_()
        self.block.conv.W_sv.zero_()
        self.block.conv.W_vv.zero_()
        if self.block.conv.W_vs2 is not None:
            self.block.conv.W_vs2.zero_()
        if self.block.conv.bias_scalar is not None:
            self.block.conv.bias_scalar.zero_()

        # ── W_vs: generate ζ and δ from each velocity level ──────────────────
        W_vs = self.block.conv.W_vs
        for k in range(K):
            W_vs[_ch(k, _ZETA_KIND), k, :, 0] = 1.0 / R_EARTH  # curl  → ζ_k
            W_vs[_ch(k, _DIV_KIND), k, :, 1] = 1.0 / R_EARTH  # div   → δ_k

        # ── W_sv: PGF (lower-triangular in levels) and −∇KE (diagonal) ───────
        W_sv = self.block.conv.W_sv
        for k_out in range(K):
            for k_in in range(k_out):
                W_sv[k_out, _ch(k_in, _T_ANOM_KIND), :, 1] = (
                    -R * log_p_ratio[k_in].item() / R_EARTH
                )
            W_sv[k_out, _ch(k_out, _KE_KIND), :, 1] = -1.0 / R_EARTH

        # ── W_vs2: -V_k·∇T_k and -V_k·∇q_k (horizontal advection) ──────────
        assert self.block.conv.W_vs2 is not None
        W_vs2 = self.block.conv.W_vs2
        for k in range(K):
            W_vs2[_ch(k, _T_KIND), k, :, 1] = -1.0 / R_EARTH
            W_vs2[_ch(k, _Q_KIND), k, :, 1] = -1.0 / R_EARTH

        # ── W_ss / W_vv: diffusion via Laplacian weights ─────────────────────
        if self.diffusion_coeff is not None:
            lap_w = _compute_laplacian_weights(self.block.conv)
            nu_R2 = self.diffusion_coeff / R_EARTH**2
            for k in range(K):
                self.block.conv.W_ss[_ch(k, _T_KIND), _ch(k, _T_KIND)] = nu_R2 * lap_w
                self.block.conv.W_ss[_ch(k, _Q_KIND), _ch(k, _Q_KIND)] = nu_R2 * lap_w
            for k in range(K):
                self.block.conv.W_vv[k, k, :, 0] = nu_R2 * lap_w

        # ── pointwise_ss: only f passes through ──────────────────────────────
        self.block.pointwise_ss.weight.zero_()
        self.block.pointwise_ss.weight[f_ch, f_ch, 0, 0] = 1.0

        # ── Zero scalar MLP ──────────────────────────────────────────────────
        for module in self.block.scalar_mlp:
            if hasattr(module, "weight"):
                module.weight.zero_()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()

        # ── sv_product: +ζ_k × V_k  (Lamb) and  −f × V_k  (Coriolis) ────────
        assert self.block.sv_product is not None
        self.block.sv_product.weight.zero_()
        for k in range(K):
            self.block.sv_product.weight[_ch(k, _ZETA_KIND), k, 1] = +1.0
        self.block.sv_product.weight[f_ch, :, 1] = -1.0

        # ── Zero pointwise_vv ────────────────────────────────────────────────
        assert self.block.pointwise_vv is not None
        self.block.pointwise_vv.weight.zero_()

        # ── ke_product: diagonal weight 0.5 → KE_k = ½|V_k|² ───────────────
        self.ke_product.weight.zero_()
        for k in range(K):
            self.ke_product.weight[k, k] = 0.5

    # ── Physics helpers ──────────────────────────────────────────────────────

    def _omega(self, div: torch.Tensor) -> torch.Tensor:
        """ω = dp/dt at each level, integrated upward from ω=0 at the surface."""
        B, K, H, W = div.shape
        omega_half = div.new_zeros(B, K + 1, H, W)
        for k in range(K):
            omega_half[:, k + 1] = omega_half[:, k] + div[:, k] * self.delta_p[k]
        return 0.5 * (omega_half[:, :-1] + omega_half[:, 1:])

    def _dT_dp(self, T: torch.Tensor) -> torch.Tensor:
        """∂T/∂p by central differences (one-sided at boundaries)."""
        p = self.pressure_levels
        B, K, H, W = T.shape
        dT = T.new_zeros(B, K, H, W)
        if K == 1:
            return dT
        for k in range(1, K - 1):
            dT[:, k] = (T[:, k + 1] - T[:, k - 1]) / (p[k + 1] - p[k - 1])
        dT[:, 0] = (T[:, 1] - T[:, 0]) / (p[1] - p[0])
        dT[:, K - 1] = (T[:, K - 1] - T[:, K - 2]) / (p[K - 1] - p[K - 2])
        return dT

    # ── Main interface ────────────────────────────────────────────────────────

    def compute_tendencies(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time tendencies for all prognostic variables.

        Args:
            uv: velocity, shape (B, K, H, W, 2).
            T:  temperature, shape (B, K, H, W).
            q:  humidity, shape (B, K, H, W).

        Returns:
            (duv_dt, dT_dt, dq_dt) with the same shapes as the inputs.
        """
        B, K, H, W = T.shape

        # ── KE per level via ke_product ───────────────────────────────────────
        ke = self.ke_product(uv)  # (B, K, H, W)

        # ── Assemble block inputs ─────────────────────────────────────────────
        T_anom = T - T.mean(dim=(-2, -1), keepdim=True)
        zeros = torch.zeros_like(T)
        per_level = torch.stack(
            [T_anom, ke, zeros, zeros, T, q], dim=2
        )  # (B, K, 6, H, W)
        x_s_body = per_level.reshape(B, K * _N_KINDS, H, W)
        x_s = torch.cat(
            [x_s_body, self.f_coriolis.expand(B, 1, H, W)], dim=1
        )  # (B, 6K+1, H, W)

        # ── Single block pass ─────────────────────────────────────────────────
        y_s, y_v = self.block(x_s, uv)
        duv_dt = y_v  # (B, K, H, W, 2) — raw tendency (residual=False)

        # Extract per-level scalar outputs
        y_s_per_level = y_s[:, : K * _N_KINDS].reshape(B, K, _N_KINDS, H, W)
        div = y_s_per_level[:, :, _DIV_KIND]  # (B, K, H, W)
        dT_dt = y_s_per_level[:, :, _T_KIND]  # -V·∇T (+ ν∇²T)
        dq_dt = y_s_per_level[:, :, _Q_KIND]  # -V·∇q (+ ν∇²q)

        # ── ω from divergence ─────────────────────────────────────────────────
        omega = self._omega(div)  # (B, K, H, W)

        # ── T adiabatic vertical coupling ─────────────────────────────────────
        p_k = self.pressure_levels.view(1, K, 1, 1)
        dT_dt = dT_dt + omega * (self.R / C_P * T / p_k - self._dT_dp(T))

        return duv_dt, dT_dt, dq_dt

    def step(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one time step with RK4."""
        k1 = self.compute_tendencies(uv, T, q)
        k2 = self.compute_tendencies(
            uv + 0.5 * dt * k1[0], T + 0.5 * dt * k1[1], q + 0.5 * dt * k1[2]
        )
        k3 = self.compute_tendencies(
            uv + 0.5 * dt * k2[0], T + 0.5 * dt * k2[1], q + 0.5 * dt * k2[2]
        )
        k4 = self.compute_tendencies(uv + dt * k3[0], T + dt * k3[1], q + dt * k3[2])
        c = dt / 6.0
        return (
            uv + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            T + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            q + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        )

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate a scalar field over the sphere."""
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))

    def total_kinetic_energy(self, uv: torch.Tensor) -> torch.Tensor:
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)
        return self.integrate_area(ke).sum(dim=1)
