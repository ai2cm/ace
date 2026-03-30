"""Multi-level hydrostatic primitive equations in hybrid σ-p coordinates,
encoded in a VectorDiscoBlock.

The same physics as ``HybridCoordinateStepper`` but expressed in as few
DISCO forward passes as possible:

Block 1  (n_s = 4K + 2,  n_v = K + 1)
---------------------------------------
Per-level scalar channels — 4 kinds each:

    kind 0 — KE_k   kinetic energy (precomputed pointwise)
    kind 1 — φ_k    geopotential (precomputed via hydrostatic recurrence)
    kind 2 — ζ_k    vorticity   (zero input; filled by W_vs curl)
    kind 3 — δ_k    divergence  (zero input; filled by W_vs div)

Channel 4K   — f    Coriolis parameter (constant, passes through pointwise_ss)
Channel 4K+1 — p_s  surface pressure (used by W_sv for ∇p_s)

Vector channels 0..K-1 — V_k  horizontal velocity at each level
Vector channel  K      — dedicated ∇p_s output (zero input; filled by W_sv)

Encodings in Block 1
--------------------
W_vs  (diagonal, V_k → ζ_k and δ_k):
    W_vs[ch(k, ZETA), k, :, 0] = 1/R  →  ζ_k  (curl)
    W_vs[ch(k, DIV),  k, :, 1] = 1/R  →  δ_k  (divergence)

W_sv  (three groups):
    W_sv[k, ch(k, PHI),  :, 1] = -1/R  →  -∇φ_k     (hydrostatic PGF)
    W_sv[k, ch(k, KE),   :, 1] = -1/R  →  -∇KE_k    (Lamb term)
    W_sv[K, ps_ch,       :, 1] = +1/R  →  +∇p_s     (dedicated vector ch. K)

pointwise_ss:
    [f_ch, f_ch] = 1  →  f passes through for sv_product
    (all other rows zero)

sv_product:
    weight[ch(k, ZETA), k, 1] = +1   →  +ζ_k × V_k  (vorticity advection)
    weight[f_ch,        k, 1] = -1   →  -f   × V_k  (Coriolis), k = 0..K-1

Block 1 output
--------------
    y_v[:, :K] - uv  =  -∇KE_k - ∇φ_k + ζ_k×V_k - f×V_k     (partial momentum)
    y_v[:, K]        =  ∇p_s                                    (vector gradient)
    y_s[:, ch(k, ZETA)]  =  ζ_k
    y_s[:, ch(k, DIV)]   =  δ_k

adv_conv  (VectorDiscoConvS2, W_vs2 only, n_s = 2K, n_v = K)
-----------------------------------------------------------
    W_vs2[k,   k, :, 1] = -1/R  →  -V_k · ∇T_k
    W_vs2[K+k, k, :, 1] = -1/R  →  -V_k · ∇q_k

Explicit computations (no DISCO)
----------------------------------
* KE_k = 0.5(u² + v²)             (nonlinear, before block)
* p_k = a_k + b_k p_s             (pointwise, before block)
* dp_k = Δa_k + Δb_k p_s         (pointwise, before block)
* φ_k  hydrostatic recurrence     (state-dependent log-ratio, before block)
* V_k · ∇p_s = V_k · y_v[:,K]    (pointwise dot with block output)
* ∂p_s/∂t                        (vertical sum)
* PGF correction -(RT b/p)∇p_s   (pointwise; b/p_k is state-dependent)
* C_k recurrence                  (vertical, state-dependent)
* ω_k = b_k(∂p_s/∂t + V·∇p_s)+C_k (pointwise)
* ∂X/∂p finite differences        (vertical)
* Vertical advection -C_k ∂X/∂p  (pointwise)
* Adiabatic T term (κT/p) ω       (pointwise)

Two DISCO passes (block + adv_conv) suffice because:
* All remaining explicit terms are either pointwise nonlinearities with
  state-dependent coefficients, or vertical recurrences/differences — none
  involve additional horizontal spatial filtering.
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.block import VectorDiscoBlock

R_EARTH = 6.371e6  # m
C_P = 1004.0       # J/kg/K

# Per-level scalar kind indices
_KE_KIND = 0
_PHI_KIND = 1
_ZETA_KIND = 2
_DIV_KIND = 3
_N_KINDS = 4


def _ch(level: int, kind: int) -> int:
    """Channel index for a given (level, kind) pair (interleaved layout)."""
    return level * _N_KINDS + kind


class HybridCoordinateBlockStepper(nn.Module):
    """Multi-level primitive equations in hybrid σ-p coordinates using VectorDiscoBlock.

    Encodes the full horizontal dynamics in two DISCO forward passes:
    - ``self.block`` (Block 1): vorticity, divergence, -∇KE, -∇φ, Coriolis,
      vorticity advection, and ∇p_s — all in one ``VectorDiscoBlock`` pass.
    - ``self.adv_conv``: horizontal T and q advection via W_vs2.

    The remaining terms (hydrostatic φ recurrence, PGF correction, ∂p_s/∂t,
    C_k, ω, vertical advection) are computed explicitly because they involve
    state-dependent coefficients or vertical operations.

    State: ``(uv, T, q, p_s)`` — same as ``HybridCoordinateStepper``.

    Parameters
    ----------
    shape:
        ``(nlat, nlon)`` grid dimensions.
    a_mid, b_mid:
        Level mid-point hybrid coefficients; ``p_k = a_k + b_k p_s``.
    a_interface, b_interface:
        Interface hybrid coefficients (length K+1).
    R:
        Gas constant for dry air (J/kg/K).
    omega:
        Earth rotation rate (rad/s).
    phi_surface:
        Surface geopotential (m²/s²).
    kernel_shape, theta_cutoff:
        DISCO filter parameters.
    diffusion_coeff, diffusion_order:
        Optional ∇^(2n) hyperdiffusion.
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

        self.register_buffer("a_mid", a_m)
        self.register_buffer("b_mid", b_m)
        self.register_buffer("a_interface", a_i)
        self.register_buffer("b_interface", b_i)

        delta_a = a_i[1:] - a_i[:-1]
        delta_b = b_i[1:] - b_i[:-1]
        self.register_buffer("delta_a", delta_a)
        self.register_buffer("delta_b", delta_b)

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))
        self.register_buffer(
            "area_weights",
            (quad_weights * 2.0 * math.pi / self.nlon).float(),
        )

        # ── Block 1 (n_s = 4K+2, n_v = K+1) ────────────────────────────────
        # 4 kinds per level (KE, φ, ζ, δ), plus f (4K) and p_s (4K+1).
        # K velocity channels (0..K-1) plus one dedicated ∇p_s channel (K).
        n_s = K * _N_KINDS + 2
        n_v = K + 1
        self.block = VectorDiscoBlock(
            n_scalar=n_s,
            n_vector=n_v,
            shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            activation="none",
        )

        # ── adv_conv: -V·∇T and -V·∇q via W_vs2 ────────────────────────────
        self.adv_conv = VectorDiscoConvS2(
            in_channels_scalar=2 * K,
            in_channels_vector=K,
            out_channels_scalar=2 * K,
            out_channels_vector=0,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # ── Diffusion convs (1-channel, used with reshape for K levels) ──────
        disco_kw = dict(
            in_shape=shape, out_shape=shape,
            kernel_shape=kernel_shape, theta_cutoff=theta_cutoff, bias=False,
        )
        self.grad_conv = VectorDiscoConvS2(
            in_channels_scalar=1, in_channels_vector=0,
            out_channels_scalar=0, out_channels_vector=1, **disco_kw,
        )
        self.div_conv = VectorDiscoConvS2(
            in_channels_scalar=0, in_channels_vector=1,
            out_channels_scalar=1, out_channels_vector=0, **disco_kw,
        )

        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self._init_physics_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_physics_weights(self) -> None:
        """Set all block and conv weights to encode the hybrid primitive equations."""
        K = self.n_levels
        f_ch = K * _N_KINDS       # channel index of the Coriolis parameter
        ps_ch = K * _N_KINDS + 1  # channel index of surface pressure

        # ── Zero all block conv parameters ───────────────────────────────────
        self.block.conv.W_ss.zero_()
        self.block.conv.W_vs.zero_()
        self.block.conv.W_sv.zero_()
        self.block.conv.W_vv.zero_()
        if self.block.conv.W_vs2 is not None:
            self.block.conv.W_vs2.zero_()
        if self.block.conv.bias_scalar is not None:
            self.block.conv.bias_scalar.zero_()

        # ── W_vs: V_k → ζ_k (curl, d=0) and δ_k (div, d=1) ─────────────────
        # W_vs shape: (n_s, n_v=K+1, K_v, 2)
        # Diagonal coupling: V_k feeds only level-k scalar channels.
        # Vector channel K (∇p_s) contributes nothing to scalars.
        W_vs = self.block.conv.W_vs
        for k in range(K):
            W_vs[_ch(k, _ZETA_KIND), k, :, 0] = 1.0 / R_EARTH  # curl → ζ_k
            W_vs[_ch(k, _DIV_KIND),  k, :, 1] = 1.0 / R_EARTH  # div  → δ_k

        # ── W_sv: -∇φ_k and -∇KE_k into velocity ch.; +∇p_s into ch. K ──────
        # W_sv shape: (n_v=K+1, n_s, K_v, 2)
        W_sv = self.block.conv.W_sv
        for k in range(K):
            W_sv[k, _ch(k, _PHI_KIND), :, 1] = -1.0 / R_EARTH  # -∇φ_k  → V_k
            W_sv[k, _ch(k, _KE_KIND),  :, 1] = -1.0 / R_EARTH  # -∇KE_k → V_k
        W_sv[K, ps_ch, :, 1] = 1.0 / R_EARTH  # +∇p_s → dedicated channel K

        # ── pointwise_ss: only f passes through ──────────────────────────────
        self.block.pointwise_ss.weight.zero_()
        self.block.pointwise_ss.weight[f_ch, f_ch, 0, 0] = 1.0

        # ── Zero scalar MLP ──────────────────────────────────────────────────
        if self.block.scalar_mlp is not None:
            for module in self.block.scalar_mlp:
                if hasattr(module, "weight"):
                    module.weight.zero_()
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.zero_()

        # ── sv_product: +ζ_k×V_k (Lamb) and -f×V_k (Coriolis) ───────────────
        # weight shape: (n_s, n_v=K+1, 2)   index 1 = rotation (rot90)
        # Column K (∇p_s channel) stays zero — no sv_product for ∇p_s.
        self.block.sv_product.weight.zero_()
        for k in range(K):
            self.block.sv_product.weight[_ch(k, _ZETA_KIND), k, 1] = +1.0
        self.block.sv_product.weight[f_ch, :K, 1] = -1.0  # Coriolis k=0..K-1

        # ── Zero pointwise_vv ────────────────────────────────────────────────
        self.block.pointwise_vv.weight.zero_()

        # ── Gradient/div convs: used only for ∇²ˢ diffusion ──────────────────
        self.grad_conv.W_sv.zero_()
        self.grad_conv.W_sv[0, 0, :, 1] = 1.0 / R_EARTH
        self.div_conv.W_vs.zero_()
        self.div_conv.W_vs[0, 0, :, 1] = 1.0 / R_EARTH

        # ── adv_conv: -V·∇T and -V·∇q via W_vs2 ─────────────────────────────
        # Scalar layout: indices 0..K-1 = T levels, K..2K-1 = q levels.
        self.adv_conv.W_ss.zero_()
        self.adv_conv.W_vs.zero_()
        W_vs2 = self.adv_conv.W_vs2  # shape (2K, K, K_v, 2)
        W_vs2.zero_()
        for k in range(K):
            W_vs2[k,     k, :, 1] = -1.0 / R_EARTH  # -V_k · ∇T_k
            W_vs2[K + k, k, :, 1] = -1.0 / R_EARTH  # -V_k · ∇q_k

    # ── Physics helpers (identical to HybridCoordinateStepper) ───────────────

    def _level_pressures(self, p_s: torch.Tensor) -> torch.Tensor:
        """p_k = a_k + b_k p_s, (B, H, W) → (B, K, H, W)."""
        a = self.a_mid.view(1, -1, 1, 1)
        b = self.b_mid.view(1, -1, 1, 1)
        return a + b * p_s.unsqueeze(1)

    def _layer_dp(self, p_s: torch.Tensor) -> torch.Tensor:
        """Δp_k = Δa_k + Δb_k p_s, (B, H, W) → (B, K, H, W)."""
        da = self.delta_a.view(1, -1, 1, 1)
        db = self.delta_b.view(1, -1, 1, 1)
        return da + db * p_s.unsqueeze(1)

    def _hydrostatic(self, T: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        """Geopotential from T via hydrostatic balance.

        φ_0 = φ_surface
        φ_k = φ_{k-1} + R T_{k-1} ln(p_{k-1}/p_k)   k ≥ 1
        """
        B, K, H, W = T.shape
        phi = T.new_zeros(B, K, H, W)
        phi[:, 0] = self.phi_surface
        for k in range(1, K):
            log_ratio = torch.log(p_k[:, k - 1] / p_k[:, k])
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

        C_{k+1/2} = C_{k-1/2} - Δb_k ∂p_s/∂t - Δp_k div_k - Δb_k V_k·∇p_s
        Bottom BC: C_{-1/2} = 0.
        """
        B, K, H, W = div.shape
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
        db = self.delta_b.view(1, K, 1, 1)
        C_half = div.new_zeros(B, K + 1, H, W)
        for k in range(K):
            C_half[:, k + 1] = (
                C_half[:, k]
                - db[:, k] * dp_s_dt
                - dp_k[:, k] * div[:, k]
                - db[:, k] * v_dot_gps[:, k]
            )
        return C_half

    def _dX_dp(self, X: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        """∂X/∂p by central differences; one-sided at top/bottom boundaries."""
        B, K, H, W = X.shape
        dX = X.new_zeros(B, K, H, W)
        if K == 1:
            return dX
        for k in range(1, K - 1):
            dp = p_k[:, k + 1] - p_k[:, k - 1]
            dX[:, k] = (X[:, k + 1] - X[:, k - 1]) / dp
        dX[:, 0]     = (X[:, 1]     - X[:, 0])     / (p_k[:, 1]     - p_k[:, 0])
        dX[:, K - 1] = (X[:, K - 1] - X[:, K - 2]) / (p_k[:, K - 1] - p_k[:, K - 2])
        return dX

    # ── Diffusion helpers ─────────────────────────────────────────────────────

    def _gradient(self, s: torch.Tensor) -> torch.Tensor:
        """∇s for (B, K, H, W) → (B, K, H, W, 2) via grad_conv."""
        B, K, H, W = s.shape
        s_flat = s.reshape(B * K, 1, H, W)
        dummy = s.new_zeros(B * K, 0, H, W, 2)
        _, g = self.grad_conv(s_flat, dummy)
        return g.reshape(B, K, H, W, 2)

    def _laplacian(self, s: torch.Tensor) -> torch.Tensor:
        """∇²s = ∇·(∇s) for (B, K, H, W)."""
        B, K, H, W = s.shape
        grad = self._gradient(s)
        uv_flat = grad.reshape(B * K, 1, H, W, 2)
        dummy_s = grad.new_zeros(B * K, 0, H, W)
        d_flat, _ = self.div_conv(dummy_s, uv_flat)
        return d_flat.reshape(B, K, H, W)

    def _apply_diffusion(self, s: torch.Tensor) -> torch.Tensor:
        lap = self._laplacian(s)
        for _ in range(self.diffusion_order - 1):
            lap = self._laplacian(lap)
        return (-1) ** (self.diffusion_order + 1) * self.diffusion_coeff * lap  # type: ignore[operator]

    # ── Main tendency computation ─────────────────────────────────────────────

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

        # ── Precompute KE, p_k, dp_k, φ_k ────────────────────────────────────
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)  # (B, K, H, W)
        p_k  = self._level_pressures(p_s)                 # (B, K, H, W)
        dp_k = self._layer_dp(p_s)                        # (B, K, H, W)
        phi  = self._hydrostatic(T, p_k)                  # (B, K, H, W)

        # ── Assemble block inputs ─────────────────────────────────────────────
        zeros = torch.zeros_like(T)
        per_level = torch.stack([ke, phi, zeros, zeros], dim=2)  # (B, K, 4, H, W)
        x_s_body = per_level.reshape(B, K * _N_KINDS, H, W)
        x_s = torch.cat([
            x_s_body,
            self.f_coriolis.expand(B, 1, H, W),
            p_s.unsqueeze(1),
        ], dim=1)  # (B, 4K+2, H, W)

        x_v = torch.cat([
            uv,
            uv.new_zeros(B, 1, H, W, 2),
        ], dim=1)  # (B, K+1, H, W, 2)

        # ── Block 1: one pass → momentum, ζ, δ, ∇p_s ─────────────────────────
        y_s, y_v = self.block(x_s, x_v)

        # Extract divergence and ∇p_s from block output
        y_s_levels = y_s[:, : K * _N_KINDS].reshape(B, K, _N_KINDS, H, W)
        div    = y_s_levels[:, :, _DIV_KIND]  # (B, K, H, W)
        grad_ps = y_v[:, K]                   # (B, H, W, 2)

        # Partial momentum tendency: -∇KE_k - ∇φ_k + ζ_k×V_k - f×V_k
        duv_partial = y_v[:, :K] - uv          # (B, K, H, W, 2)

        # ── Surface pressure tendency ──────────────────────────────────────────
        db = self.delta_b.view(1, K, 1, 1)
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)       # (B, K, H, W)
        dp_s_dt = (dp_k * div + db * v_dot_gps).sum(dim=1)   # (B, H, W)

        # ── C interfaces and mid-level C ───────────────────────────────────────
        C_half = self._C_interfaces(div, uv, grad_ps, dp_k, dp_s_dt)
        C_k = 0.5 * (C_half[:, :-1] + C_half[:, 1:])  # (B, K, H, W)

        # ── PGF correction: -(R T b / p) ∇p_s ────────────────────────────────
        b_k = self.b_mid.view(1, K, 1, 1)
        pgf_corr = -(self.R * T * b_k / p_k).unsqueeze(-1) * grad_ps.unsqueeze(1)

        # ── Vertical advection of V: -C_k ∂V/∂p ──────────────────────────────
        dV0_dp = self._dX_dp(uv[..., 0], p_k)
        dV1_dp = self._dX_dp(uv[..., 1], p_k)
        vert_adv_uv = torch.stack([-C_k * dV0_dp, -C_k * dV1_dp], dim=-1)

        duv_dt = duv_partial + pgf_corr + vert_adv_uv

        # ── adv_conv (W_vs2): -V·∇T and -V·∇q ────────────────────────────────
        tq = torch.cat([T, q], dim=1)    # (B, 2K, H, W)
        dTq, _ = self.adv_conv(tq, uv)  # (B, 2K, H, W)
        dT_dt = dTq[:, :K]
        dq_dt = dTq[:, K:]

        # ── ω and adiabatic T ─────────────────────────────────────────────────
        omega = b_k * (dp_s_dt.unsqueeze(1) + v_dot_gps) + C_k
        kappa = self.R / C_P
        dT_dp = self._dX_dp(T, p_k)
        dT_dt = dT_dt + omega * (kappa * T / p_k - dT_dp)

        # ── Vertical advection of q ───────────────────────────────────────────
        dq_dt = dq_dt - C_k * self._dX_dp(q, p_k)

        # ── Optional diffusion ────────────────────────────────────────────────
        if self.diffusion_coeff is not None:
            dT_dt  = dT_dt  + self._apply_diffusion(T)
            dq_dt  = dq_dt  + self._apply_diffusion(q)
            duv_dt = duv_dt + torch.stack(
                [self._apply_diffusion(uv[..., 0]),
                 self._apply_diffusion(uv[..., 1])], dim=-1,
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
        k2 = _tend(uv + 0.5*dt*k1[0], T + 0.5*dt*k1[1],
                   q  + 0.5*dt*k1[2], p_s + 0.5*dt*k1[3])
        k3 = _tend(uv + 0.5*dt*k2[0], T + 0.5*dt*k2[1],
                   q  + 0.5*dt*k2[2], p_s + 0.5*dt*k2[3])
        k4 = _tend(uv +     dt*k3[0], T +     dt*k3[1],
                   q  +     dt*k3[2], p_s +     dt*k3[3])

        c = dt / 6.0
        return (
            uv  + c * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
            T   + c * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
            q   + c * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]),
            p_s + c * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]),
        )

    def geopotential(self, T: torch.Tensor, p_s: torch.Tensor) -> torch.Tensor:
        """Geopotential at each level from temperature and surface pressure."""
        return self._hydrostatic(T, self._level_pressures(p_s))

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate scalar field over the sphere, shape (..., nlat, nlon)."""
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))
