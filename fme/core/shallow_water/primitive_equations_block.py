"""Multi-level isobaric primitive equations encoded in a VectorDiscoBlock.

Equivalent to ``PrimitiveEquationsStepper`` but encodes the complete momentum
tendency in a single ``VectorDiscoBlock`` forward pass by treating each
vertical level as a separate channel.

Channel layout (n_s = 4K + 1, n_v = K)
----------------------------------------
Channels are interleaved per level so that the folded view

    x_s[:, :4K].reshape(B, K, 4, H, W)

gives axis-2 kinds:

    kind 0 — T_k   temperature (input; drives PGF via W_sv)
    kind 1 — KE_k  kinetic energy (precomputed input; drives -∇KE via W_sv)
    kind 2 — ζ_k   vorticity (zero input; filled by W_vs curl; used by sv_product)
    kind 3 — δ_k   divergence (zero input; filled by W_vs div; extracted for ω)

Channel 4K — f   Coriolis parameter (constant; passed through pointwise_ss)

Vector channels 0..K-1 — V_k  horizontal velocity at each level.

Physics encoded in the block
------------------------------
* W_sv:      lower-triangular T → -∇φ_k  (hydrostatic geopotential PGF)
             diagonal      KE → -∇KE_k  (Lamb kinetic-energy term)
* W_vs:      V_k → ζ_k (curl, component 0) and δ_k (div, component 1)
* sv_product:  ζ_k × V_k  (+1, vorticity advection)
               f   × V_k  (-1, Coriolis)

Physics computed externally
-----------------------------
* KE precomputation (pointwise, before block call)
* ω from vertically-integrated δ_k (extracted from block scalar output)
* Horizontal T and q advection  (-V · ∇T,  -V · ∇q)
* T vertical coupling  (ω (κT/p − ∂T/∂p))
* Optional ∇^(2n) diffusion
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.block import VectorDiscoBlock


R_EARTH = 6.371e6  # m
C_P = 1004.0  # J/kg/K

# Per-level scalar kind indices
_T_KIND = 0
_KE_KIND = 1
_ZETA_KIND = 2
_DIV_KIND = 3
_N_KINDS = 4


def _ch(level: int, kind: int) -> int:
    """Channel index for a given (level, kind) pair (interleaved layout)."""
    return level * _N_KINDS + kind


class PrimitiveEquationsBlockStepper(nn.Module):
    """Multi-level isobaric primitive equations using a single VectorDiscoBlock.

    Equivalent to ``PrimitiveEquationsStepper`` in physics, but the full
    momentum tendency (PGF + Coriolis + vorticity advection − ∇KE) is encoded
    in one block pass via fixed weights.  Temperature and humidity tendencies
    are still computed with conventional DISCO operators outside the block
    (horizontal advection requires a dot-product operation not yet in the block
    architecture).

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
        diffusion_order: int = 1,
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
            diffusion_coeff: ν for ∇^(2n) diffusion. None disables it.
            diffusion_order: n in ∇^(2n); 1 = Laplacian, 2 = biharmonic.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.n_levels = n_levels
        self.R = R
        self.phi_surface = phi_surface
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_order = diffusion_order

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

        # Layer thicknesses for ω integration (same as PrimitiveEquationsStepper)
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

        # ── Block (encodes momentum tendency) ────────────────────────────────
        n_s = K * _N_KINDS + 1  # 4 kinds per level + f
        n_v = K
        self.block = VectorDiscoBlock(
            n_scalar=n_s,
            n_vector=n_v,
            shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            activation="none",
        )

        # ── Gradient conv for T/q advection (can't go in block) ──────────────
        disco_kw = dict(
            in_shape=shape, out_shape=shape,
            kernel_shape=kernel_shape, theta_cutoff=theta_cutoff, bias=False,
        )
        self.grad_conv = VectorDiscoConvS2(
            in_channels_scalar=1, in_channels_vector=0,
            out_channels_scalar=0, out_channels_vector=1, **disco_kw
        )
        self.div_conv = VectorDiscoConvS2(
            in_channels_scalar=0, in_channels_vector=1,
            out_channels_scalar=1, out_channels_vector=0, **disco_kw
        )

        # Freeze all parameters; physics weights are fixed
        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self._init_physics_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_physics_weights(self) -> None:
        """Set all block and conv weights to encode the primitive equations."""
        K = self.n_levels
        R = self.R
        log_p_ratio = self.log_p_ratio  # (K-1,)
        f_ch = K * _N_KINDS  # channel index of the Coriolis parameter

        # ── Zero all block conv parameters ───────────────────────────────────
        self.block.conv.W_ss.zero_()
        self.block.conv.W_vs.zero_()
        self.block.conv.W_sv.zero_()
        self.block.conv.W_vv.zero_()
        if self.block.conv.bias_scalar is not None:
            self.block.conv.bias_scalar.zero_()

        # ── W_vs: generate ζ and δ from each velocity level ──────────────────
        # W_vs shape: (n_s, K, K_v, 2)
        # Organised as (K, N_KINDS, K, K_v, 2) → [lev_out, kind_out, lev_in, kern, d]
        # Diagonal coupling: V_k feeds only level-k scalar channels.
        W_vs = self.block.conv.W_vs
        for k in range(K):
            W_vs[_ch(k, _ZETA_KIND), k, :, 0] = 1.0 / R_EARTH  # curl  → ζ_k
            W_vs[_ch(k, _DIV_KIND),  k, :, 1] = 1.0 / R_EARTH  # div   → δ_k

        # ── W_sv: PGF (lower-triangular in levels) and −∇KE (diagonal) ───────
        # W_sv shape: (K, n_s, K_v, 2)
        # Organised as (K, K, N_KINDS, K_v, 2) → [lev_out, lev_in, kind_in, kern, d]
        W_sv = self.block.conv.W_sv
        for k_out in range(K):
            # PGF: ∑_{k_in < k_out}  −R log(p_{k_in}/p_{k_in+1}) ∇T_{k_in}
            for k_in in range(k_out):
                W_sv[k_out, _ch(k_in, _T_KIND), :, 1] = (
                    -R * log_p_ratio[k_in].item() / R_EARTH
                )
            # −∇KE_k: diagonal coupling from KE channel at the same level
            W_sv[k_out, _ch(k_out, _KE_KIND), :, 1] = -1.0 / R_EARTH

        # ── pointwise_ss: only f passes through; T, KE, ζ, δ are zero ────────
        # (T and KE reach W_sv via x_scalar; they must not bleed into s used by
        # sv_product, so their pointwise rows stay zero.)
        self.block.pointwise_ss.weight.zero_()
        self.block.pointwise_ss.weight[f_ch, f_ch, 0, 0] = 1.0

        # ── Zero scalar MLP (residual init → identity when weights=0) ─────────
        for module in self.block.scalar_mlp:
            if hasattr(module, "weight"):
                module.weight.zero_()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.zero_()

        # ── sv_product: +ζ_k × V_k  (Lamb) and  −f × V_k  (Coriolis) ────────
        # sv_product weight shape: (n_s, n_v=K, 2)
        # With w_rotate=+1:  output += s * rot90(V)  where rot90(u,v)=(−v,u)
        # Convention matches PrimitiveEquationsStepper:
        #   +ζ rot90(V) = vorticity advection term    (weight = +1)
        #   −f rot90(V) = Coriolis force               (weight = −1)
        self.block.sv_product.weight.zero_()
        for k in range(K):
            self.block.sv_product.weight[_ch(k, _ZETA_KIND), k, 1] = +1.0
        self.block.sv_product.weight[f_ch, :, 1] = -1.0

        # ── Zero pointwise_vv (no direct vector skip needed) ─────────────────
        self.block.pointwise_vv.weight.zero_()

        # ── Gradient conv: physical gradient for T/q advection ───────────────
        self.grad_conv.W_sv.zero_()
        self.grad_conv.W_sv[0, 0, :, 1] = 1.0 / R_EARTH
        self.div_conv.W_vs.zero_()
        self.div_conv.W_vs[0, 0, :, 1] = 1.0 / R_EARTH

    # ── Differential operators ────────────────────────────────────────────────

    def _gradient(self, s: torch.Tensor) -> torch.Tensor:
        """∇s for (B, K, H, W) → (B, K, H, W, 2) using the standalone grad_conv."""
        B, K, H, W = s.shape
        s_flat = s.reshape(B * K, 1, H, W)
        dummy = s.new_zeros(B * K, 0, H, W, 2)
        _, g = self.grad_conv(s_flat, dummy)
        return g.reshape(B, K, H, W, 2)

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

    def _laplacian(self, s: torch.Tensor) -> torch.Tensor:
        """∇²s = ∇·(∇s) for (B, K, H, W) fields."""
        B, K, H, W = s.shape
        grad = self._gradient(s)  # (B, K, H, W, 2)
        uv_flat = grad.reshape(B * K, 1, H, W, 2)
        dummy_s = grad.new_zeros(B * K, 0, H, W)
        d_flat, _ = self.div_conv(dummy_s, uv_flat)
        return d_flat.reshape(B, K, H, W)

    # ── Main interface ────────────────────────────────────────────────────────

    def compute_tendencies(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time tendencies for all prognostic variables.

        The momentum tendency comes entirely from the block.  T and q
        tendencies are computed with conventional operators outside the block.

        Args:
            uv: velocity, shape (B, K, H, W, 2).
            T:  temperature, shape (B, K, H, W).
            q:  humidity, shape (B, K, H, W).

        Returns:
            (duv_dt, dT_dt, dq_dt) with the same shapes as the inputs.
        """
        B, K, H, W = T.shape

        # ── Precompute KE per level ───────────────────────────────────────────
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)  # (B, K, H, W)

        # ── Build block scalar input ──────────────────────────────────────────
        # Interleaved layout: [T_0, KE_0, 0, 0, T_1, KE_1, 0, 0, ..., f]
        zeros = torch.zeros_like(T)
        per_level = torch.stack(
            [T, ke, zeros, zeros], dim=2
        )  # (B, K, 4, H, W)
        x_s_body = per_level.reshape(B, K * _N_KINDS, H, W)
        f_exp = self.f_coriolis.expand(B, 1, H, W)
        x_s = torch.cat([x_s_body, f_exp], dim=1)  # (B, 4K+1, H, W)

        # ── Block forward: encodes PGF + Coriolis + ζ×V − ∇KE ───────────────
        y_s, y_v = self.block(x_s, uv)
        duv_dt = y_v - uv  # (B, K, H, W, 2)

        # ── Extract divergence from block scalar output ───────────────────────
        # x_s for ζ/δ channels was 0, so y_s[ζ/δ] = W_vs output = ζ_k / δ_k.
        y_s_per_level = y_s[:, : K * _N_KINDS].reshape(B, K, _N_KINDS, H, W)
        div = y_s_per_level[:, :, _DIV_KIND]  # (B, K, H, W)

        # ── ω from divergence ─────────────────────────────────────────────────
        omega = self._omega(div)  # (B, K, H, W)

        # ── Temperature tendency ──────────────────────────────────────────────
        grad_T = self._gradient(T)  # (B, K, H, W, 2)
        dT_dt = -(uv[..., 0] * grad_T[..., 0] + uv[..., 1] * grad_T[..., 1])
        p_k = self.pressure_levels.view(1, K, 1, 1)
        dT_dt = dT_dt + omega * (self.R / C_P * T / p_k - self._dT_dp(T))

        # ── Humidity tendency ─────────────────────────────────────────────────
        grad_q = self._gradient(q)
        dq_dt = -(uv[..., 0] * grad_q[..., 0] + uv[..., 1] * grad_q[..., 1])

        # ── Optional diffusion ────────────────────────────────────────────────
        if self.diffusion_coeff is not None:
            dT_dt = dT_dt + self._apply_diffusion(T)
            dq_dt = dq_dt + self._apply_diffusion(q)
            duv_dt = duv_dt + torch.stack(
                [self._apply_diffusion(uv[..., 0]),
                 self._apply_diffusion(uv[..., 1])],
                dim=-1,
            )

        return duv_dt, dT_dt, dq_dt

    def _apply_diffusion(self, s: torch.Tensor) -> torch.Tensor:
        lap = self._laplacian(s)
        for _ in range(self.diffusion_order - 1):
            lap = self._laplacian(lap)
        return (-1) ** (self.diffusion_order + 1) * self.diffusion_coeff * lap  # type: ignore[operator]

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
        k4 = self.compute_tendencies(
            uv + dt * k3[0], T + dt * k3[1], q + dt * k3[2]
        )
        c = dt / 6.0
        return (
            uv + c * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            T  + c * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            q  + c * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        )

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate a scalar field over the sphere."""
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))

    def total_kinetic_energy(self, uv: torch.Tensor) -> torch.Tensor:
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)
        return self.integrate_area(ke).sum(dim=1)
