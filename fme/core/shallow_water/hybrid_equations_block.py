"""Multi-level hydrostatic primitive equations in hybrid σ-p coordinates,
encoded in a single VectorDiscoBlock pass.

The same physics as ``HybridCoordinateStepper`` but with all horizontal
operations (dynamics, advection, diffusion) packed into one DISCO forward
pass.

Block  (n_s = 6K + 2,  n_v = K + 1,  residual = False)
--------------------------------------------------------
Per-level scalar channels — 6 kinds each:

    kind 0 — KE_k   kinetic energy (precomputed pointwise)
    kind 1 — φ_k    geopotential (precomputed via hydrostatic recurrence)
    kind 2 — ζ_k    vorticity   (zero input; filled by W_vs curl)
    kind 3 — δ_k    divergence  (zero input; filled by W_vs div)
    kind 4 — T_k    temperature (advected by W_vs2, diffused by W_ss)
    kind 5 — q_k    specific humidity (advected by W_vs2, diffused by W_ss)

Channel 6K   — f    Coriolis parameter (passes through pointwise_ss)
Channel 6K+1 — p_s  surface pressure (used by W_sv for ∇p_s)

Vector channels 0..K-1 — V_k  horizontal velocity at each level
Vector channel  K      — dedicated ∇p_s output (zero input; filled by W_sv)

Encodings
---------
W_vs:   V_k → ζ_k (curl) and δ_k (divergence)
W_sv:   φ_k, KE_k → -∇φ_k, -∇KE_k; p_s → +∇p_s
W_vs2:  -V_k·∇T_k, -V_k·∇q_k (horizontal advection)
W_ss:   ν∇²T_k, ν∇²q_k (scalar diffusion, approximate Laplacian)
W_vv:   ν∇²V_k (velocity diffusion, approximate Laplacian)
sv_product: ζ_k×V_k (vorticity advection), -f×V_k (Coriolis)

Block output (residual=False → raw tendencies)
----------------------------------------------
    y_v[:, :K]  =  -∇KE - ∇φ + ζ×V - f×V + ν∇²V    (partial momentum)
    y_v[:, K]   =  ∇p_s                               (gradient)
    y_s T_k     =  -V·∇T + ν∇²T                      (partial T tendency)
    y_s q_k     =  -V·∇q + ν∇²q                      (partial q tendency)
    y_s ζ_k/δ_k =  vorticity / divergence

Vertical linear layers  (frozen Conv2d(·, ·, 1))
-------------------------------------------------
hydrostatic_conv:  φ_k via cumulative sum of R T ln(p/p+1) increments
C_conv:            C_{k+1/2} via cumulative sum of per-level forcing
dX_dp_num_conv:    ∂X/∂p numerators via tridiagonal stencil

Explicit computations (pointwise, no DISCO)
-------------------------------------------
KE, p_k, dp_k, φ_k (inputs), V·∇p_s, ∂p_s/∂t, PGF correction,
ω, vertical advection, adiabatic T.

One DISCO pass suffices because all remaining explicit terms are either
pointwise nonlinearities with state-dependent coefficients, or vertical
operations handled by the frozen Conv2d layers.
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._disco_utils import _disco_s2_contraction_fft
from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.block import VectorDiscoBlock
from fme.core.shallow_water.scalar_vector_product import VectorDotProduct

R_EARTH = 6.371e6  # m
C_P = 1004.0  # J/kg/K

# Per-level scalar kind indices
_KE_KIND = 0
_PHI_KIND = 1
_ZETA_KIND = 2
_DIV_KIND = 3
_T_KIND = 4
_Q_KIND = 5
_N_KINDS = 6


def _legendre_p(degree: int, x: torch.Tensor) -> torch.Tensor:
    """Legendre polynomial P_l(x) via Bonnet's recurrence."""
    if degree == 0:
        return torch.ones_like(x)
    if degree == 1:
        return x.clone()
    p_prev = torch.ones_like(x)
    p_curr = x.clone()
    for i in range(1, degree):
        p_next = ((2 * i + 1) * x * p_curr - i * p_prev) / (i + 1)
        p_prev = p_curr
        p_curr = p_next
    return p_curr


def _compute_laplacian_weights(conv: VectorDiscoConvS2) -> torch.Tensor:
    """Compute scalar kernel weights approximating ∇² on the unit sphere.

    Uses Legendre polynomial test fields whose eigenvalues under the
    Laplace-Beltrami operator are known exactly:
        ∇² P_l(cos θ) = -l(l+1) P_l(cos θ).

    Returns a (K_s,) tensor *w* such that, for a single channel,
    ``einsum('k,...k...->...', w, scalar_basis_contractions)`` approximates
    the unit-sphere Laplacian of the input field.
    """
    K_s = conv._scalar_kernel_size
    H = conv.nlat_out
    W = conv.nlon_out
    device = conv.psi_scalar_fft.device

    colats = precompute_latitudes(H)[0].to(device)
    cos_colat = torch.cos(colats)  # (H,)

    n_tests = K_s + 2  # slightly overdetermined for robustness
    A_rows: list[torch.Tensor] = []
    b_rows: list[torch.Tensor] = []

    for degree in range(n_tests):
        p_l = _legendre_p(degree, cos_colat)  # (H,)
        field = p_l.reshape(1, 1, H, 1).expand(1, 1, H, W).contiguous()

        # Scalar basis contractions: (1, 1, K_s, H, W)
        f_s = _disco_s2_contraction_fft(
            field, conv.psi_scalar_fft, conv.scalar_gather_idx, W
        )
        # Average over longitude (isotropic → longitude-independent)
        f_s_avg = f_s[0, 0, :, :, :].mean(dim=-1).T  # (H, K_s)

        target = -degree * (degree + 1) * p_l  # (H,)

        A_rows.append(f_s_avg)
        b_rows.append(target)

    A = torch.cat(A_rows, dim=0)  # (n_tests*H, K_s)
    b = torch.cat(b_rows, dim=0)  # (n_tests*H,)

    return torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)


def _ch(level: int, kind: int) -> int:
    """Channel index for a given (level, kind) pair (interleaved layout)."""
    return level * _N_KINDS + kind


def isobaric_coefficients(
    pressure_levels: list[float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Convert isobaric pressure levels to hybrid coefficients with b=0.

    Interface pressures are midpoints between adjacent levels, with the
    surface interface at p[0] and the top interface at 0.

    Args:
        pressure_levels: pressure at each level (Pa), surface-to-top.

    Returns:
        ``(a_mid, b_mid, a_interface, b_interface)``
    """
    K = len(pressure_levels)
    p = pressure_levels
    a_mid = list(p)
    b_mid = [0.0] * K
    a_int: list[float] = [p[0]]
    for k in range(K - 1):
        a_int.append(0.5 * (p[k] + p[k + 1]))
    a_int.append(0.0)
    b_int = [0.0] * (K + 1)
    return a_mid, b_mid, a_int, b_int


class HybridCoordinateBlockStepper(nn.Module):
    """Multi-level primitive equations in hybrid σ-p coordinates using VectorDiscoBlock.

    Encodes all horizontal dynamics in a single DISCO forward pass:

    ``self.block`` (n_s = 6K+2, n_v = K+1):
        Per-level scalar channels (6 kinds per level):
            KE_k, φ_k, ζ_k, δ_k, T_k, q_k

        Plus global channels: f (Coriolis), p_s (surface pressure).

        Vector channels: V_0..V_{K-1} (wind), plus ∇p_s (channel K).

    Block encodings:
        W_vs   — V_k → ζ_k (curl) and δ_k (divergence)
        W_sv   — φ_k, KE_k → -∇φ_k, -∇KE_k; p_s → +∇p_s
        W_vs2  — -V_k·∇T_k, -V_k·∇q_k (horizontal advection)
        W_ss   — ν∇²T_k, ν∇²q_k (scalar diffusion via Laplacian weights)
        W_vv   — ν∇²V_k (velocity diffusion via Laplacian weights)
        sv_product — ζ_k×V_k (vorticity advection), -f×V_k (Coriolis)

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
    diffusion_coeff:
        Optional ν for ∇² diffusion (m²/s). None disables diffusion.
    isobaric:
        If True, treat as isobaric (fixed pressure levels, b=0).
        Skips vertical advection of V and q, PGF correction, and
        surface pressure tendency. Use ``isobaric_coefficients`` to
        convert pressure levels to hybrid coefficients.
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
        isobaric: bool = False,
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
        self.isobaric = isobaric

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

        # ── Block (n_s = 6K+2, n_v = K+1) ──────────────────────────────────
        # 6 kinds per level (KE, φ, ζ, δ, T, q), plus f and p_s.
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
            residual=False,
        )

        # ── ke_product: KE_k = ½|V_k|² via diagonal VectorDotProduct ────────
        self.ke_product = VectorDotProduct(n_scalar=K, n_vector=K)

        # ── Vertical linear layers (replace for-loops, frozen) ──────────────
        # Each is a 1×1 Conv2d acting pointwise over (H, W), mixing K levels.
        self.C_conv = nn.Conv2d(K, K + 1, 1, bias=False)
        if K >= 2:
            self.hydrostatic_conv = nn.Conv2d(K - 1, K, 1, bias=False)
            self.dX_dp_num_conv = nn.Conv2d(K, K, 1, bias=False)
        else:
            self.hydrostatic_conv = None
            self.dX_dp_num_conv = None

        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            self._init_physics_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_physics_weights(self) -> None:
        """Set all block and conv weights to encode the hybrid primitive equations."""
        K = self.n_levels
        f_ch = K * _N_KINDS  # channel index of the Coriolis parameter
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
        W_vs = self.block.conv.W_vs
        for k in range(K):
            W_vs[_ch(k, _ZETA_KIND), k, :, 0] = 1.0 / R_EARTH  # curl → ζ_k
            W_vs[_ch(k, _DIV_KIND), k, :, 1] = 1.0 / R_EARTH  # div  → δ_k

        # ── W_sv: -∇φ_k and -∇KE_k into velocity ch.; +∇p_s into ch. K ──────
        W_sv = self.block.conv.W_sv
        for k in range(K):
            W_sv[k, _ch(k, _PHI_KIND), :, 1] = -1.0 / R_EARTH  # -∇φ_k  → V_k
            W_sv[k, _ch(k, _KE_KIND), :, 1] = -1.0 / R_EARTH  # -∇KE_k → V_k
        W_sv[K, ps_ch, :, 1] = 1.0 / R_EARTH  # +∇p_s → dedicated channel K

        # ── W_vs2: -V_k·∇T_k and -V_k·∇q_k (horizontal advection) ──────────
        assert self.block.conv.W_vs2 is not None
        W_vs2 = self.block.conv.W_vs2
        for k in range(K):
            W_vs2[_ch(k, _T_KIND), k, :, 1] = -1.0 / R_EARTH  # -V_k · ∇T_k
            W_vs2[_ch(k, _Q_KIND), k, :, 1] = -1.0 / R_EARTH  # -V_k · ∇q_k

        # ── W_ss / W_vv: diffusion via Laplacian weights ─────────────────────
        if self.diffusion_coeff is not None:
            lap_w = _compute_laplacian_weights(self.block.conv)
            nu_R2 = self.diffusion_coeff / R_EARTH**2
            # Scalar diffusion: ν∇²T_k and ν∇²q_k
            for k in range(K):
                self.block.conv.W_ss[_ch(k, _T_KIND), _ch(k, _T_KIND)] = nu_R2 * lap_w
                self.block.conv.W_ss[_ch(k, _Q_KIND), _ch(k, _Q_KIND)] = nu_R2 * lap_w
            # Vector diffusion: ν∇²V_k (stretch component d=0 only)
            for k in range(K):
                self.block.conv.W_vv[k, k, :, 0] = nu_R2 * lap_w

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
        assert self.block.sv_product is not None
        self.block.sv_product.weight.zero_()
        for k in range(K):
            self.block.sv_product.weight[_ch(k, _ZETA_KIND), k, 1] = +1.0
        self.block.sv_product.weight[f_ch, :K, 1] = -1.0  # Coriolis k=0..K-1

        # ── Zero pointwise_vv ────────────────────────────────────────────────
        assert self.block.pointwise_vv is not None
        self.block.pointwise_vv.weight.zero_()

        # ── ke_product: diagonal weight 0.5 → KE_k = ½|V_k|² ───────────────
        self.ke_product.weight.zero_()
        for k in range(K):
            self.ke_product.weight[k, k] = 0.5

        # ── Vertical linear layers ────────────────────────────────────────────
        W_C = self.C_conv.weight  # (K+1, K, 1, 1)
        W_C.zero_()
        for k in range(K):
            W_C[k + 1, : k + 1, 0, 0] = 1.0

        if self.hydrostatic_conv is not None:
            W_H = self.hydrostatic_conv.weight  # (K, K-1, 1, 1)
            W_H.zero_()
            for k in range(1, K):
                W_H[k, :k, 0, 0] = 1.0

        if self.dX_dp_num_conv is not None:
            W_dX = self.dX_dp_num_conv.weight  # (K, K, 1, 1)
            W_dX.zero_()
            W_dX[0, 0, 0, 0] = -1.0
            W_dX[0, 1, 0, 0] = +1.0  # bottom
            for k in range(1, K - 1):  # interior
                W_dX[k, k - 1, 0, 0] = -1.0
                W_dX[k, k + 1, 0, 0] = +1.0
            W_dX[K - 1, K - 2, 0, 0] = -1.0
            W_dX[K - 1, K - 1, 0, 0] = +1.0  # top

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
        """Geopotential via hydrostatic balance using hydrostatic_conv.

        φ_k = φ_surface + ∑_{j<k} R T_j ln(p_j/p_{j+1})

        The log-pressure ratio is spatially varying (hybrid coords), so it is
        computed pointwise; the cumulative sum is encoded in hydrostatic_conv.
        """
        B, K, H, W = T.shape
        if self.hydrostatic_conv is None:  # K == 1
            return T.new_full((B, 1, H, W), self.phi_surface)
        log_ratio = torch.log(p_k[:, :-1] / p_k[:, 1:])  # (B, K-1, H, W)
        increments = self.R * T[:, :-1] * log_ratio  # (B, K-1, H, W)
        return self.phi_surface + self.hydrostatic_conv(increments)

    def _C_interfaces(
        self,
        div: torch.Tensor,
        uv: torch.Tensor,
        grad_ps: torch.Tensor,
        dp_k: torch.Tensor,
        dp_s_dt: torch.Tensor,
    ) -> torch.Tensor:
        """C = (∂p/∂η)η̇ at K+1 interfaces via C_conv (lower-triangular cumsum).

        forcing_k = -Δb_k ∂p_s/∂t - Δp_k div_k - Δb_k V_k·∇p_s
        C_{k+1/2} = ∑_{j=0}^{k} forcing_j,   C_{-1/2} = 0  (row 0 of C_conv)
        """
        B, K, H, W = div.shape
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
        db = self.delta_b.view(1, K, 1, 1)
        forcing = (
            -db * dp_s_dt.unsqueeze(1) - dp_k * div - db * v_dot_gps
        )  # (B, K, H, W)
        return self.C_conv(forcing)  # (B, K+1, H, W)

    def _dX_dp(self, X: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        """∂X/∂p via dX_dp_num_conv (tridiagonal stencil) ÷ pointwise Δp.

        dX_dp_num_conv encodes the fixed finite-difference numerators; the
        pressure-difference denominators are state-dependent (hybrid coords)
        and are divided in pointwise after the conv.
        """
        B, K, H, W = X.shape
        if self.dX_dp_num_conv is None:  # K == 1
            return X.new_zeros(B, K, H, W)
        numerators = self.dX_dp_num_conv(X)  # (B, K, H, W)
        dp = p_k.new_empty(B, K, H, W)
        dp[:, 0] = p_k[:, 1] - p_k[:, 0]  # bottom one-sided
        dp[:, K - 1] = p_k[:, K - 1] - p_k[:, K - 2]  # top one-sided
        if K >= 3:
            dp[:, 1:-1] = p_k[:, 2:] - p_k[:, :-2]  # interior central
        return numerators / dp

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

        # ── KE, p_k, dp_k, φ_k ───────────────────────────────────────────────
        ke = self.ke_product(uv)  # (B, K, H, W)
        p_k = self._level_pressures(p_s)  # (B, K, H, W)
        dp_k = self._layer_dp(p_s)  # (B, K, H, W)
        phi = self._hydrostatic(T, p_k)  # (B, K, H, W)

        # ── Assemble block inputs ─────────────────────────────────────────────
        # Mean-subtract φ and p_s before they enter the block: W_sv computes
        # ∇φ and ∇p_s, and subtracting the spatial mean eliminates the
        # O(|field|·ε) error from the discrete gradient of a constant.
        phi_anom = phi - phi.mean(dim=(-2, -1), keepdim=True)
        ps_anom = p_s - p_s.mean(dim=(-2, -1), keepdim=True)
        zeros = torch.zeros_like(T)
        per_level = torch.stack([ke, phi_anom, zeros, zeros, T, q], dim=2)
        x_s_body = per_level.reshape(B, K * _N_KINDS, H, W)
        x_s = torch.cat(
            [
                x_s_body,
                self.f_coriolis.expand(B, 1, H, W),
                ps_anom.unsqueeze(1),
            ],
            dim=1,
        )  # (B, 6K+2, H, W)

        x_v = torch.cat(
            [
                uv,
                uv.new_zeros(B, 1, H, W, 2),
            ],
            dim=1,
        )  # (B, K+1, H, W, 2)

        # ── Single block pass → momentum, ζ, δ, ∇p_s, advection, diffusion ──
        y_s, y_v = self.block(x_s, x_v)

        # Extract per-level outputs from scalar channels
        y_s_levels = y_s[:, : K * _N_KINDS].reshape(B, K, _N_KINDS, H, W)
        div = y_s_levels[:, :, _DIV_KIND]  # (B, K, H, W)
        dT_dt = y_s_levels[:, :, _T_KIND]  # -V·∇T (+ ν∇²T if diffusion)
        dq_dt = y_s_levels[:, :, _Q_KIND]  # -V·∇q (+ ν∇²q if diffusion)

        # Vector outputs: momentum tendency and ∇p_s
        grad_ps = y_v[:, K]  # (B, H, W, 2)
        # duv includes: -∇KE - ∇φ + ζ×V - f×V (+ ν∇²V if diffusion)
        duv_dt = y_v[:, :K]  # (B, K, H, W, 2)

        # ── Vertical operations ────────────────────────────────────────────────
        if not self.isobaric:
            # Full hybrid: surface pressure tendency, C interfaces, PGF
            # correction, vertical advection of V and q.
            db = self.delta_b.view(1, K, 1, 1)
            v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
            dp_s_dt = (dp_k * div + db * v_dot_gps).sum(dim=1)  # (B, H, W)

            C_half = self._C_interfaces(div, uv, grad_ps, dp_k, dp_s_dt)
            C_k = 0.5 * (C_half[:, :-1] + C_half[:, 1:])  # (B, K, H, W)

            b_k = self.b_mid.view(1, K, 1, 1)
            pgf_corr = -(self.R * T * b_k / p_k).unsqueeze(-1) * grad_ps.unsqueeze(1)
            dV0_dp = self._dX_dp(uv[..., 0], p_k)
            dV1_dp = self._dX_dp(uv[..., 1], p_k)
            vert_adv_uv = torch.stack([-C_k * dV0_dp, -C_k * dV1_dp], dim=-1)
            duv_dt = duv_dt + pgf_corr + vert_adv_uv

            omega = b_k * (dp_s_dt.unsqueeze(1) + v_dot_gps) + C_k
        else:
            # Isobaric: omega from cumulative divergence, no V/q vertical
            # advection, no PGF correction, no surface pressure evolution.
            C_half = self.C_conv(-dp_k * div)
            omega = 0.5 * (C_half[:, :-1] + C_half[:, 1:])
            dp_s_dt = p_s.new_zeros(B, H, W)

        # ── ω and adiabatic T (common) ────────────────────────────────────────
        kappa = self.R / C_P
        dT_dp = self._dX_dp(T, p_k)
        dT_dt = dT_dt + omega * (kappa * T / p_k - dT_dp)

        if not self.isobaric:
            dq_dt = dq_dt - C_k * self._dX_dp(q, p_k)

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
