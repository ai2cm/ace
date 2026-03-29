"""Multi-level hydrostatic primitive equations in sigma (σ = p/p_s) coordinates.

Compared to the isobaric ``PrimitiveEquationsStepper``, the key differences are:

* **Surface pressure** ``p_s(λ, φ, t)`` is a prognostic variable.  Its tendency
  is the vertically-integrated mass-flux divergence::

      ∂p_s/∂t = -∑_k  ∇·(p_s V_k)  Δσ_k

* **Pressure gradient force** includes a correction for sloping sigma surfaces::

      PGF_k = -∇φ_k  -  R T_k ∇(ln p_s)

  The first term is identical to the isobaric case; the second is new and gives
  the bottom level non-trivial dynamics even over a flat surface.

* **Sigma vertical velocity** σ̇ = dσ/dt (diagnostic, from continuity)::

      (p_s σ̇)_{k+1/2} = (p_s σ̇)_{k-1/2}  -  ∇·(p_s V_k)  Δσ_k

* **Temperature** evolves via the full dry-adiabatic equation::

      ∂T/∂t = -V·∇T  +  σ̇(κT/σ - ∂T/∂σ)  +  (κT/p_s)(∂p_s/∂t + V·∇p_s)

  The last term couples the temperature tendency to the surface-pressure
  tendency (evaluated at the same RK4 stage).

* **Velocity** gains vertical advection::

      ∂V/∂t = -∇φ - R T ∇(ln p_s) - f×V - ∇KE + ζ×V  -  σ̇ ∂V/∂σ

Geopotential hydrostatics are unchanged from isobaric coordinates because
ln(σ_{k-1}/σ_k) = ln(p_{k-1}/p_k) (p_s cancels in the ratio).

Pressure levels (Pa) at each grid point are p_k = σ_k · p_s(λ, φ, t).
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct

R_EARTH = 6.371e6  # m
C_P = 1004.0       # J/kg/K, dry air specific heat at constant pressure


class SigmaCoordinateStepper(nn.Module):
    """Multi-level hydrostatic primitive equations in σ = p/p_s coordinates.

    State: (uv, T, q, p_s)
        uv:  (B, K, H, W, 2) — horizontal velocity (m/s)
        T:   (B, K, H, W)    — temperature (K)
        q:   (B, K, H, W)    — specific humidity (passive tracer)
        p_s: (B, H, W)       — surface pressure (Pa)

    Sigma levels σ_k ∈ (0, 1] are fixed.  σ_0 = 1 is the surface level,
    σ_{K-1} is the top level.  The actual pressure at each grid point is
    p_k(λ,φ,t) = σ_k · p_s(λ,φ,t).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        n_levels: int = 5,
        sigma_levels: list[float] | None = None,
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
            shape: (nlat, nlon) grid dimensions.
            n_levels: number of sigma levels K.
            sigma_levels: σ at each level, ordered surface-to-top (σ₀=1 to
                σ_{K-1}>0).  Defaults to K levels log-spaced from 1 to 0.2.
            R: gas constant for dry air (J/kg/K).
            omega: Earth rotation rate (rad/s).
            phi_surface: surface geopotential φ_s (m²/s²).  For a flat
                Earth this is a constant; it enters the geopotential but
                cancels in the pressure gradient when flat.
            kernel_shape: DISCO kernel shape for differential operators.
            theta_cutoff: filter radius (rad). Defaults to 2 grid spacings.
            diffusion_coeff: ν for ∇^(2n) diffusion (m²ⁿ/s). None = off.
            diffusion_order: order n; 1 → ∇², 2 → ∇⁴.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.n_levels = n_levels
        self.R = R
        self.phi_surface = phi_surface
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_order = diffusion_order

        if sigma_levels is None:
            sigma_levels = [
                math.exp(-k * math.log(5.0) / max(1, n_levels - 1))
                for k in range(n_levels)
            ]  # σ from 1.0 (surface) to 0.2 (top)
        if len(sigma_levels) != n_levels:
            raise ValueError(
                f"len(sigma_levels)={len(sigma_levels)} != n_levels={n_levels}"
            )

        sig = torch.tensor(sigma_levels, dtype=torch.float32)
        self.register_buffer("sigma_levels", sig)

        # ln(σ_k / σ_{k+1}) for k = 0..K-2, shape (K-1,)
        # Same as ln(p_k/p_{k+1}) since p_s cancels — used in hydrostatics.
        if n_levels > 1:
            log_sig_ratio = torch.log(sig[:-1] / sig[1:])
        else:
            log_sig_ratio = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("log_sig_ratio", log_sig_ratio)

        # Layer thicknesses Δσ_k = σ_{k-1/2} - σ_{k+1/2}, shape (K,)
        # Interface sigma:
        #   σ_{-1/2}   = σ[0]   (surface boundary; σ̇ = 0 here)
        #   σ_{k+1/2}  = (σ[k] + σ[k+1]) / 2   for k = 0..K-2
        #   σ_{K-1/2}  = 0      (top boundary)
        if n_levels > 1:
            sig_lower = torch.cat([sig[:1], 0.5 * (sig[:-1] + sig[1:])])
            sig_upper = torch.cat([0.5 * (sig[:-1] + sig[1:]), sig.new_zeros(1)])
        else:
            sig_lower = sig[:1]
            sig_upper = sig.new_zeros(1)
        delta_sig = (sig_lower - sig_upper).float()
        self.register_buffer("delta_sigma", delta_sig)

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        # Coriolis parameter f = 2Ω sin φ, shape (1, 1, nlat, 1)
        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))

        area = (quad_weights * 2.0 * math.pi / self.nlon).float()
        self.register_buffer("area_weights", area)

        # ── DISCO differential operators ──────────────────────────────────────
        disco_kw = dict(
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # Gradient: scalar → vector  (∇φ, ∇T, ∇q, ∇ln(p_s), ∇KE, ...)
        self.grad_conv = VectorDiscoConvS2(
            in_channels_scalar=1, in_channels_vector=0,
            out_channels_scalar=0, out_channels_vector=1, **disco_kw
        )
        # Vorticity: vector → scalar  (curl for ζ×V Lamb form)
        self.vort_conv = VectorDiscoConvS2(
            in_channels_scalar=0, in_channels_vector=1,
            out_channels_scalar=1, out_channels_vector=0, **disco_kw
        )
        # Divergence: vector → scalar  (∇·V for p_s tendency and σ̇)
        self.div_conv = VectorDiscoConvS2(
            in_channels_scalar=0, in_channels_vector=1,
            out_channels_scalar=1, out_channels_vector=0, **disco_kw
        )

        # ScalarVectorProduct for Coriolis and vorticity advection
        self.coriolis_product = ScalarVectorProduct(n_scalar=1, n_vector=1)
        self.vort_product = ScalarVectorProduct(n_scalar=1, n_vector=1)

        # Freeze all parameters
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
        """∇s for a (B, K, H, W) field → (B, K, H, W, 2)."""
        B, K, H, W = s.shape
        s_flat = s.reshape(B * K, 1, H, W)
        dummy_v = s.new_zeros(B * K, 0, H, W, 2)
        _, g = self.grad_conv(s_flat, dummy_v)
        return g.reshape(B, K, H, W, 2)

    def _gradient_2d(self, s: torch.Tensor) -> torch.Tensor:
        """∇s for a (B, H, W) field → (B, H, W, 2)."""
        B, H, W = s.shape
        s_4d = s.unsqueeze(1)           # (B, 1, H, W)
        return self._gradient(s_4d).squeeze(1)   # (B, H, W, 2)

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
        """∇²s = ∇·(∇s), (B, K, H, W) → (B, K, H, W)."""
        return self._divergence(self._gradient(s))

    def _apply_diffusion(self, s: torch.Tensor) -> torch.Tensor:
        """(-1)^(n+1) · ν · ∇^(2n) s."""
        lap = self._laplacian(s)
        for _ in range(self.diffusion_order - 1):
            lap = self._laplacian(lap)
        sign = (-1) ** (self.diffusion_order + 1)
        return sign * self.diffusion_coeff * lap  # type: ignore[operator]

    # ── Physics helpers ────────────────────────────────────────────────────

    def _hydrostatic(self, T: torch.Tensor) -> torch.Tensor:
        """Geopotential from T via hydrostatic balance.

        φ_0 = φ_surface
        φ_k = φ_{k-1} + R T_{k-1} ln(σ_{k-1}/σ_k)   k ≥ 1

        Identical to the isobaric formula because p_s cancels in ln(p_{k-1}/p_k).

        Args:
            T: (B, K, H, W)
        Returns:
            phi: (B, K, H, W)
        """
        B, K, H, W = T.shape
        phi = T.new_zeros(B, K, H, W)
        phi[:, 0] = self.phi_surface
        for k in range(1, K):
            phi[:, k] = phi[:, k - 1] + self.R * T[:, k - 1] * self.log_sig_ratio[k - 1]
        return phi

    def _sigma_dot(
        self,
        div: torch.Tensor,
        uv: torch.Tensor,
        grad_log_ps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute σ̇ = dσ/dt at each level (diagnostic from continuity).

        Integrates ∂(p_s σ̇)/∂σ = -∇·(p_s V) = -(p_s div + V·∇p_s)
        upward from σ̇ = 0 at the surface boundary.

        Since p_s appears on both sides, we compute (p_s σ̇) and divide:
            (p_s σ̇)_{k+1/2} = (p_s σ̇)_{k-1/2} - [p_s div_k + V_k·∇p_s] Δσ_k
        But we factor out p_s and work with (σ̇_half):
            σ̇_{k+1/2} = σ̇_{k-1/2} - [div_k + V_k·∇(ln p_s)] Δσ_k

        Args:
            div:          (B, K, H, W)    — ∇·V at each level
            uv:           (B, K, H, W, 2) — velocity
            grad_log_ps:  (B, H, W, 2)    — ∇(ln p_s)

        Returns:
            sigma_dot: (B, K, H, W)
        """
        B, K, H, W = div.shape
        # V_k · ∇(ln p_s): broadcast grad_log_ps over levels
        glps = grad_log_ps.unsqueeze(1)   # (B, 1, H, W, 2)
        v_dot_glps = (uv * glps).sum(-1)  # (B, K, H, W)

        sigma_dot_half = div.new_zeros(B, K + 1, H, W)
        # sigma_dot_half[:, 0] = 0  (surface BC: σ=1 fixed)
        for k in range(K):
            sigma_dot_half[:, k + 1] = (
                sigma_dot_half[:, k]
                - (div[:, k] + v_dot_glps[:, k]) * self.delta_sigma[k]
            )
        return 0.5 * (sigma_dot_half[:, :-1] + sigma_dot_half[:, 1:])

    def _dX_dsigma(self, X: torch.Tensor) -> torch.Tensor:
        """∂X/∂σ by finite differences (central interior, one-sided boundaries).

        Args:
            X: (B, K, H, W)
        Returns:
            dX_dsig: (B, K, H, W)
        """
        sig = self.sigma_levels  # (K,)
        B, K, H, W = X.shape
        dX = X.new_zeros(B, K, H, W)
        if K == 1:
            return dX
        # Interior
        for k in range(1, K - 1):
            dX[:, k] = (X[:, k + 1] - X[:, k - 1]) / (sig[k + 1] - sig[k - 1])
        # Boundaries
        dX[:, 0]     = (X[:, 1] - X[:, 0])     / (sig[1]     - sig[0])
        dX[:, K - 1] = (X[:, K - 1] - X[:, K - 2]) / (sig[K - 1] - sig[K - 2])
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
            uv:  velocity,         (B, K, H, W, 2)
            T:   temperature,      (B, K, H, W)
            q:   humidity,         (B, K, H, W)
            p_s: surface pressure, (B, H, W)

        Returns:
            (duv_dt, dT_dt, dq_dt, dp_s_dt) — same shapes as inputs.
        """
        B, K, H, W = T.shape

        # ── Geopotential ──────────────────────────────────────────────────
        phi = self._hydrostatic(T)  # (B, K, H, W)

        # ── ∇(ln p_s) for pressure-gradient correction and σ̇ ─────────────
        log_ps = torch.log(p_s)                        # (B, H, W)
        grad_log_ps = self._gradient_2d(log_ps)        # (B, H, W, 2)

        # ── Divergence and σ̇ ─────────────────────────────────────────────
        div = self._divergence(uv)                     # (B, K, H, W)
        sigma_dot = self._sigma_dot(div, uv, grad_log_ps)  # (B, K, H, W)

        # ── Surface pressure tendency ─────────────────────────────────────
        # ∂p_s/∂t = -p_s ∑_k [div_k + V_k·∇(ln p_s)] Δσ_k
        glps_bcast = grad_log_ps.unsqueeze(1)                   # (B,1,H,W,2)
        v_dot_glps = (uv * glps_bcast).sum(-1)                  # (B,K,H,W)
        dp_s_dt_integrand = (div + v_dot_glps) * self.delta_sigma.view(1, K, 1, 1)
        dp_s_dt = -p_s * dp_s_dt_integrand.sum(dim=1)           # (B, H, W)

        # ── Pressure gradient force ───────────────────────────────────────
        # PGF_k = -∇φ_k  -  R T_k ∇(ln p_s)
        grad_phi = self._gradient(phi)                 # (B, K, H, W, 2)
        # R T_k ∇(ln p_s): broadcast over levels
        RT_k = (self.R * T).unsqueeze(-1)              # (B, K, H, W, 1)
        pgf_correction = RT_k * glps_bcast             # (B, K, H, W, 2)
        pgf = -grad_phi - pgf_correction               # (B, K, H, W, 2)

        # ── Momentum: Lamb form + Coriolis + vertical advection ───────────
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)  # (B, K, H, W)
        grad_ke = self._gradient(ke)                   # (B, K, H, W, 2)

        vort = self._vorticity(uv)                     # (B, K, H, W)
        BK = B * K
        uv_flat   = uv.reshape(BK, 1, H, W, 2)
        vort_flat = vort.reshape(BK, 1, H, W)
        f_flat    = self.f_coriolis.expand(BK, 1, H, W)

        vort_adv  = self.vort_product(vort_flat, uv_flat).reshape(B, K, H, W, 2)
        coriolis  = self.coriolis_product(f_flat, uv_flat).reshape(B, K, H, W, 2)

        # Vertical advection of momentum: -σ̇ ∂V/∂σ
        dV_dsig   = self._dX_dsigma(uv[..., 0])       # (B, K, H, W)
        dV_dsig_v = self._dX_dsigma(uv[..., 1])
        vert_adv_uv = torch.stack([-sigma_dot * dV_dsig,
                                    -sigma_dot * dV_dsig_v], dim=-1)

        duv_dt = pgf - coriolis - grad_ke + vort_adv + vert_adv_uv

        # ── Temperature ───────────────────────────────────────────────────
        # ∂T/∂t = -V·∇T + σ̇(κT/σ - ∂T/∂σ) + (κT/p_s)(∂p_s/∂t + V·∇p_s)
        grad_T = self._gradient(T)                     # (B, K, H, W, 2)
        horiz_adv_T = -(uv[..., 0] * grad_T[..., 0] + uv[..., 1] * grad_T[..., 1])

        sig_k = self.sigma_levels.view(1, K, 1, 1)
        kappa = self.R / C_P
        dT_dsig = self._dX_dsigma(T)
        vert_T = sigma_dot * (kappa * T / sig_k - dT_dsig)

        # (κT/p_s)(∂p_s/∂t + V·∇p_s)
        grad_ps = grad_log_ps * p_s.unsqueeze(-1)      # (B, H, W, 2)
        v_dot_grad_ps = (uv * grad_ps.unsqueeze(1)).sum(-1)  # (B, K, H, W)
        ps_term = (kappa * T / p_s.unsqueeze(1)) * (dp_s_dt.unsqueeze(1) + v_dot_grad_ps)

        dT_dt = horiz_adv_T + vert_T + ps_term

        # ── Humidity ─────────────────────────────────────────────────────
        grad_q = self._gradient(q)
        dq_dt = (-(uv[..., 0] * grad_q[..., 0] + uv[..., 1] * grad_q[..., 1])
                 - sigma_dot * self._dX_dsigma(q))

        # ── Optional diffusion ────────────────────────────────────────────
        if self.diffusion_coeff is not None:
            dT_dt  = dT_dt  + self._apply_diffusion(T)
            dq_dt  = dq_dt  + self._apply_diffusion(q)
            duv_dt = duv_dt + torch.stack(
                [self._apply_diffusion(uv[..., 0]),
                 self._apply_diffusion(uv[..., 1])], dim=-1)

        return duv_dt, dT_dt, dq_dt, dp_s_dt

    def step(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        p_s: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one time step with RK4.

        Args:
            uv:  (B, K, H, W, 2)
            T:   (B, K, H, W)
            q:   (B, K, H, W)
            p_s: (B, H, W)
            dt:  time step (s)

        Returns:
            uv_new, T_new, q_new, p_s_new
        """
        def _tend(uv_, T_, q_, ps_):
            return self.compute_tendencies(uv_, T_, q_, ps_)

        k1 = _tend(uv, T, q, p_s)
        k2 = _tend(uv + 0.5*dt*k1[0], T + 0.5*dt*k1[1],
                   q  + 0.5*dt*k1[2], p_s + 0.5*dt*k1[3])
        k3 = _tend(uv + 0.5*dt*k2[0], T + 0.5*dt*k2[1],
                   q  + 0.5*dt*k2[2], p_s + 0.5*dt*k2[3])
        k4 = _tend(uv + dt*k3[0],     T + dt*k3[1],
                   q  + dt*k3[2],     p_s + dt*k3[3])

        c = dt / 6.0
        uv_new  = uv  + c * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        T_new   = T   + c * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        q_new   = q   + c * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        ps_new  = p_s + c * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        return uv_new, T_new, q_new, ps_new

    def geopotential(self, T: torch.Tensor) -> torch.Tensor:
        """Geopotential at each level from temperature (convenience wrapper)."""
        return self._hydrostatic(T)

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate scalar field over the sphere, shape (..., nlat, nlon)."""
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))
