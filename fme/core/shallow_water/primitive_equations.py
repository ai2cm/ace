"""Multi-level hydrostatic primitive equations on the sphere.

Implements the hydrostatic primitive equations in isobaric (fixed pressure
level) coordinates. Evolves:

  - horizontal velocity V = (u, v) at each level  [vector, prognostic]
  - temperature T at each level                    [scalar, prognostic]
  - specific humidity q at each level              [scalar, passive tracer]

Equations at each level k:

    dV_k/dt = -∇φ_k - f×V_k - ∇(|V_k|²/2) + ζ_k×V_k
    dT_k/dt = -V_k · ∇T_k + ω_k · (κ·T_k/p_k − ∂T_k/∂p)
    dq_k/dt = -V_k · ∇q_k

Geopotential (hydrostatic, diagnostic from T):

    φ_0 = φ_surface
    φ_k = φ_{k-1} + R * T_{k-1} * ln(p_{k-1} / p_k)    for k ≥ 1

where pressure levels p are ordered from surface (p_0, highest) to top
(p_{K-1}, lowest), and κ = R/c_p ≈ 0.286.

The temperature equation is the full dry-adiabatic thermodynamic equation
derived from dθ/dt = 0 (conservation of potential temperature):

    ∂T/∂t = -V·∇T + ω·(κ·T/p − ∂T/∂p)

where ω = dp/dt is the pressure vertical velocity (diagnostic, from
continuity: ∂ω/∂p = -∇·V, integrated upward from ω=0 at the surface).
The term κ·T/p·ω is adiabatic (de)compression heating/cooling, and
-ω·∂T/∂p is vertical temperature advection.

The momentum advection uses the Lamb form: (V·∇)V = ∇(|V|²/2) − ζ×V,
where ζ = curl(V) is the vertical vorticity. This form separates into
a gradient term (handled by the scalar→vector operator) and a vorticity
rotation term (handled by ScalarVectorProduct).

Humidity uses the advective form −V·∇q (passive tracer, no vertical
coupling).
"""

import math

import torch
import torch.nn as nn

from fme.core.disco._quadrature import precompute_latitudes
from fme.core.disco._vector_convolution import VectorDiscoConvS2
from fme.core.shallow_water.scalar_vector_product import ScalarVectorProduct

R_EARTH = 6.371e6  # m, Earth's mean radius used to convert angular → physical gradients
C_P = 1004.0  # J/kg/K, dry air specific heat at constant pressure


class PrimitiveEquationsStepper(nn.Module):
    """Multi-level hydrostatic primitive equations stepper on the sphere.

    Evolves velocity, temperature, and humidity on K fixed pressure levels
    using the hydrostatic primitive equations. Uses VectorDiscoConvS2 with
    frozen physics weights for the differential operators (gradient, vorticity)
    and ScalarVectorProduct for Coriolis and vorticity advection. Time
    integration uses RK4.

    Temperature drives the pressure gradient force via the hydrostatic
    geopotential. Humidity is a passive tracer advected by the velocity.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        n_levels: int = 3,
        pressure_levels: list[float] | None = None,
        g: float = 9.81,
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
            n_levels: number of pressure levels K.
            pressure_levels: pressure at each level in Pa, ordered from
                surface (highest) to top (lowest). Defaults to K levels
                log-spaced from 100000 Pa to 10000 Pa.
            g: gravitational acceleration (m/s²), used for reference only.
            R: gas constant for dry air (J/kg/K).
            omega: Earth's rotation rate (rad/s).
            phi_surface: surface geopotential (m²/s²), default 0.
            kernel_shape: radial filter kernel shape for differential ops.
            theta_cutoff: filter support radius in radians. Defaults to
                2 grid spacings.
            diffusion_coeff: kinematic diffusion coefficient ν (m²ⁿ/s) for
                ∇^(2n) diffusion applied to V, T, and q.  None disables
                diffusion entirely (default).  A positive value enables
                scale-selective damping; combine with diffusion_order to
                control the degree of scale-selectivity.  Rule of thumb for
                order-1 (∇²) at grid spacing Δx with e-folding time τ at the
                2Δx scale: ν ≈ Δx² / (π² τ).
            diffusion_order: order n of the diffusion operator ∇^(2n).
                1 → ∇² (Laplacian, least scale-selective),
                2 → ∇⁴ (biharmonic, more scale-selective).  Default 1.
        """
        super().__init__()
        self.nlat, self.nlon = shape
        self.n_levels = n_levels
        self.g = g
        self.R = R
        self.phi_surface = phi_surface
        self.diffusion_coeff = diffusion_coeff
        self.diffusion_order = diffusion_order

        if pressure_levels is None:
            pressure_levels = [
                100000.0 * math.exp(-k * math.log(10.0) / max(1, n_levels - 1))
                for k in range(n_levels)
            ]
        if len(pressure_levels) != n_levels:
            raise ValueError(
                f"len(pressure_levels)={len(pressure_levels)} != n_levels={n_levels}"
            )

        p = torch.tensor(pressure_levels, dtype=torch.float32)
        self.register_buffer("pressure_levels", p)

        # ln(p[k] / p[k+1]) for k = 0..K-2, shape (K-1,)
        # Used in the hydrostatic geopotential integration
        if n_levels > 1:
            log_p_ratio = torch.log(p[:-1] / p[1:])
        else:
            log_p_ratio = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("log_p_ratio", log_p_ratio)

        # Layer thicknesses Δp_k = p_{k-1/2} - p_{k+1/2}, shape (K,)
        # Interface pressures:
        #   p_{-1/2} = p[0]  (surface boundary; ω=0 here)
        #   p_{k+1/2} = (p[k] + p[k+1]) / 2   for k = 0..K-2
        #   p_{K-1/2} = 0    (top boundary)
        if n_levels > 1:
            p_lower = torch.cat([p[:1], 0.5 * (p[:-1] + p[1:])])  # shape (K,)
            p_upper = torch.cat([0.5 * (p[:-1] + p[1:]), p.new_zeros(1)])  # shape (K,)
        else:
            p_lower = p[:1]
            p_upper = p.new_zeros(1)
        delta_p = (p_lower - p_upper).float()  # shape (K,), positive
        self.register_buffer("delta_p", delta_p)

        if theta_cutoff is None:
            theta_cutoff = 2.0 * math.pi / (self.nlat - 1)

        # Coriolis parameter f = 2*omega*sin(lat), shape (1, 1, nlat, 1)
        colats, quad_weights = precompute_latitudes(self.nlat)
        lats = math.pi / 2.0 - colats
        f_coriolis = (2.0 * omega * torch.sin(lats)).float()
        self.register_buffer("f_coriolis", f_coriolis.reshape(1, 1, -1, 1))

        # Area weights for diagnostics, shape (nlat,)
        area = (quad_weights * 2.0 * math.pi / self.nlon).float()
        self.register_buffer("area_weights", area)

        # Gradient operator: 1 scalar → 1 vector
        # Used for: ∇φ_k, ∇(|V_k|²/2), ∇T_k, ∇q_k
        # Processes all K levels by treating (B*K) as the batch dimension.
        self.grad_conv = VectorDiscoConvS2(
            in_channels_scalar=1,
            in_channels_vector=0,
            out_channels_scalar=0,
            out_channels_vector=1,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # Vorticity operator: 1 vector → 1 scalar (curl component)
        # Used for: ζ_k = curl(V_k) in the Lamb-form nonlinear advection.
        self.vort_conv = VectorDiscoConvS2(
            in_channels_scalar=0,
            in_channels_vector=1,
            out_channels_scalar=1,
            out_channels_vector=0,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # Divergence operator: 1 vector → 1 scalar (divergence component)
        # W_vs[..., 1] gives horizontal divergence ∇·V (per ShallowWaterStepper
        # convention). Used to compute ω via the continuity equation.
        self.div_conv = VectorDiscoConvS2(
            in_channels_scalar=0,
            in_channels_vector=1,
            out_channels_scalar=1,
            out_channels_vector=0,
            in_shape=shape,
            out_shape=shape,
            kernel_shape=kernel_shape,
            theta_cutoff=theta_cutoff,
            bias=False,
        )

        # Coriolis: ScalarVectorProduct(f, V_k) → f×V_k per level
        self.coriolis_product = ScalarVectorProduct(n_scalar=1, n_vector=1)

        # Vorticity advection: ScalarVectorProduct(ζ_k, V_k) → ζ_k×V_k
        self.vort_product = ScalarVectorProduct(n_scalar=1, n_vector=1)

        # Freeze all parameters — this is a fixed-weight physics model
        for param in self.parameters():
            param.requires_grad = False

        with torch.no_grad():
            # Gradient: W_sv[out_v=0, in_s=0, :, 1] approximates ∇s in
            # angular units (per radian). Dividing by R_EARTH converts to
            # physical units (per metre) so the pressure gradient force
            # dV/dt = -∇φ has the correct m/s² dimension.
            self.grad_conv.W_sv.zero_()
            self.grad_conv.W_sv[0, 0, :, 1] = 1.0 / R_EARTH

            # Vorticity: the d=0 component of W_vs (the "div" variable in
            # the forward) gives the physical curl/vorticity operator, while
            # d=1 (the "curl" variable) gives divergence — as established by
            # the ShallowWaterStepper which uses W_vs[..., 1] for divergence.
            # The 1/R_EARTH factor converts angular vorticity (per radian)
            # to physical vorticity (1/s) when V is in m/s.
            self.vort_conv.W_vs.zero_()
            self.vort_conv.W_vs[0, 0, :, 0] = 1.0 / R_EARTH

            # Divergence: d=1 component of W_vs gives horizontal divergence
            # ∇·V (same 1/R_EARTH scaling as vorticity).
            self.div_conv.W_vs.zero_()
            self.div_conv.W_vs[0, 0, :, 1] = 1.0 / R_EARTH

            # Coriolis: weight[s=0, v=0, mode=1] = 1.0 gives the 90°
            # rotation f×V following the ShallowWaterStepper convention.
            self.coriolis_product.weight.zero_()
            self.coriolis_product.weight[0, 0, 1] = 1.0

            # Vorticity advection: same rotation convention as Coriolis
            self.vort_product.weight.zero_()
            self.vort_product.weight[0, 0, 1] = 1.0

    def _hydrostatic_integration(self, T: torch.Tensor) -> torch.Tensor:
        """Compute geopotential from temperature via hydrostatic balance.

        φ_0 = φ_surface
        φ_k = φ_{k-1} + R * T_{k-1} * ln(p_{k-1} / p_k)  for k ≥ 1

        Args:
            T: temperature at each level, shape (B, K, H, W).

        Returns:
            phi: geopotential at each level, shape (B, K, H, W).
        """
        B, K, H, W = T.shape
        phi = T.new_zeros(B, K, H, W)
        phi[:, 0] = self.phi_surface
        for k in range(1, K):
            phi[:, k] = phi[:, k - 1] + self.R * T[:, k - 1] * self.log_p_ratio[k - 1]
        return phi

    def _gradient(self, s: torch.Tensor) -> torch.Tensor:
        """Compute spherical gradient of a scalar field at all levels.

        Processes all (B, K) slices in a single convolution call by
        flattening (B, K) → (B*K) as the batch dimension.

        Args:
            s: scalar field, shape (B, K, H, W).

        Returns:
            grad_s: gradient vector, shape (B, K, H, W, 2).
        """
        B, K, H, W = s.shape
        s_flat = s.reshape(B * K, 1, H, W)
        dummy_v = s.new_zeros(B * K, 0, H, W, 2)
        _, grad_flat = self.grad_conv(s_flat, dummy_v)  # (B*K, 1, H, W, 2)
        return grad_flat.reshape(B, K, H, W, 2)

    def _vorticity(self, uv: torch.Tensor) -> torch.Tensor:
        """Compute vertical vorticity of a vector field at all levels.

        Args:
            uv: vector field, shape (B, K, H, W, 2).

        Returns:
            vort: vorticity, shape (B, K, H, W).
        """
        B, K, H, W, _ = uv.shape
        uv_flat = uv.reshape(B * K, 1, H, W, 2)
        dummy_s = uv.new_zeros(B * K, 0, H, W)
        vort_flat, _ = self.vort_conv(dummy_s, uv_flat)  # (B*K, 1, H, W)
        return vort_flat.reshape(B, K, H, W)

    def _divergence(self, uv: torch.Tensor) -> torch.Tensor:
        """Compute horizontal divergence ∇·V at all levels.

        Args:
            uv: vector field, shape (B, K, H, W, 2).

        Returns:
            div: divergence, shape (B, K, H, W).
        """
        B, K, H, W, _ = uv.shape
        uv_flat = uv.reshape(B * K, 1, H, W, 2)
        dummy_s = uv.new_zeros(B * K, 0, H, W)
        div_flat, _ = self.div_conv(dummy_s, uv_flat)  # (B*K, 1, H, W)
        return div_flat.reshape(B, K, H, W)

    def _omega(self, div: torch.Tensor) -> torch.Tensor:
        """Compute vertical pressure velocity ω = dp/dt at each level.

        Integrates the continuity equation ∂ω/∂p = -∇·V upward from the
        surface (ω = 0 at p_{-1/2} = p[0]) to obtain ω at half-level
        interfaces, then averages to level centres:

            ω_{k+1/2} = ω_{k-1/2} + div_k · Δp_k
            ω_k       = (ω_{k-1/2} + ω_{k+1/2}) / 2

        Args:
            div: horizontal divergence at each level, shape (B, K, H, W).

        Returns:
            omega: vertical pressure velocity at each level, shape (B, K, H, W).
        """
        B, K, H, W = div.shape
        omega_half = div.new_zeros(B, K + 1, H, W)  # ω_{k-1/2}; omega_half[:,0]=0
        for k in range(K):
            omega_half[:, k + 1] = omega_half[:, k] + div[:, k] * self.delta_p[k]
        return 0.5 * (omega_half[:, :-1] + omega_half[:, 1:])

    def _dT_dp(self, T: torch.Tensor) -> torch.Tensor:
        """Compute ∂T/∂p at each level by finite differences.

        Uses central differences at interior levels and one-sided
        differences at the surface and top boundaries.

        Args:
            T: temperature at each level, shape (B, K, H, W).

        Returns:
            dT_dp: shape (B, K, H, W).
        """
        p = self.pressure_levels  # shape (K,)
        B, K, H, W = T.shape
        dT_dp = T.new_zeros(B, K, H, W)
        if K == 1:
            return dT_dp
        # Interior levels: central differences
        for k in range(1, K - 1):
            dT_dp[:, k] = (T[:, k + 1] - T[:, k - 1]) / (p[k + 1] - p[k - 1])
        # Surface (k=0) and top (k=K-1): one-sided differences
        dT_dp[:, 0] = (T[:, 1] - T[:, 0]) / (p[1] - p[0])
        dT_dp[:, K - 1] = (T[:, K - 1] - T[:, K - 2]) / (p[K - 1] - p[K - 2])
        return dT_dp

    def _laplacian(self, s: torch.Tensor) -> torch.Tensor:
        """Spherical Laplacian ∇²s = ∇·(∇s), shape (B, K, H, W)."""
        return self._divergence(self._gradient(s))

    def _apply_diffusion(self, s: torch.Tensor) -> torch.Tensor:
        """Return the diffusion tendency (−1)^(n+1) · ν · ∇^(2n) s.

        The sign alternates so that the operator is always dissipative:
          n=1 (∇²): tendency = +ν ∇²s  (positive at minima → fills them in)
          n=2 (∇⁴): tendency = −ν ∇⁴s  (negative where field peaks)
        """
        lap = self._laplacian(s)
        for _ in range(self.diffusion_order - 1):
            lap = self._laplacian(lap)
        sign = (-1) ** (self.diffusion_order + 1)
        return sign * self.diffusion_coeff * lap  # type: ignore[operator]

    def compute_tendencies(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time tendencies from the primitive equations.

        Args:
            uv: velocity at each level, shape (B, K, H, W, 2).
            T: temperature at each level, shape (B, K, H, W).
            q: humidity at each level, shape (B, K, H, W).

        Returns:
            duv_dt: velocity tendency, shape (B, K, H, W, 2).
            dT_dt: temperature tendency, shape (B, K, H, W).
                Includes horizontal advection, adiabatic compression
                heating, and vertical temperature advection.
            dq_dt: humidity tendency, shape (B, K, H, W).
        """
        B, K, H, W = T.shape

        # Hydrostatic geopotential from temperature.
        # Subtract the spatial mean before differentiating: only ∇φ matters,
        # and mean-subtraction eliminates the O(|φ|·ε) error from discrete
        # gradient of a constant, where ε is the discretization error.
        phi = self._hydrostatic_integration(T)  # (B, K, H, W)
        phi = phi - phi.mean(dim=(-2, -1), keepdim=True)

        # Pressure gradient force: -∇φ_k per level
        grad_phi = self._gradient(phi)  # (B, K, H, W, 2)

        # Nonlinear momentum advection via Lamb form: (V·∇)V = ∇KE − ζ×V
        # where KE = |V|²/2 is the kinetic energy per unit mass.
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)  # (B, K, H, W)
        grad_ke = self._gradient(ke)  # (B, K, H, W, 2)

        vort = self._vorticity(uv)  # (B, K, H, W)
        vort_flat = vort.reshape(B * K, 1, H, W)
        uv_flat = uv.reshape(B * K, 1, H, W, 2)
        vort_adv_flat = self.vort_product(vort_flat, uv_flat)  # (B*K, 1, H, W, 2)
        vort_adv = vort_adv_flat.reshape(B, K, H, W, 2)

        # Coriolis force: f×V_k at each level
        f = self.f_coriolis.expand(B * K, 1, H, W)
        coriolis_flat = self.coriolis_product(f, uv_flat)  # (B*K, 1, H, W, 2)
        coriolis = coriolis_flat.reshape(B, K, H, W, 2)

        # Total momentum tendency:
        #   dV/dt = -∇φ - f×V - (V·∇)V
        #         = -∇φ - f×V - ∇KE + ζ×V
        duv_dt = -grad_phi - coriolis - grad_ke + vort_adv

        # Temperature: full dry-adiabatic thermodynamic equation
        # dT/dt = -V·∇T + ω·(κ·T/p − ∂T/∂p)
        # The first term is horizontal advection; the second combines
        # adiabatic (de)compression heating (κ·T/p·ω) and vertical
        # temperature advection (−ω·∂T/∂p). Derived from dθ/dt = 0.
        grad_T = self._gradient(T)  # (B, K, H, W, 2)
        dT_dt = -(uv[..., 0] * grad_T[..., 0] + uv[..., 1] * grad_T[..., 1])

        div = self._divergence(uv)  # (B, K, H, W)
        omega = self._omega(div)  # (B, K, H, W)
        p_k = self.pressure_levels.view(1, K, 1, 1)
        dT_dt = dT_dt + omega * (self.R / C_P * T / p_k - self._dT_dp(T))

        # Humidity advection: -V_k · ∇q_k (passive tracer)
        grad_q = self._gradient(q)  # (B, K, H, W, 2)
        dq_dt = -(uv[..., 0] * grad_q[..., 0] + uv[..., 1] * grad_q[..., 1])

        # Optional ∇^(2n) diffusion applied to all prognostic variables.
        if self.diffusion_coeff is not None:
            dT_dt = dT_dt + self._apply_diffusion(T)
            dq_dt = dq_dt + self._apply_diffusion(q)
            duv_dt = duv_dt + torch.stack(
                [self._apply_diffusion(uv[..., 0]), self._apply_diffusion(uv[..., 1])],
                dim=-1,
            )

        return duv_dt, dT_dt, dq_dt

    def step(
        self,
        uv: torch.Tensor,
        T: torch.Tensor,
        q: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one time step using RK4.

        Args:
            uv: velocity, shape (B, K, H, W, 2).
            T: temperature, shape (B, K, H, W).
            q: humidity, shape (B, K, H, W).
            dt: time step.

        Returns:
            uv_new, T_new, q_new with the same shapes as the inputs.
        """
        k1_uv, k1_T, k1_q = self.compute_tendencies(uv, T, q)
        k2_uv, k2_T, k2_q = self.compute_tendencies(
            uv + 0.5 * dt * k1_uv,
            T + 0.5 * dt * k1_T,
            q + 0.5 * dt * k1_q,
        )
        k3_uv, k3_T, k3_q = self.compute_tendencies(
            uv + 0.5 * dt * k2_uv,
            T + 0.5 * dt * k2_T,
            q + 0.5 * dt * k2_q,
        )
        k4_uv, k4_T, k4_q = self.compute_tendencies(
            uv + dt * k3_uv,
            T + dt * k3_T,
            q + dt * k3_q,
        )
        uv_new = uv + (dt / 6.0) * (k1_uv + 2 * k2_uv + 2 * k3_uv + k4_uv)
        T_new = T + (dt / 6.0) * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)
        q_new = q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        return uv_new, T_new, q_new

    def integrate_area(self, field: torch.Tensor) -> torch.Tensor:
        """Integrate a scalar field over the sphere.

        Args:
            field: shape (..., nlat, nlon)

        Returns:
            Integral value, shape (...) with the lat/lon dims removed.
        """
        return (field * self.area_weights.unsqueeze(-1)).sum(dim=(-2, -1))

    def total_kinetic_energy(self, uv: torch.Tensor) -> torch.Tensor:
        """Compute total kinetic energy summed over all levels.

        Args:
            uv: velocity, shape (B, K, H, W, 2).

        Returns:
            Total KE, shape (B,).
        """
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)  # (B, K, H, W)
        return self.integrate_area(ke).sum(dim=1)  # sum over levels → (B,)

    def geopotential(self, T: torch.Tensor) -> torch.Tensor:
        """Compute geopotential at each level from temperature.

        Convenience wrapper around _hydrostatic_integration.

        Args:
            T: temperature, shape (B, K, H, W).

        Returns:
            phi: geopotential, shape (B, K, H, W).
        """
        return self._hydrostatic_integration(T)
