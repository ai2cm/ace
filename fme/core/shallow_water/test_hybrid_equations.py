"""Tests for HybridCoordinateStepper.

The hybrid coordinate generalises sigma (a=0, b=σ) and should reproduce
sigma results exactly when those coefficients are used.  Tests also cover
hybrid-specific behaviour (spatially varying p_k, PGF correction scaling,
top-interface C=0).
"""

import math

import pytest
import torch

from fme.core.shallow_water.hybrid_equations import (
    HybridCoordinateStepper,
    cam_like_coefficients,
    sigma_coefficients,
)

SHAPE = (16, 32)
N_LEVELS = 3
DT = 60.0  # 1-minute time step (s)
T_BACKGROUND = 280.0
P_SURFACE = 100000.0


def _gaussian_bump(nlat, nlon, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=15.0):
    from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes

    colats = precompute_latitudes(nlat)[0].float()
    lons = precompute_longitudes(nlon).float()
    lat = (math.pi / 2.0 - colats).unsqueeze(1)
    lon = lons.unsqueeze(0)
    dlat = lat - math.radians(lat0_deg)
    dlon = lon - math.radians(lon0_deg)
    dist2 = dlat**2 + (dlon * torch.cos(lat)) ** 2
    return torch.exp(-dist2 / (2 * math.radians(sigma_deg) ** 2))


def _sigma_stepper(**kw) -> HybridCoordinateStepper:
    """Hybrid stepper configured as pure sigma (a=0, b=σ)."""
    import math as _math

    sigma_levels = [
        _math.exp(-k * _math.log(5.0) / max(1, N_LEVELS - 1)) for k in range(N_LEVELS)
    ]
    a_mid, b_mid, a_int, b_int = sigma_coefficients(sigma_levels)
    return HybridCoordinateStepper(
        shape=SHAPE,
        a_mid=a_mid,
        b_mid=b_mid,
        a_interface=a_int,
        b_interface=b_int,
        **kw,
    )


def _cam_stepper(**kw) -> HybridCoordinateStepper:
    """Hybrid stepper configured with CAM-like coefficients."""
    a_mid, b_mid, a_int, b_int = cam_like_coefficients(N_LEVELS)
    return HybridCoordinateStepper(
        shape=SHAPE,
        a_mid=a_mid,
        b_mid=b_mid,
        a_interface=a_int,
        b_interface=b_int,
        **kw,
    )


class TestHybridCoordinateStepper:
    def test_resting_uniform_state_has_zero_tendency(self):
        """V=0, uniform T, uniform p_s → all tendencies zero."""
        stepper = _sigma_stepper()
        B = 1
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        duv_dt, dT_dt, dq_dt, dp_s_dt = stepper.compute_tendencies(uv, T, q, p_s)

        assert duv_dt.abs().max() < 1e-6, "velocity tendency should be zero"
        assert dT_dt.abs().max() < 1e-6, "temperature tendency should be zero"
        assert dq_dt.abs().max() < 1e-6, "humidity tendency should be zero"
        assert dp_s_dt.abs().max() < 1e-6, "surface pressure tendency should be zero"

    def test_output_shapes_preserved(self):
        """step() returns tensors of the same shapes as the inputs."""
        stepper = _sigma_stepper()
        B = 2
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        uv_new, T_new, q_new, ps_new = stepper.step(uv, T, q, p_s, DT)

        assert uv_new.shape == uv.shape
        assert T_new.shape == T.shape
        assert q_new.shape == q.shape
        assert ps_new.shape == p_s.shape

    def test_coefficient_validation(self):
        """Mismatched coefficient lengths raise ValueError."""
        a_mid, b_mid, a_int, b_int = sigma_coefficients([1.0, 0.5, 0.2])
        with pytest.raises(ValueError):
            HybridCoordinateStepper(
                shape=SHAPE,
                a_mid=a_mid[:-1],
                b_mid=b_mid,  # wrong length
                a_interface=a_int,
                b_interface=b_int,
            )
        with pytest.raises(ValueError):
            HybridCoordinateStepper(
                shape=SHAPE,
                a_mid=a_mid,
                b_mid=b_mid,
                a_interface=a_int[:-1],
                b_interface=b_int,  # wrong length
            )

    def test_level_pressures_sigma_case(self):
        """p_k = b_k * p_s exactly when a_k = 0 (pure sigma)."""
        stepper = _sigma_stepper()
        B = 1
        p_s = torch.full((B, *SHAPE), P_SURFACE)
        p_k = stepper._level_pressures(p_s)  # (B, K, H, W)

        b_k = stepper.b_mid.view(1, N_LEVELS, 1, 1)
        expected = b_k * P_SURFACE
        assert (
            p_k - expected
        ).abs().max() < 1.0, "sigma case: p_k should equal b_k * P_SURFACE"

    def test_level_pressures_hybrid_vary_with_ps(self):
        """With a≠0, p_k changes less than p_s (a provides a pressure floor)."""
        stepper = _cam_stepper()
        B = 1
        p_s_low = torch.full((B, *SHAPE), 0.9 * P_SURFACE)
        p_s_high = torch.full((B, *SHAPE), 1.1 * P_SURFACE)

        p_k_low = stepper._level_pressures(p_s_low)
        p_k_high = stepper._level_pressures(p_s_high)

        dp_s = 0.2 * P_SURFACE
        # At each level, Δp_k = b_k * Δp_s ≤ Δp_s (since b_k ≤ 1)
        dp_k = (p_k_high - p_k_low).abs()
        b_k = stepper.b_mid.view(1, N_LEVELS, 1, 1)
        expected_dp_k = b_k.abs() * dp_s
        assert (
            dp_k - expected_dp_k
        ).abs().max() < 1.0, "p_k change should scale as b_k * Δp_s"
        # Upper levels (small b) change less than lower levels
        assert (
            dp_k[:, -1] < dp_k[:, 0]
        ).all(), "upper levels (small b) should change less than lower levels"

    def test_geopotential_increases_upward(self):
        """φ increases from bottom level to top level for warm columns."""
        stepper = _sigma_stepper()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        p_s = torch.full((B, *SHAPE), P_SURFACE)
        phi = stepper.geopotential(T, p_s)
        for k in range(1, N_LEVELS):
            assert (
                phi[:, k] > phi[:, k - 1]
            ).all(), f"φ should increase: level {k} not > level {k-1}"

    def test_geopotential_hybrid_vs_sigma(self):
        """Geopotential is identical for hybrid-sigma and pure-sigma steppers.

        When a=0 (b=σ), p_k = σ_k p_s and ln(p_{k-1}/p_k) = ln(σ_{k-1}/σ_k)
        which is the same as the sigma formula.  The two implementations must
        agree to floating-point precision.
        """
        import math as _math

        sigma_levels = [
            _math.exp(-k * _math.log(5.0) / max(1, N_LEVELS - 1))
            for k in range(N_LEVELS)
        ]
        from fme.core.shallow_water.sigma_equations import SigmaCoordinateStepper

        sig_stepper = SigmaCoordinateStepper(
            shape=SHAPE, n_levels=N_LEVELS, sigma_levels=sigma_levels
        )
        hyb_stepper = _sigma_stepper()

        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        phi_sig = sig_stepper.geopotential(T)
        phi_hyb = hyb_stepper.geopotential(T, p_s)

        assert (
            (phi_hyb - phi_sig).abs().max() < 1e-2
        ), "hybrid sigma geopotential should match SigmaCoordinateStepper"

    def test_C_top_interface_is_zero(self):
        """Top-interface C must be zero (no mass flux through the model lid)."""
        stepper = _cam_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        div = stepper._divergence(uv)
        assert div.abs().max() > 1e-8, "test requires nonzero divergence"

        grad_ps = stepper._gradient_2d(p_s)
        dp_k = stepper._layer_dp(p_s)
        db = stepper.delta_b.view(1, N_LEVELS, 1, 1)
        v_dot_gps = (uv * grad_ps.unsqueeze(1)).sum(-1)
        dp_s_dt = (dp_k * div + db * v_dot_gps).sum(dim=1)

        C_half = stepper._C_interfaces(div, uv, grad_ps, dp_k, dp_s_dt)

        top = C_half[:, -1]
        assert (
            top.abs().max() < 1e-4
        ), f"top-interface C should be zero: max = {top.abs().max():.3e}"

    def test_mass_conservation(self):
        """Global ∫ dp_s/dt dA = 0 (mass is conserved)."""
        stepper = _cam_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        _, _, _, dp_s_dt = stepper.compute_tendencies(uv, T, q, p_s)

        global_mean = stepper.integrate_area(dp_s_dt[0]).item()
        assert (
            abs(global_mean) < 1e3
        ), f"global dp_s/dt should be near zero: {global_mean:.2f} Pa·sr/s"

    def test_no_nans_after_many_steps(self):
        """Integration remains finite over 200 steps."""
        stepper = _cam_stepper(diffusion_coeff=1e5)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        for _ in range(200):
            uv, T, q, p_s = stepper.step(uv, T, q, p_s, DT)

        assert torch.isfinite(uv).all()
        assert torch.isfinite(T).all()
        assert torch.isfinite(p_s).all()

    def test_zero_humidity_invariant(self):
        """q = 0 remains zero: dq/dt = −V·∇q − C_k ∂q/∂p = 0 when q=0."""
        stepper = _cam_stepper(diffusion_coeff=None)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        for _ in range(50):
            uv, T, q, p_s = stepper.step(uv, T, q, p_s, DT)

        assert q.abs().max() < 1e-10, f"q should remain zero: {q.abs().max():.2e}"

    def test_pgf_correction_scales_with_b(self):
        """PGF correction −(RT b/p)∇p_s is proportional to b.

        At upper levels (small b_k), the PGF correction is smaller than at
        lower levels (b_k ≈ 1).  With a spatially varying p_s (nonzero ∇p_s)
        and V=0 (so only PGF drives momentum), the top-level duv/dt should be
        smaller in magnitude than the bottom-level duv/dt.
        """
        stepper = _cam_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, lat0_deg=0.0, lon0_deg=180.0, sigma_deg=20.0)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        # Spatially varying p_s creates ∇p_s ≠ 0
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 1000.0 * bump.unsqueeze(0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        duv_dt, _, _, _ = stepper.compute_tendencies(uv, T, q, p_s)

        # Top level (smallest b_k) should have smaller |PGF correction| than bottom
        top_mag = duv_dt[:, -1].norm(dim=-1).mean().item()
        bot_mag = duv_dt[:, 0].norm(dim=-1).mean().item()
        b_top = stepper.b_mid[-1].item()
        b_bot = stepper.b_mid[0].item()
        assert top_mag < bot_mag, (
            f"top-level PGF (b={b_top:.3f}) should be < bottom-level (b={b_bot:.3f}): "
            f"{top_mag:.3e} vs {bot_mag:.3e}"
        )

    def test_hybrid_sigma_matches_sigma_stepper(self):
        """With a=0, b=σ, full tendencies match SigmaCoordinateStepper exactly.

        The hybrid coordinate reduces to pure sigma when a_k = 0, b_k = σ_k.
        All four tendencies (uv, T, q, dp_s/dt) should agree to near float32
        precision between HybridCoordinateStepper and SigmaCoordinateStepper.
        """
        import math as _math

        sigma_levels = [
            _math.exp(-k * _math.log(5.0) / max(1, N_LEVELS - 1))
            for k in range(N_LEVELS)
        ]
        from fme.core.shallow_water.sigma_equations import SigmaCoordinateStepper

        sig_stepper = SigmaCoordinateStepper(
            shape=SHAPE,
            n_levels=N_LEVELS,
            sigma_levels=sigma_levels,
            omega=0.0,
            diffusion_coeff=None,
        )
        hyb_stepper = _sigma_stepper(omega=0.0, diffusion_coeff=None)

        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 3.0 * bump.unsqueeze(0).unsqueeze(0)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        duv_sig, dT_sig, dq_sig, dps_sig = sig_stepper.compute_tendencies(uv, T, q, p_s)
        duv_hyb, dT_hyb, dq_hyb, dps_hyb = hyb_stepper.compute_tendencies(uv, T, q, p_s)

        rtol = 1e-4  # float32-level tolerance

        def _relerr(a, b, name):
            scale = a.abs().max().clamp(min=1e-10)
            err = (a - b).abs().max() / scale
            assert err < rtol, f"{name}: relative error {err:.2e} exceeds {rtol}"

        _relerr(duv_hyb, duv_sig, "duv/dt")
        _relerr(dT_hyb, dT_sig, "dT/dt")
        _relerr(dq_hyb, dq_sig, "dq/dt")
        _relerr(dps_hyb, dps_sig, "dp_s/dt")

    def test_cam_coefficients_smoke(self):
        """CAM-like coefficients produce a runnable stepper with physical pressures."""
        a_mid, b_mid, a_int, b_int = cam_like_coefficients(N_LEVELS)
        stepper = HybridCoordinateStepper(
            shape=SHAPE,
            a_mid=a_mid,
            b_mid=b_mid,
            a_interface=a_int,
            b_interface=b_int,
        )
        B = 1
        p_s = torch.full((B, *SHAPE), P_SURFACE)
        p_k = stepper._level_pressures(p_s)

        # All pressures should be positive and below p_s
        assert (p_k > 0).all(), "all level pressures should be positive"
        assert (p_k <= p_s.unsqueeze(1)).all(), "level pressures should not exceed p_s"
        # Pressures should increase from top (index K-1) to surface (index 0)
        for k in range(1, N_LEVELS):
            assert (
                p_k[:, k] < p_k[:, k - 1]
            ).all(), f"pressure should decrease upward: level {k} not < level {k-1}"
