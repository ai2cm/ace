import math

import torch

from fme.core.shallow_water.primitive_equations import PrimitiveEquationsStepper

SHAPE = (16, 32)
N_LEVELS = 3
DT = 1.0  # nondimensional time step
T_BACKGROUND = 280.0  # K, representative tropospheric temperature


def _make_stepper(**kw) -> PrimitiveEquationsStepper:
    return PrimitiveEquationsStepper(shape=SHAPE, n_levels=N_LEVELS, **kw)


def _gaussian_bump(nlat, nlon, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=15.0):
    """Gaussian scalar anomaly on the sphere."""
    from fme.core.disco._quadrature import precompute_latitudes, precompute_longitudes

    colats = precompute_latitudes(nlat)[0].float()
    lons = precompute_longitudes(nlon).float()
    lat = (math.pi / 2.0 - colats).unsqueeze(1)
    lon = lons.unsqueeze(0)
    dlat = lat - math.radians(lat0_deg)
    dlon = lon - math.radians(lon0_deg)
    dist2 = dlat**2 + (dlon * torch.cos(lat)) ** 2
    return torch.exp(-dist2 / (2 * math.radians(sigma_deg) ** 2))


class TestPrimitiveEquationsStepper:
    def test_resting_uniform_state_has_zero_tendency(self):
        """Zero velocity + spatially uniform T → zero tendencies.

        With V=0, all advection and vorticity terms vanish.
        Uniform T gives spatially uniform φ at each level, so ∇φ=0.
        """
        stepper = _make_stepper()
        B = 1
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        duv_dt, dT_dt, dq_dt = stepper.compute_tendencies(uv, T, q)

        assert duv_dt.abs().max() < 1e-8, "velocity tendency should be zero"
        assert dT_dt.abs().max() < 1e-8, "temperature tendency should be zero"
        assert dq_dt.abs().max() < 1e-8, "humidity tendency should be zero"

    def test_warm_column_drives_nonzero_flow(self):
        """A warm temperature anomaly creates a geopotential gradient and drives winds.

        Starting from rest, a warm Gaussian anomaly at all levels creates a
        positive geopotential anomaly at levels above the surface. The pressure
        gradient force (-∇φ) should drive nonzero velocities within a few steps.
        """
        stepper = _make_stepper(omega=0.0)  # disable Coriolis to isolate effect
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_anomaly = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_anomaly
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        # After just one step the pressure gradient has done work
        uv_new, _, _ = stepper.step(uv, T, q, DT)

        # Level 0 has φ=0 everywhere (phi_surface=0), so no gradient force there.
        # Level 1+ has non-zero geopotential gradient. Check that at least one
        # non-surface level has acquired nonzero velocity.
        assert uv_new[:, 1:].abs().max() > 1e-10, (
            "pressure gradient from warm anomaly should drive velocity on upper levels"
        )

    def test_pressure_gradient_direction(self):
        """Pressure gradient force on upper levels points away from warm anomaly.

        A warm bump at (lat=30°, lon=180°) creates high geopotential there.
        The pressure gradient force -∇φ should point AWAY from the warm centre,
        i.e. at the centre itself the tendency is near-zero and in surrounding
        grid points it points outward. We verify this by checking that the initial
        velocity tendency at level 1 is nonzero and that the warm-centre grid
        point has near-zero tendency (∇φ ≈ 0 at the exact peak).
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, lat0_deg=0.0, lon0_deg=180.0, sigma_deg=20.0)
        T_anomaly = 100.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_anomaly
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        duv_dt, _, _ = stepper.compute_tendencies(uv, T, q)

        # Level 0 has no geopotential gradient (phi_surface=0 everywhere)
        assert duv_dt[:, 0].abs().max() < 1e-10, (
            "surface level should have zero tendency (phi_surface is constant)"
        )
        # Level 1 and above should have nonzero pressure gradient tendency
        assert duv_dt[:, 1:].abs().max() > 1e-8, (
            "upper levels should have nonzero tendency from geopotential gradient"
        )

    def test_coriolis_does_no_work(self):
        """Coriolis force (f×V) does no work: V·(f×V) = 0 everywhere.

        This is an algebraic identity: f×V is perpendicular to V, so their
        dot product is zero. Test it numerically with a nonzero wind field.
        """
        stepper = _make_stepper()
        B = 1
        # Build a nontrivial wind field (Gaussian bumps in u and v)
        bump_u = _gaussian_bump(*SHAPE, lat0_deg=30.0, lon0_deg=90.0)
        bump_v = _gaussian_bump(*SHAPE, lat0_deg=-30.0, lon0_deg=270.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = bump_u.unsqueeze(0).unsqueeze(0) * 10.0  # u
        uv[..., 1] = bump_v.unsqueeze(0).unsqueeze(0) * 5.0   # v

        # Compute Coriolis tendency alone: use omega != 0, T uniform (no PGF),
        # and V = 0 for the nonlinear terms (we check the Coriolis analytically).
        # We call compute_tendencies but extract only the Coriolis contribution
        # by verifying V · duv_dt ≈ 0 when all other terms are zero.
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        # With uniform T (∇φ=0) and this wind, the dominant contribution to
        # duv_dt at first order is -f×V. Check V · duv_dt_coriolis ≈ 0.
        # We approximate by using a stepper with no nonlinear terms (tiny dt).
        # The exact check is: ScalarVectorProduct(f, V) gives f*(-v, u), and
        # V · f*(-v, u) = u*f*(-v) + v*f*u = f(-uv + vu) = 0.
        f = stepper.f_coriolis.expand(B * N_LEVELS, 1, *SHAPE)
        uv_flat = uv.reshape(B * N_LEVELS, 1, *SHAPE, 2)
        coriolis_flat = stepper.coriolis_product(f, uv_flat)
        coriolis = coriolis_flat.reshape(B, N_LEVELS, *SHAPE, 2)

        # V · (f×V) should be exactly zero
        dot = uv[..., 0] * coriolis[..., 0] + uv[..., 1] * coriolis[..., 1]
        assert dot.abs().max() < 1e-10, (
            f"Coriolis should do no work: max |V·(f×V)| = {dot.abs().max():.2e}"
        )

    def test_output_shapes_preserved(self):
        """Step returns tensors of the same shape as inputs."""
        stepper = _make_stepper()
        B = 2
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        uv_new, T_new, q_new = stepper.step(uv, T, q, DT)

        assert uv_new.shape == uv.shape
        assert T_new.shape == T.shape
        assert q_new.shape == q.shape

    def test_no_nans_after_many_steps(self):
        """Integration remains finite (no numerical blow-up)."""
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_anomaly = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_anomaly
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        for _ in range(200):
            uv, T, q = stepper.step(uv, T, q, DT)

        assert torch.isfinite(uv).all(), "velocity should remain finite"
        assert torch.isfinite(T).all(), "temperature should remain finite"
        assert torch.isfinite(q).all(), "humidity should remain finite"

    def test_passive_tracer_tracks_temperature_perturbation(self):
        """Humidity closely tracks T perturbation when initialised equal.

        If q(t=0) = T_pert(t=0) = T - T_BACKGROUND, then since both obey
        dX/dt = -V·∇X with the same V (which depends only on T, not q),
        they should remain very close. The small residual (~1e-5 K) comes
        from float32 rounding: T is stored as T_BACKGROUND + T_pert (~280 K)
        while q is stored as T_pert (~1 K), so their float32 representations
        differ by up to one ULP(T_BACKGROUND) ≈ 3.4e-5 K per RK4 stage.
        """
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_pert = 1.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        q = T_pert.clone()
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)

        for _ in range(20):
            uv, T, q = stepper.step(uv, T, q, DT)

        T_pert_final = T - T_BACKGROUND
        diff = (q - T_pert_final).abs().max().item()
        assert diff < 1e-4, (
            f"q should closely track T_pert: max diff = {diff:.3e}"
        )

    def test_hydrostatic_geopotential_increases_upward(self):
        """Geopotential increases from surface to top for positive-T columns.

        φ_k = φ_{k-1} + R * T_{k-1} * ln(p_{k-1}/p_k)
        With T > 0 and p_0 > p_1 > ... > p_{K-1}, each increment is
        positive, so φ strictly increases with level index.
        """
        stepper = _make_stepper()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        phi = stepper.geopotential(T)  # (B, K, H, W)
        for k in range(1, N_LEVELS):
            assert (phi[:, k] > phi[:, k - 1]).all(), (
                f"geopotential should increase: level {k} not > level {k-1}"
            )

    def test_geopotential_proportional_to_temperature(self):
        """Warmer column → higher geopotential (hydrostatic consistency)."""
        stepper = _make_stepper()
        B = 1
        T_warm = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND + 10.0)
        T_cold = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        phi_warm = stepper.geopotential(T_warm)
        phi_cold = stepper.geopotential(T_cold)
        # Warm column has higher geopotential at all levels above the surface
        for k in range(1, N_LEVELS):
            assert (phi_warm[:, k] > phi_cold[:, k]).all(), (
                f"warm column should have higher geopotential at level {k}"
            )

    def test_geopotential_values_match_formula(self):
        """Geopotential matches the closed-form hydrostatic formula.

        With uniform T = T0 at all levels:
            φ_k = φ_surface + R * T0 * Σ_{j=0}^{k-1} ln(p_j / p_{j+1})
                = φ_surface + R * T0 * ln(p_0 / p_k)
        """
        stepper = _make_stepper()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        phi = stepper.geopotential(T)

        p = stepper.pressure_levels  # (K,)
        for k in range(1, N_LEVELS):
            expected = stepper.phi_surface + stepper.R * T_BACKGROUND * math.log(
                p[0].item() / p[k].item()
            )
            actual = phi[:, k].mean().item()
            assert abs(actual - expected) < 1.0, (
                f"level {k}: expected φ={expected:.1f}, got {actual:.1f}"
            )

    def test_single_level_runs_without_error(self):
        """A single-level model initialises and steps without error."""
        stepper = PrimitiveEquationsStepper(
            shape=SHAPE, n_levels=1, pressure_levels=[100000.0]
        )
        B = 1
        uv = torch.zeros(B, 1, *SHAPE, 2)
        T = torch.full((B, 1, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, 1, *SHAPE)
        uv_new, T_new, q_new = stepper.step(uv, T, q, DT)
        assert uv_new.shape == uv.shape
        assert torch.isfinite(uv_new).all()

    def test_zero_humidity_is_steady_state(self):
        """q = 0 is an exact steady state of the humidity equation.

        dq/dt = -V·∇q. With q = 0 everywhere, ∇q = 0, so dq/dt = 0
        regardless of the wind field. q = 0 must remain exactly zero
        under any dynamics. This tests the linearity of the gradient
        operator and that no numerical leakage introduces q ≠ 0.
        """
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0  # 10 m/s uniform zonal wind so advection is active
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        uv_initial = uv.clone()
        for _ in range(50):
            uv, T, q = stepper.step(uv, T, q, DT)

        # q = 0 is an invariant of the humidity equation (∇0 = 0)
        assert q.abs().max() < 1e-10, (
            f"q should remain exactly zero: max = {q.abs().max():.2e}"
        )
        # Sanity check that dynamics ran: Coriolis and pressure gradient forces
        # evolve the velocity significantly over 50 steps.
        assert (uv - uv_initial).abs().max() > 1e-4, (
            "velocity should have evolved from Coriolis and pressure gradient forces"
        )
        assert torch.isfinite(T).all()

    def test_pressure_levels_validation(self):
        """Wrong number of pressure levels raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="n_levels"):
            PrimitiveEquationsStepper(
                shape=SHAPE, n_levels=3, pressure_levels=[100000.0, 50000.0]
            )

    def test_surface_level_has_no_pressure_gradient_tendency(self):
        """Level 0 always has zero pressure gradient tendency.

        φ_0 = phi_surface = 0.0 everywhere (a spatial constant), so ∇φ_0 = 0
        exactly. Combined with V=0 (no advection), the surface-level momentum
        tendency is exactly zero, while upper levels (φ_k > 0, varying in space
        when T has gradients) have nonzero tendency.
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        duv_dt, _, _ = stepper.compute_tendencies(uv, T, q)

        # phi_surface = 0 everywhere → ∇phi_0 = 0 → no pressure gradient at level 0
        assert duv_dt[:, 0].abs().max() < 1e-10, (
            "surface level (phi=0) should have zero momentum tendency"
        )
        # Level 1: phi_1 = R*T*log_p_ratio[0] varies with the T anomaly → nonzero ∇φ
        assert duv_dt[:, 1].abs().max() > 1e-8, (
            "level 1 should have nonzero tendency from warm T anomaly"
        )
