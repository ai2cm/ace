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

    def test_warm_column_drives_divergent_flow(self):
        """A warm temperature anomaly raises geopotential and drives outward flow.

        Starting from rest, a warm Gaussian anomaly at the surface level
        creates a positive geopotential anomaly. The pressure gradient force
        (-∇φ) should drive winds away from the warm centre, generating
        nonzero velocities after a time step.
        """
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE)  # (nlat, nlon)
        T_anomaly = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_anomaly
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        for _ in range(10):
            uv, T, q = stepper.step(uv, T, q, DT)

        assert uv.abs().max() > 1e-6, "warm anomaly should drive nonzero winds"

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

    def test_passive_tracer_tracks_temperature(self):
        """Humidity initialized equal to (T - T_mean) evolves like temperature.

        Since q and T obey the same advection equation (-V·∇s), if q = a*T + b
        initially, it should remain approximately equal to a*T + b for short
        integrations (the difference is zero to machine precision initially,
        and grows only through nonlinear effects from T driving the geopotential).

        For small anomalies the temperature perturbation is small enough
        that the T-driven geopotential changes are negligible, so q and T
        perturbations should track each other closely.
        """
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_pert = 1.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        q = T_pert.clone()  # q starts equal to T perturbation
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)

        for _ in range(20):
            uv, T, q = stepper.step(uv, T, q, DT)

        T_pert_final = T - T_BACKGROUND
        # q and T_pert should remain close since they obey the same equation
        diff = (q - T_pert_final).abs().max().item()
        assert diff < 0.1, (
            f"q and T perturbation diverged too quickly: max diff = {diff:.3e}"
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

    def test_temperature_advection_with_uniform_wind(self):
        """With uniform eastward wind and a T gradient, T is advected.

        A uniform horizontal velocity (same at all levels) should advect
        the temperature field so that the total T integral stays roughly
        constant (transport, not diffusion). We check that the temperature
        field actually changes and doesn't blow up.
        """
        stepper = _make_stepper(omega=0.0)  # no Coriolis to keep things simple
        B = 1
        bump = _gaussian_bump(*SHAPE)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        # Initialise with a small eastward wind
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 0.01  # small eastward u
        q = torch.zeros(B, N_LEVELS, *SHAPE)

        T_initial = T.clone()
        for _ in range(50):
            uv, T, q = stepper.step(uv, T, q, DT)

        # Temperature field should have changed (advected)
        assert (T - T_initial).abs().max() > 1e-4, "temperature should be advected"
        # But should remain finite
        assert torch.isfinite(T).all()
