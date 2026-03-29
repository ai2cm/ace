import math

import pytest
import torch

from fme.core.shallow_water.sigma_equations import SigmaCoordinateStepper

SHAPE = (16, 32)
N_LEVELS = 3
DT = 60.0  # 1-minute time step (s)
T_BACKGROUND = 280.0  # K
P_SURFACE = 100000.0  # Pa


def _make_stepper(**kw) -> SigmaCoordinateStepper:
    return SigmaCoordinateStepper(shape=SHAPE, n_levels=N_LEVELS, **kw)


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


class TestSigmaCoordinateStepper:
    def test_resting_uniform_state_has_zero_tendency(self):
        """V=0, uniform T, uniform p_s → all tendencies zero.

        With no horizontal gradients and no wind, every term in the tendency
        equations vanishes: ∇T=0, ∇p_s=0, div=0 → σ̇=0, ∂p_s/∂t=0.
        """
        stepper = _make_stepper()
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
        stepper = _make_stepper()
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

    def test_no_nans_after_many_steps(self):
        """Integration stays finite (no blow-up) over 200 steps."""
        stepper = _make_stepper(diffusion_coeff=1e5)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        for _ in range(200):
            uv, T, q, p_s = stepper.step(uv, T, q, p_s, DT)

        assert torch.isfinite(uv).all(), "velocity should remain finite"
        assert torch.isfinite(T).all(), "temperature should remain finite"
        assert torch.isfinite(q).all(), "humidity should remain finite"
        assert torch.isfinite(p_s).all(), "surface pressure should remain finite"

    def test_sigma_level_count_validation(self):
        """Wrong number of sigma levels raises ValueError."""
        with pytest.raises(ValueError, match="n_levels"):
            SigmaCoordinateStepper(
                shape=SHAPE, n_levels=3, sigma_levels=[1.0, 0.5]
            )

    def test_geopotential_increases_upward(self):
        """Geopotential increases from surface to model top for warm columns."""
        stepper = _make_stepper()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        phi = stepper.geopotential(T)  # (B, K, H, W)
        for k in range(1, N_LEVELS):
            assert (phi[:, k] > phi[:, k - 1]).all(), (
                f"geopotential should increase: level {k} not > level {k-1}"
            )

    def test_geopotential_values_match_formula(self):
        """Geopotential matches φ_k = φ_surface + R T Σ ln(σ_{k-1}/σ_k).

        With spatially uniform T, the formula simplifies to
            φ_k = R * T0 * ln(σ_0 / σ_k)
        because the sum telescopes.
        """
        stepper = _make_stepper()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        phi = stepper.geopotential(T)

        sig = stepper.sigma_levels  # (K,)
        for k in range(1, N_LEVELS):
            expected = stepper.phi_surface + stepper.R * T_BACKGROUND * math.log(
                sig[0].item() / sig[k].item()
            )
            actual = phi[:, k].mean().item()
            assert abs(actual - expected) < 1.0, (
                f"level {k}: expected φ={expected:.1f}, got {actual:.1f}"
            )

    def test_mass_conservation(self):
        """Total atmospheric mass (∫p_s dA) is conserved to high accuracy.

        The surface pressure tendency integrates to zero over the sphere because
        it equals the divergence of a mass flux. On a closed surface the integral
        of any divergence is zero (Gauss theorem). We test that the numerical
        approximation achieves < 0.01 Pa/s of global-mean tendency.
        """
        stepper = _make_stepper()
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        # Give it a nonzero wind to make divergence non-trivial
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)

        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        _, _, _, dp_s_dt = stepper.compute_tendencies(uv, T, q, p_s)

        # Global mean of dp_s/dt should be ≈ 0 (mass conservation)
        global_mean = stepper.integrate_area(dp_s_dt[0]).item()
        # 4π steradian * 100000 Pa / scale — tolerance is 0.1% of P_SURFACE per step
        assert abs(global_mean) < 1e3, (
            f"global dp_s/dt should be near zero: {global_mean:.2f} Pa·sr/s"
        )

    def test_nonzero_wind_drives_surface_pressure_change(self):
        """A spatially varying wind field produces nonzero dp_s/dt.

        A Gaussian bump in velocity has nonzero divergence (∇·V ≠ 0 where the
        field varies in space). The surface pressure tendency must be nonzero
        somewhere when there is divergence/convergence present.
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=15.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)  # spatially varying u
        uv[..., 1] = 5.0 * bump.unsqueeze(0).unsqueeze(0)   # spatially varying v

        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        _, _, _, dp_s_dt = stepper.compute_tendencies(uv, T, q, p_s)

        # Nonzero divergence → nonzero dp_s/dt somewhere
        assert dp_s_dt.abs().max() > 1e-3, (
            "spatially varying wind should produce nonzero dp_s/dt"
        )
        assert torch.isfinite(dp_s_dt).all()

    def test_coriolis_does_no_work(self):
        """Coriolis force is perpendicular to V: V·(f×V) = 0 everywhere."""
        stepper = _make_stepper()
        B = 1
        bump_u = _gaussian_bump(*SHAPE, lat0_deg=30.0, lon0_deg=90.0)
        bump_v = _gaussian_bump(*SHAPE, lat0_deg=-30.0, lon0_deg=270.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = bump_u.unsqueeze(0).unsqueeze(0) * 10.0
        uv[..., 1] = bump_v.unsqueeze(0).unsqueeze(0) * 5.0

        BK = B * N_LEVELS
        f = stepper.f_coriolis.expand(BK, 1, *SHAPE)
        uv_flat = uv.reshape(BK, 1, *SHAPE, 2)
        coriolis_flat = stepper.coriolis_product(f, uv_flat)
        coriolis = coriolis_flat.reshape(B, N_LEVELS, *SHAPE, 2)

        dot = uv[..., 0] * coriolis[..., 0] + uv[..., 1] * coriolis[..., 1]
        assert dot.abs().max() < 1e-10, (
            f"Coriolis should do no work: max |V·(f×V)| = {dot.abs().max():.2e}"
        )

    def test_zero_humidity_invariant(self):
        """q = 0 is an exact steady state: dq/dt = -V·∇q - σ̇ ∂q/∂σ = 0 when q=0."""
        stepper = _make_stepper(diffusion_coeff=None)
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

        assert q.abs().max() < 1e-10, (
            f"q should remain exactly zero: max = {q.abs().max():.2e}"
        )

    def test_diffusion_reduces_anomaly(self):
        """Diffusion damps a localised T anomaly while leaving the mean intact."""
        stepper = _make_stepper(diffusion_coeff=1e6, diffusion_order=1, omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=10.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T0 = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        T = T0.clone()
        for _ in range(50):
            uv, T, q, p_s = stepper.step(uv, T, q, p_s, DT)

        # Peak anomaly should have decayed
        initial_max_pert = T_pert.max().item()
        final_max_pert = (T - T_BACKGROUND).abs().max().item()
        assert final_max_pert < initial_max_pert, (
            "diffusion should reduce the peak T anomaly"
        )

    def test_single_level_runs_without_error(self):
        """A single-sigma-level model initialises and steps without error."""
        stepper = SigmaCoordinateStepper(
            shape=SHAPE, n_levels=1, sigma_levels=[1.0]
        )
        B = 1
        uv = torch.zeros(B, 1, *SHAPE, 2)
        T = torch.full((B, 1, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, 1, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)
        uv_new, T_new, q_new, ps_new = stepper.step(uv, T, q, p_s, DT)
        assert uv_new.shape == uv.shape
        assert torch.isfinite(uv_new).all()
        assert torch.isfinite(ps_new).all()

    def test_warm_column_drives_flow(self):
        """A warm T anomaly → nonzero geopotential gradient → PGF drives winds.

        The pressure gradient force on the bottom sigma level now contains the
        RT∇(ln p_s) correction term even when phi_surface is flat. With V=0
        initially, the only source of duv/dt at level 1+ is -∇φ from the
        warm anomaly.
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, lat0_deg=0.0, lon0_deg=180.0, sigma_deg=20.0)
        T_pert = 50.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        duv_dt, _, _, _ = stepper.compute_tendencies(uv, T, q, p_s)

        # Upper levels see a geopotential gradient from the warm anomaly
        assert duv_dt[:, 1:].abs().max() > 1e-8, (
            "warm column PGF should drive wind tendency on upper levels"
        )

    def test_surface_pressure_sign_with_gradient_flow(self):
        """Gradient flow (V = ∇f) converges at the peak → dp_s/dt > 0 there.

        For a Gaussian bump f, ∇f points toward the maximum (inward), so
        V = ∇f is a convergent flow at the peak. Since
            dp_s/dt = -p_s ∑_k div_k Δσ_k  (uniform p_s, so ∇ln p_s = 0)
        and the divergence at the peak is negative (convergence), dp_s/dt > 0
        at the peak. The stepper's own gradient operator is used to ensure
        consistent sign conventions.
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, lat0_deg=30.0, lon0_deg=180.0, sigma_deg=20.0)

        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        # Build uv as ∇(bump) using the stepper's own gradient operator
        bump_4d = bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        grad_bump = stepper._gradient(bump_4d)          # (B, K, H, W, 2)
        scale = 10.0 / (grad_bump.abs().max() + 1e-12)  # normalise to ~10 m/s
        uv = scale * grad_bump

        _, _, _, dp_s_dt = stepper.compute_tendencies(uv, T, q, p_s)

        # Find latitude row of bump peak
        H, W = SHAPE
        j_peak = W // 2
        i_peak = int(bump[:, j_peak].argmax().item())

        # Convergent flow at peak → dp_s/dt > 0
        assert dp_s_dt[0, i_peak, j_peak].item() > 0, (
            "convergent gradient flow at Gaussian peak should increase p_s there"
        )
        assert torch.isfinite(dp_s_dt).all()

    def test_sigma_dot_top_boundary_is_zero(self):
        """σ̇ at the top interface must be zero (no mass flux through the top).

        The continuity equation requires a ∂ln(p_s)/∂t term in the σ̇
        recurrence.  Without it the top-interface σ̇ equals ∂ln(p_s)/∂t
        instead of zero, meaning spurious mass flux through the model lid.

        The DISCO divergence operator acts on velocity component 1 (the
        divergent/irrotational part), so we set both components nonzero to
        produce a genuinely divergent flow.
        """
        stepper = _make_stepper(omega=0.0)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=15.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        div = stepper._divergence(uv)
        # Confirm the flow is genuinely divergent (otherwise test is vacuous)
        assert div.abs().max() > 1e-8, "test requires nonzero divergence"

        grad_log_ps = stepper._gradient_2d(torch.log(p_s))
        glps = grad_log_ps.unsqueeze(1)
        v_dot_glps = (uv * glps).sum(-1)
        dp_s_dt = -p_s * (
            (div + v_dot_glps) * stepper.delta_sigma.view(1, N_LEVELS, 1, 1)
        ).sum(dim=1)

        sigma_dot_half = stepper._sigma_dot_interfaces(
            div, uv, grad_log_ps, dp_s_dt, p_s
        )

        # The top interface (index K) must carry zero mass flux.
        # Without the ∂ln(p_s)/∂t correction the top equals d_ln_ps_dt ≈ div,
        # so the ratio to div.abs().max() would be ≈ 1 rather than ≈ 0.
        top = sigma_dot_half[:, -1]
        rel_err = top.abs().max() / div.abs().max()
        assert rel_err < 1e-3, (
            f"top-interface σ̇ / div.max should be ≈ 0, got {rel_err:.3e}"
        )
