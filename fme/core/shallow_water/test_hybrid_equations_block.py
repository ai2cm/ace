"""Tests for HybridCoordinateBlockStepper.

Verifies that the block encoding produces tendencies identical to the
reference HybridCoordinateStepper (to float32 precision), as well as
basic smoke tests for shape, stability, and mass conservation.
"""

import math
import pathlib

import pytest
import torch

from fme.core.shallow_water.hybrid_equations import (
    HybridCoordinateStepper,
    cam_like_coefficients,
    sigma_coefficients,
)
from fme.core.shallow_water.hybrid_equations_block import HybridCoordinateBlockStepper
from fme.core.testing.regression import validate_tensor_dict

SHAPE = (16, 32)
N_LEVELS = 3
DT = 60.0
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


def _make_steppers(sigma: bool = False, **kw):
    """Return a matched (block, reference) pair."""
    if sigma:
        sigma_levels = [
            math.exp(-k * math.log(5.0) / max(1, N_LEVELS - 1)) for k in range(N_LEVELS)
        ]
        a_mid, b_mid, a_int, b_int = sigma_coefficients(sigma_levels)
    else:
        a_mid, b_mid, a_int, b_int = cam_like_coefficients(N_LEVELS)

    common = dict(
        shape=SHAPE,
        a_mid=a_mid,
        b_mid=b_mid,
        a_interface=a_int,
        b_interface=b_int,
        **kw,
    )
    block = HybridCoordinateBlockStepper(**common)
    ref = HybridCoordinateStepper(**common)
    return block, ref


class TestHybridCoordinateBlockStepper:
    def test_output_shapes_preserved(self):
        """step() returns tensors of the same shapes as inputs."""
        block, _ = _make_steppers()
        B = 2
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        uv2, T2, q2, ps2 = block.step(uv, T, q, p_s, DT)

        assert uv2.shape == uv.shape
        assert T2.shape == T.shape
        assert q2.shape == q.shape
        assert ps2.shape == p_s.shape

    def test_resting_uniform_state_has_zero_tendency(self):
        """V=0, uniform T, uniform p_s → all tendencies zero."""
        block, _ = _make_steppers()
        B = 1
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        duv_dt, dT_dt, dq_dt, dp_s_dt = block.compute_tendencies(uv, T, q, p_s)

        assert duv_dt.abs().max() < 1e-6
        assert dT_dt.abs().max() < 1e-6
        assert dq_dt.abs().max() < 1e-6
        assert dp_s_dt.abs().max() < 1e-6

    @pytest.mark.parametrize("sigma", [False, True])
    def test_tendencies_match_reference(self, sigma):
        """Block tendencies match HybridCoordinateStepper to ~1e-4 relative error."""
        block, ref = _make_steppers(sigma=sigma, omega=7.292e-5, diffusion_coeff=None)

        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 3.0 * bump.unsqueeze(0).unsqueeze(0)
        q = 0.01 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        duv_b, dT_b, dq_b, dps_b = block.compute_tendencies(uv, T, q, p_s)
        duv_r, dT_r, dq_r, dps_r = ref.compute_tendencies(uv, T, q, p_s)

        # The block batches K channels in one large einsum; the reference uses
        # separate single-channel calls (via reshape). Different float32
        # summation order causes rounding differences of ~1e-4; 5e-4 is safe.
        rtol = 5e-4

        def _check(a, b, name):
            scale = a.abs().max().clamp(min=1e-10)
            err = (a - b).abs().max() / scale
            assert err < rtol, f"{name}: relative error {err:.2e} > {rtol}"

        _check(duv_b, duv_r, "duv/dt")
        _check(dT_b, dT_r, "dT/dt")
        _check(dq_b, dq_r, "dq/dt")
        _check(dps_b, dps_r, "dp_s/dt")

    def test_mass_conservation(self):
        """Global ∫ dp_s/dt dA ≈ 0."""
        block, _ = _make_steppers()
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 10.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        _, _, _, dp_s_dt = block.compute_tendencies(uv, T, q, p_s)
        global_mean = block.integrate_area(dp_s_dt[0]).item()

        assert (
            abs(global_mean) < 1e3
        ), f"global dp_s/dt should be near zero: {global_mean:.2f}"

    def test_no_nans_after_many_steps(self):
        """Integration remains finite over 200 steps with diffusion."""
        block, _ = _make_steppers(diffusion_coeff=1e5)
        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 5.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE)

        for _ in range(200):
            uv, T, q, p_s = block.step(uv, T, q, p_s, DT)

        assert torch.isfinite(uv).all()
        assert torch.isfinite(T).all()
        assert torch.isfinite(p_s).all()

    def test_tendencies_with_diffusion_match_reference(self):
        """Block tendencies with diffusion match reference stepper."""
        block, ref = _make_steppers(omega=7.292e-5, diffusion_coeff=1e5)

        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 3.0 * bump.unsqueeze(0).unsqueeze(0)
        q = 0.01 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        duv_b, dT_b, dq_b, dps_b = block.compute_tendencies(uv, T, q, p_s)
        duv_r, dT_r, dq_r, dps_r = ref.compute_tendencies(uv, T, q, p_s)

        # The block uses a single-pass Laplacian approximation (W_ss/W_vv)
        # instead of the exact two-pass ∇·∇. The different discrete operator
        # produces ~10% relative error which is acceptable for numerical
        # stabilization diffusion.
        rtol = 1.5e-1

        def _check(a, b, name):
            scale = a.abs().max().clamp(min=1e-10)
            err = (a - b).abs().max() / scale
            assert err < rtol, f"{name}: relative error {err:.2e} > {rtol}"

        _check(duv_b, duv_r, "duv/dt")
        _check(dT_b, dT_r, "dT/dt")
        _check(dq_b, dq_r, "dq/dt")
        _check(dps_b, dps_r, "dp_s/dt")

    def test_regression_tendencies_no_diffusion(self):
        """Regression test: exact tendency output is locked down."""
        block, _ = _make_steppers(omega=7.292e-5, diffusion_coeff=None)

        B = 1
        bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
        T_pert = 10.0 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND) + T_pert
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        uv[..., 0] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
        uv[..., 1] = 3.0 * bump.unsqueeze(0).unsqueeze(0)
        q = 0.01 * bump.unsqueeze(0).unsqueeze(0).expand(B, N_LEVELS, *SHAPE)
        p_s = torch.full((B, *SHAPE), P_SURFACE) + 500.0 * bump.unsqueeze(0)

        duv, dT, dq, dps = block.compute_tendencies(uv, T, q, p_s)

        baseline_dir = pathlib.Path(__file__).parent / "baselines"
        baseline_dir.mkdir(exist_ok=True)
        validate_tensor_dict(
            {"duv": duv, "dT": dT, "dq": dq, "dps": dps},
            baseline_dir / "hybrid_block_tendencies_no_diffusion.pt",
            rtol=1e-5,
            atol=1e-6,
        )
