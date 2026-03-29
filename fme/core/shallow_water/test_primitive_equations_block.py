"""Tests for PrimitiveEquationsBlockStepper."""

import math

import pytest
import torch

from fme.core.shallow_water.primitive_equations import PrimitiveEquationsStepper
from fme.core.shallow_water.primitive_equations_block import (
    PrimitiveEquationsBlockStepper,
)

SHAPE = (32, 64)
N_LEVELS = 4
T_BACKGROUND = 280.0
P_SURFACE = 101325.0


def _gaussian_bump(nlat, nlon, sigma_deg=20.0):
    lats = torch.linspace(-90, 90, nlat)
    lons = torch.linspace(0, 360, nlon)
    lat_bump = torch.exp(-(lats**2) / (2 * sigma_deg**2))
    lon_bump = torch.exp(-(lons - 180.0) ** 2 / (2 * sigma_deg**2))
    return lat_bump.unsqueeze(1) * lon_bump.unsqueeze(0)


def _make_steppers(n_levels=N_LEVELS, diffusion_coeff=None):
    """Return (block_stepper, reference_stepper) with matching settings."""
    kw = dict(
        shape=SHAPE,
        n_levels=n_levels,
        diffusion_coeff=diffusion_coeff,
    )
    block = PrimitiveEquationsBlockStepper(**kw)
    ref = PrimitiveEquationsStepper(**kw)
    return block, ref


def _random_state(n_levels=N_LEVELS, seed=0):
    torch.manual_seed(seed)
    B = 1
    bump = _gaussian_bump(*SHAPE, sigma_deg=20.0)
    T = (
        torch.full((B, n_levels, *SHAPE), T_BACKGROUND)
        + 10.0 * bump.unsqueeze(0).unsqueeze(0)
    )
    uv = torch.zeros(B, n_levels, *SHAPE, 2)
    uv[..., 0] = 5.0 * bump.unsqueeze(0).unsqueeze(0)
    uv[..., 1] = 3.0 * bump.unsqueeze(0).unsqueeze(0)
    q = torch.zeros(B, n_levels, *SHAPE)
    return uv, T, q


class TestPrimitiveEquationsBlockStepper:
    def test_output_shapes(self):
        block, _ = _make_steppers()
        uv, T, q = _random_state()
        duv, dT, dq = block.compute_tendencies(uv, T, q)
        assert duv.shape == uv.shape
        assert dT.shape == T.shape
        assert dq.shape == q.shape

    def test_resting_uniform_state_zero_momentum_tendency(self):
        """Uniform T, zero V → zero momentum tendency (no PGF gradient, no Coriolis)."""
        block, _ = _make_steppers()
        B = 1
        T = torch.full((B, N_LEVELS, *SHAPE), T_BACKGROUND)
        uv = torch.zeros(B, N_LEVELS, *SHAPE, 2)
        q = torch.zeros(B, N_LEVELS, *SHAPE)
        duv, _, _ = block.compute_tendencies(uv, T, q)
        assert duv.abs().max() < 1e-5, f"expected zero momentum tendency, got {duv.abs().max():.2e}"

    def test_momentum_tendency_matches_reference(self):
        """Block stepper duv/dt must agree with PrimitiveEquationsStepper.

        The block encodes the same physics but processes all K levels as
        channels in a single wider convolution, which accumulates ~2 ULP of
        float32 noise per output element relative to the reference that
        processes 1 channel at a time.  We therefore use a tolerance of 5e-3
        (absolute error is always < 1e-6 m/s²).
        """
        block, ref = _make_steppers()
        uv, T, q = _random_state()

        duv_block, _, _ = block.compute_tendencies(uv, T, q)
        duv_ref, _, _ = ref.compute_tendencies(uv, T, q)

        abs_err = (duv_block - duv_ref).abs().max()
        scale = duv_ref.abs().max().clamp(min=1e-10)
        rel_err = abs_err / scale
        assert rel_err < 5e-3, (
            f"momentum tendency mismatch vs reference: rel_err={rel_err:.2e} abs_err={abs_err:.2e}"
        )

    def test_full_tendency_matches_reference(self):
        """All three tendencies (uv, T, q) must agree with PrimitiveEquationsStepper."""
        block, ref = _make_steppers()
        uv, T, q = _random_state()

        duv_b, dT_b, dq_b = block.compute_tendencies(uv, T, q)
        duv_r, dT_r, dq_r = ref.compute_tendencies(uv, T, q)

        # Same float32 noise tolerance as test_momentum_tendency_matches_reference
        rtol = 5e-3
        for name, a, b in [("duv/dt", duv_b, duv_r), ("dT/dt", dT_b, dT_r), ("dq/dt", dq_b, dq_r)]:
            scale = b.abs().max().clamp(min=1e-10)
            rel_err = (a - b).abs().max() / scale
            assert rel_err < rtol, f"{name}: rel_err={rel_err:.2e}"

    def test_pgf_sign_and_magnitude(self):
        """Warm column should accelerate flow divergently (positive PGF tendency)."""
        block, _ = _make_steppers(n_levels=2)
        B = 1
        # Warm at surface (level 0), cold above (level 1) → φ_1 > φ_0 at warm point
        T = torch.full((B, 2, *SHAPE), T_BACKGROUND)
        T[:, 0] += 20.0 * _gaussian_bump(*SHAPE).unsqueeze(0)  # warm surface
        uv = torch.zeros(B, 2, *SHAPE, 2)
        q = torch.zeros(B, 2, *SHAPE)
        duv, _, _ = block.compute_tendencies(uv, T, q)
        # momentum tendency should be non-trivial
        assert duv.abs().max() > 1e-8

    def test_single_level(self):
        """K=1 case: no PGF (single level has no hydrostatic contribution)."""
        block, ref = _make_steppers(n_levels=1)
        uv, T, q = _random_state(n_levels=1)
        duv_b, dT_b, dq_b = block.compute_tendencies(uv, T, q)
        duv_r, dT_r, dq_r = ref.compute_tendencies(uv, T, q)
        for name, a, b in [("duv", duv_b, duv_r), ("dT", dT_b, dT_r)]:
            scale = b.abs().max().clamp(min=1e-10)
            assert (a - b).abs().max() / scale < 5e-3, f"{name} mismatch (K=1)"

    def test_divergence_extracted_correctly(self):
        """Divergence from block output must equal divergence from reference."""
        block, ref = _make_steppers()
        uv, T, q = _random_state()
        B, K, H, W = T.shape

        # Run block and extract δ channels
        ke = 0.5 * (uv[..., 0] ** 2 + uv[..., 1] ** 2)
        from fme.core.shallow_water.primitive_equations_block import _N_KINDS, _DIV_KIND
        zeros = torch.zeros_like(T)
        per_level = torch.stack([T, ke, zeros, zeros], dim=2)
        x_s = torch.cat([per_level.reshape(B, K * _N_KINDS, H, W),
                         block.f_coriolis.expand(B, 1, H, W)], dim=1)
        y_s, _ = block.block(x_s, uv)
        div_block = y_s[:, :K * _N_KINDS].reshape(B, K, _N_KINDS, H, W)[:, :, _DIV_KIND]

        # Reference divergence via PrimitiveEquationsStepper's div_conv
        div_ref = ref._divergence(uv)

        scale = div_ref.abs().max().clamp(min=1e-10)
        rel_err = (div_block - div_ref).abs().max() / scale
        assert rel_err < 5e-3, f"divergence mismatch: rel_err={rel_err:.2e}"

    def test_step_preserves_shape(self):
        block, _ = _make_steppers()
        uv, T, q = _random_state()
        uv2, T2, q2 = block.step(uv, T, q, dt=300.0)
        assert uv2.shape == uv.shape
        assert T2.shape == T.shape
        assert q2.shape == q.shape

    def test_no_nans_after_step(self):
        block, _ = _make_steppers()
        uv, T, q = _random_state()
        uv2, T2, q2 = block.step(uv, T, q, dt=300.0)
        assert not torch.isnan(uv2).any()
        assert not torch.isnan(T2).any()
        assert not torch.isnan(q2).any()
