import math

import pytest
import torch

from fme.core.disco._convolution import _precompute_convolution_tensor_s2
from fme.core.disco._disco_utils import _get_psi, _precompute_psi_banded
from fme.core.disco._filter_basis import get_filter_basis
from fme.core.disco._vector_convolution import (
    _precompute_vector_convolution_tensor_s2,
    build_vector_psi_fft,
)

IMG_SHAPE = (16, 32)
THETA_CUTOFF = math.pi / 15


def _make_filter_basis(kernel_shape=3, basis_type="piecewise linear"):
    return get_filter_basis(kernel_shape, basis_type)


class TestPrecomputeVectorConvolution:
    def test_scalar_vals_match_existing(self):
        """Scalar values and indices match the existing precomputation."""
        fb = _make_filter_basis()
        idx_ref, vals_ref, _ = _precompute_convolution_tensor_s2(
            IMG_SHAPE,
            IMG_SHAPE,
            fb,
            theta_cutoff=THETA_CUTOFF,
            merge_quadrature=True,
            basis_norm_mode="mean",
        )
        idx_new, vals_scalar, _, _ = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE,
            IMG_SHAPE,
            fb,
            theta_cutoff=THETA_CUTOFF,
            basis_norm_mode="mean",
        )
        torch.testing.assert_close(idx_new, idx_ref)
        torch.testing.assert_close(vals_scalar, vals_ref)

    def test_cos_sin_pythagorean(self):
        """cos^2(γ) + sin^2(γ) = 1 at every support point."""
        fb = _make_filter_basis()
        _, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        # vals_cos = vals_scalar * cos(γ), so
        # vals_cos^2 + vals_sin^2 = vals_scalar^2
        nonzero = vals_scalar.abs() > 1e-10
        assert nonzero.any(), "expected some nonzero filter values"
        lhs = vals_cos[nonzero] ** 2 + vals_sin[nonzero] ** 2
        rhs = vals_scalar[nonzero] ** 2
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_same_meridian_zero_rotation(self):
        """Points on the same meridian (Δlon = 0) have frame rotation γ = 0."""
        fb = _make_filter_basis()
        nlon_in = IMG_SHAPE[1]
        idx, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        input_lon = idx[2] % nlon_in
        mask = (input_lon == 0) & (vals_scalar.abs() > 1e-10)
        assert mask.any(), "expected some support points at Δlon = 0"

        cos_ratio = vals_cos[mask] / vals_scalar[mask]
        sin_ratio = vals_sin[mask] / vals_scalar[mask]
        torch.testing.assert_close(
            cos_ratio, torch.ones_like(cos_ratio), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            sin_ratio, torch.zeros_like(sin_ratio), atol=1e-5, rtol=1e-5
        )

    def test_sin_antisymmetric_under_longitude_flip(self):
        """sin(γ) flips sign and cos(γ) is unchanged under Δlon → −Δlon."""
        fb = _make_filter_basis()
        nlat_in, nlon_in = IMG_SHAPE
        idx, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        ker = idx[0]
        row = idx[1]
        col = idx[2]
        input_lat = col // nlon_in
        input_lon = col % nlon_in
        mirror_lon = (-input_lon) % nlon_in

        # Build lookup from (ker, row, input_lat, input_lon) → flat index
        nlat_out = IMG_SHAPE[0]
        flat_key = (
            ker * (nlat_out * nlat_in * nlon_in)
            + row * (nlat_in * nlon_in)
            + input_lat * nlon_in
            + input_lon
        )
        mirror_key = (
            ker * (nlat_out * nlat_in * nlon_in)
            + row * (nlat_in * nlon_in)
            + input_lat * nlon_in
            + mirror_lon
        )

        key_to_idx = {}
        for i, k in enumerate(flat_key.tolist()):
            key_to_idx[k] = i

        orig_indices = []
        mirr_indices = []
        for i, (mk, lon) in enumerate(zip(mirror_key.tolist(), input_lon.tolist())):
            if mk in key_to_idx and lon != 0 and lon != nlon_in // 2:
                orig_indices.append(i)
                mirr_indices.append(key_to_idx[mk])

        assert len(orig_indices) > 0, "expected matched longitude-mirror pairs"
        orig = torch.tensor(orig_indices)
        mirr = torch.tensor(mirr_indices)

        torch.testing.assert_close(
            vals_scalar[orig], vals_scalar[mirr], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(vals_cos[orig], vals_cos[mirr], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            vals_sin[orig], -vals_sin[mirr], atol=1e-5, rtol=1e-5
        )


class TestBuildVectorPsiFft:
    def test_scalar_fft_matches_existing(self):
        """psi_scalar_fft matches the existing scalar-only pipeline."""
        fb = _make_filter_basis()
        nlat_in, nlon_in = IMG_SHAPE
        nlat_out, nlon_out = IMG_SHAPE
        kernel_size = fb.kernel_size

        idx_ref, vals_ref, _ = _precompute_convolution_tensor_s2(
            IMG_SHAPE,
            IMG_SHAPE,
            fb,
            theta_cutoff=THETA_CUTOFF,
            merge_quadrature=True,
            basis_norm_mode="mean",
        )
        psi_ref = _get_psi(
            kernel_size, idx_ref, vals_ref, nlat_in, nlon_in, nlat_out, nlon_out
        )
        psi_ref_fft, gather_ref = _precompute_psi_banded(psi_ref, nlat_in, nlon_in)

        psi_scalar_fft, _, _, gather_new = build_vector_psi_fft(
            IMG_SHAPE,
            IMG_SHAPE,
            fb,
            theta_cutoff=THETA_CUTOFF,
        )

        torch.testing.assert_close(psi_scalar_fft, psi_ref_fft)
        torch.testing.assert_close(gather_new, gather_ref)

    def test_fft_shapes(self):
        """All three FFT tensors have the expected shape."""
        fb = _make_filter_basis()
        psi_s, psi_c, psi_sin, gather = build_vector_psi_fft(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        K = fb.kernel_size
        nlat_out = IMG_SHAPE[0]
        assert psi_s.shape[0] == K
        assert psi_s.shape[1] == nlat_out
        assert psi_s.shape == psi_c.shape == psi_sin.shape
        assert gather.shape[0] == nlat_out
        assert gather.shape[1] == psi_s.shape[2]  # max_bw

    @pytest.mark.parametrize(
        "basis_type,kernel_shape",
        [("piecewise linear", 3), ("morlet", (3, 3))],
    )
    def test_works_with_different_basis_types(self, basis_type, kernel_shape):
        """Precomputation succeeds with different filter basis types."""
        fb = get_filter_basis(kernel_shape, basis_type)
        psi_s, psi_c, psi_sin, gather = build_vector_psi_fft(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        assert psi_s.shape == psi_c.shape == psi_sin.shape
