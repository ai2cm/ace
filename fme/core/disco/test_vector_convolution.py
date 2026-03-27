import math

import pytest
import torch

from fme.core.disco import DiscreteContinuousConvS2, VectorDiscoConvS2
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

    @pytest.mark.parametrize(
        "basis_type,kernel_shape",
        [
            ("piecewise linear", 3),
            ("piecewise linear", 5),
            ("morlet", (3, 3)),
            ("zernike", 4),
        ],
    )
    def test_cos_sin_pythagorean(self, basis_type, kernel_shape):
        """cos^2(γ) + sin^2(γ) = 1 at every support point, all basis types."""
        fb = get_filter_basis(kernel_shape, basis_type)
        _, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        nonzero = vals_scalar.abs() > 1e-10
        assert nonzero.any(), "expected some nonzero filter values"
        lhs = vals_cos[nonzero] ** 2 + vals_sin[nonzero] ** 2
        rhs = vals_scalar[nonzero] ** 2
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        "basis_type,kernel_shape",
        [("piecewise linear", 3), ("morlet", (3, 3)), ("zernike", 4)],
    )
    def test_cos_sin_orthogonality(self, basis_type, kernel_shape):
        """cos(γ) and sin(γ) patterns are orthogonal over each filter support.

        For each (kernel index k, output latitude j), the weighted inner
        product of the two vector filter components should vanish, verifying
        they are perpendicular as functions over the filter support.
        """
        fb = get_filter_basis(kernel_shape, basis_type)
        nlat_out = IMG_SHAPE[0]
        K = fb.kernel_size
        idx, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
            IMG_SHAPE, IMG_SHAPE, fb, theta_cutoff=THETA_CUTOFF
        )
        ker_idx = idx[0]
        row_idx = idx[1]

        for k in range(K):
            for j in range(nlat_out):
                mask = (ker_idx == k) & (row_idx == j)
                if not mask.any():
                    continue
                # Weighted inner product of cos(γ) and sin(γ) patterns
                # Using |vals_scalar| as weight (quadrature already merged)
                inner = (vals_cos[mask] * vals_sin[mask]).sum()
                norm = (vals_scalar[mask] ** 2).sum()
                if norm > 1e-12:
                    # Normalized inner product should be near zero
                    assert abs(inner / norm) < 0.05, (
                        f"cos/sin not orthogonal at k={k}, j={j}: "
                        f"inner/norm={inner/norm:.4f}"
                    )

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


CONV_KWARGS: dict = dict(
    in_shape=IMG_SHAPE,
    out_shape=IMG_SHAPE,
    kernel_shape=3,
    basis_type="piecewise linear",
    basis_norm_mode="mean",
    grid_in="equiangular",
    grid_out="equiangular",
    theta_cutoff=THETA_CUTOFF,
)


def _make_conv(n_s_in=4, n_v_in=2, n_s_out=3, n_v_out=2, **kw):
    merged = {**CONV_KWARGS, **kw}
    return VectorDiscoConvS2(
        in_channels_scalar=n_s_in,
        in_channels_vector=n_v_in,
        out_channels_scalar=n_s_out,
        out_channels_vector=n_v_out,
        **merged,
    )


class TestVectorDiscoConvS2:
    def test_forward_shape(self):
        """Output tensors have the expected shapes."""
        n_s_in, n_v_in, n_s_out, n_v_out = 4, 2, 3, 5
        conv = _make_conv(n_s_in, n_v_in, n_s_out, n_v_out)
        B = 2
        x_s = torch.randn(B, n_s_in, *IMG_SHAPE)
        x_v = torch.randn(B, n_v_in, *IMG_SHAPE, 2)
        with torch.no_grad():
            y_s, y_v = conv(x_s, x_v)
        assert y_s.shape == (B, n_s_out, *IMG_SHAPE)
        assert y_v.shape == (B, n_v_out, *IMG_SHAPE, 2)

    @pytest.mark.parametrize(
        "n_s_in,n_v_in,n_s_out,n_v_out",
        [(4, 0, 3, 0), (0, 2, 0, 3), (4, 2, 0, 3), (4, 2, 3, 0)],
    )
    def test_forward_shape_edge_cases(self, n_s_in, n_v_in, n_s_out, n_v_out):
        """Works with zero-channel edge cases."""
        conv = _make_conv(n_s_in, n_v_in, n_s_out, n_v_out)
        B = 2
        x_s = torch.randn(B, n_s_in, *IMG_SHAPE)
        x_v = torch.randn(B, n_v_in, *IMG_SHAPE, 2)
        with torch.no_grad():
            y_s, y_v = conv(x_s, x_v)
        assert y_s.shape == (B, n_s_out, *IMG_SHAPE)
        assert y_v.shape == (B, n_v_out, *IMG_SHAPE, 2)

    def test_scalar_only_matches_existing(self):
        """With N_v=0, matches existing DiscreteContinuousConvS2."""
        torch.manual_seed(42)
        n_s = 4
        conv_vec = _make_conv(n_s, 0, n_s, 0, bias=True)
        conv_ref = DiscreteContinuousConvS2(
            in_channels=n_s,
            out_channels=n_s,
            bias=True,
            **CONV_KWARGS,
        )
        # Copy weights
        with torch.no_grad():
            conv_ref.weight.copy_(conv_vec.W_ss)
            conv_ref.bias.copy_(conv_vec.bias_scalar)

        x_s = torch.randn(2, n_s, *IMG_SHAPE)
        x_v = torch.randn(2, 0, *IMG_SHAPE, 2)
        with torch.no_grad():
            y_s_vec, _ = conv_vec(x_s, x_v)
            y_ref = conv_ref(x_s)

        torch.testing.assert_close(y_s_vec, y_ref, atol=1e-5, rtol=1e-5)

    def test_longitude_shift_equivariance(self):
        """Shifting input by N longitudes shifts output by N."""
        torch.manual_seed(0)
        conv = _make_conv(3, 2, 3, 2)
        shift = 7
        x_s = torch.randn(1, 3, *IMG_SHAPE)
        x_v = torch.randn(1, 2, *IMG_SHAPE, 2)

        with torch.no_grad():
            y_s, y_v = conv(x_s, x_v)
            x_s_sh = torch.roll(x_s, shift, dims=-1)
            x_v_sh = torch.roll(x_v, shift, dims=-2)
            y_s_sh, y_v_sh = conv(x_s_sh, x_v_sh)

        torch.testing.assert_close(
            y_s_sh,
            torch.roll(y_s, shift, dims=-1),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            y_v_sh,
            torch.roll(y_v, shift, dims=-2),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradient_flow_all_weights(self):
        """Gradients flow through all four weight blocks."""
        conv = _make_conv(2, 2, 2, 2)
        x_s = torch.randn(1, 2, *IMG_SHAPE)
        x_v = torch.randn(1, 2, *IMG_SHAPE, 2)
        y_s, y_v = conv(x_s, x_v)
        loss = y_s.sum() + y_v.sum()
        loss.backward()

        for name in ["W_ss", "W_vs", "W_sv", "W_vv"]:
            param = getattr(conv, name)
            assert param.grad is not None, f"no gradient for {name}"
            assert param.grad.abs().max() > 0, f"zero grad for {name}"

    def test_bias_only_on_scalars(self):
        """Bias affects scalar outputs only; vectors have no bias."""
        conv = _make_conv(2, 2, 2, 2, bias=True)
        assert conv.bias_scalar is not None
        assert conv.bias_scalar.shape == (2,)

        x_s = torch.zeros(1, 2, *IMG_SHAPE)
        x_v = torch.zeros(1, 2, *IMG_SHAPE, 2)
        with torch.no_grad():
            conv.W_ss.zero_()
            conv.W_vs.zero_()
            conv.W_sv.zero_()
            conv.W_vv.zero_()
            conv.bias_scalar.fill_(3.0)
            y_s, y_v = conv(x_s, x_v)

        torch.testing.assert_close(y_s, torch.full_like(y_s, 3.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(y_v, torch.zeros_like(y_v), atol=1e-6, rtol=1e-6)

    def test_rotation_equivariance_180_x(self):
        """R_x(pi) equivariance for scalar-only and vector-only paths.

        R_x(pi) maps (theta, phi) -> (pi-theta, 2pi-phi) and negates
        both vector components. The equiangular grid maps exactly to
        itself, so no interpolation is needed.

        The scalar->scalar and vector->vector paths are equivariant
        under this rotation. The scalar<->vector cross-type paths are
        equivariant under azimuthal rotations but not R_x(pi), because
        cos(gamma) is invariant under R_x(pi) while the output frame
        negates — this is inherent to the meridian-frame design.
        """
        torch.manual_seed(123)
        nlat, nlon = IMG_SHAPE
        lat_idx = torch.arange(nlat - 1, -1, -1)
        lon_idx = (-torch.arange(nlon)) % nlon

        def rotate_s(t):
            return t[:, :, lat_idx][:, :, :, lon_idx].contiguous()

        def rotate_v(t):
            return (-t[:, :, lat_idx][:, :, :, lon_idx]).contiguous()

        # scalar→scalar
        conv_ss = _make_conv(3, 0, 3, 0)
        x_s = torch.randn(1, 3, nlat, nlon)
        empty_v = torch.zeros(1, 0, nlat, nlon, 2)
        with torch.no_grad():
            y_s, _ = conv_ss(x_s, empty_v)
            y_s_rot, _ = conv_ss(rotate_s(x_s), empty_v)
        torch.testing.assert_close(y_s_rot, rotate_s(y_s), atol=1e-4, rtol=1e-4)

        # vector→vector (no cross-type paths)
        conv_vv = _make_conv(0, 2, 0, 2)
        empty_s = torch.zeros(1, 0, nlat, nlon)
        x_v = torch.randn(1, 2, nlat, nlon, 2)
        with torch.no_grad():
            _, y_v = conv_vv(empty_s, x_v)
            _, y_v_rot = conv_vv(empty_s, rotate_v(x_v))
        torch.testing.assert_close(y_v_rot, rotate_v(y_v), atol=1e-4, rtol=1e-4)

    def test_scalar_gradient_direction(self):
        """Scalar-to-vector gradient of cos(lon) points eastward."""
        nlat, nlon = IMG_SHAPE
        conv = _make_conv(1, 0, 0, 1, bias=False)

        lon = torch.linspace(0, 2 * math.pi, nlon + 1)[:nlon]
        s = torch.cos(lon).reshape(1, 1, 1, nlon).repeat(1, 1, nlat, 1)

        with torch.no_grad():
            conv.W_sv.zero_()
            conv.W_sv[0, 0, :, 0] = 1.0  # gradient component only

        x_v = torch.zeros(1, 0, nlat, nlon, 2)
        with torch.no_grad():
            _, y_v = conv(s, x_v)

        u = y_v[0, 0, :, :, 0]
        v = y_v[0, 0, :, :, 1]

        # cos(lon) varies only in longitude, so gradient is eastward
        # At mid-latitudes, |u| should dominate |v|
        mid = nlat // 2
        assert u[mid].abs().mean() > 2 * v[mid].abs().mean()

    def test_divergence_from_vector(self):
        """Vector-to-scalar divergence detects a divergent field."""
        nlat, nlon = IMG_SHAPE
        conv = _make_conv(0, 1, 1, 0, bias=False)

        # Create a field with u = cos(lon), v = 0 (divergent in u)
        lon = torch.linspace(0, 2 * math.pi, nlon + 1)[:nlon]
        u_field = torch.cos(lon).reshape(1, 1, 1, nlon).repeat(1, 1, nlat, 1)
        v_field = torch.zeros_like(u_field)
        x_v = torch.stack([u_field, v_field], dim=-1)

        with torch.no_grad():
            conv.W_vs.zero_()
            conv.W_vs[0, 0, :, 0] = 1.0  # divergence component only

        x_s = torch.zeros(1, 0, nlat, nlon)
        with torch.no_grad():
            y_s, _ = conv(x_s, x_v)

        # du/dx of cos(lon) ~ -sin(lon): should change sign at lon=pi
        mid = nlat // 2
        assert y_s[0, 0, mid].std() > 0.01, "divergence output is flat"

    def test_curl_from_vector(self):
        """Vector-to-scalar curl detects a rotational field."""
        nlat, nlon = IMG_SHAPE
        conv = _make_conv(0, 1, 1, 0, bias=False)

        # Create a field with u = 0, v = cos(lon) (has curl)
        lon = torch.linspace(0, 2 * math.pi, nlon + 1)[:nlon]
        u_field = torch.zeros(1, 1, nlat, nlon)
        v_field = torch.cos(lon).reshape(1, 1, 1, nlon).repeat(1, 1, nlat, 1)
        x_v = torch.stack([u_field, v_field], dim=-1)

        with torch.no_grad():
            conv.W_vs.zero_()
            conv.W_vs[0, 0, :, 1] = 1.0  # curl component only

        x_s = torch.zeros(1, 0, nlat, nlon)
        with torch.no_grad():
            y_s, _ = conv(x_s, x_v)

        # dv/dx of cos(lon) ~ -sin(lon): curl output should vary
        mid = nlat // 2
        assert y_s[0, 0, mid].std() > 0.01, "curl output is flat"
