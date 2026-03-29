import math

import pytest
import torch

from fme.core.disco import (
    DiscreteContinuousConvS2,
    VectorDiscoConvS2,
    VectorFilterBasis,
    get_vector_filter_basis,
)
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


def _precompute(**kw):
    """Call precomputation and return a dict for easy access."""
    result = _precompute_vector_convolution_tensor_s2(**kw)
    return dict(
        idx=result[0],
        vals=result[1],
        cos_gamma=result[2],
        sin_gamma=result[3],
        cos_phi=result[4],
        sin_phi=result[5],
        cos_beta=result[6],
        sin_beta=result[7],
    )


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
        r = _precompute(
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            filter_basis=fb,
            theta_cutoff=THETA_CUTOFF,
            basis_norm_mode="mean",
        )
        torch.testing.assert_close(r["idx"], idx_ref)
        torch.testing.assert_close(r["vals"], vals_ref)

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
        """cos^2 + sin^2 = 1 for all angular pairs at every support point.

        The bearing-based pairs (phi, beta) are zeroed at theta~0
        (undefined bearing at the origin), so the identity only holds
        where the angular values are nonzero.
        """
        fb = get_filter_basis(kernel_shape, basis_type)
        r = _precompute(
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            filter_basis=fb,
            theta_cutoff=THETA_CUTOFF,
        )
        nz = r["vals"].abs() > 1e-10
        assert nz.any(), "expected some nonzero filter values"
        rhs = r["vals"][nz] ** 2

        # gamma: defined everywhere (frame rotation is well-defined)
        c, s = r["cos_gamma"][nz], r["sin_gamma"][nz]
        torch.testing.assert_close(c**2 + s**2, rhs, atol=1e-5, rtol=1e-5)

        # phi, beta: zeroed at theta~0, check only nonzero entries
        for name in ("phi", "beta"):
            c, s = r[f"cos_{name}"][nz], r[f"sin_{name}"][nz]
            lhs = c**2 + s**2
            bearing_nz = lhs > 1e-10
            assert bearing_nz.any()
            torch.testing.assert_close(
                lhs[bearing_nz],
                rhs[bearing_nz],
                atol=1e-5,
                rtol=1e-5,
            )

    @pytest.mark.parametrize(
        "basis_type,kernel_shape",
        [("piecewise linear", 3), ("morlet", (3, 3)), ("zernike", 4)],
    )
    def test_cos_sin_orthogonality(self, basis_type, kernel_shape):
        """cos and sin patterns are orthogonal for all angular pairs.

        For each (kernel index k, output latitude j), the weighted inner
        product of cos and sin should vanish.
        """
        fb = get_filter_basis(kernel_shape, basis_type)
        nlat_out = IMG_SHAPE[0]
        K = fb.kernel_size
        r = _precompute(
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            filter_basis=fb,
            theta_cutoff=THETA_CUTOFF,
        )
        ker_idx = r["idx"][0]
        row_idx = r["idx"][1]

        for name in ("gamma", "phi", "beta"):
            vc = r[f"cos_{name}"]
            vs = r[f"sin_{name}"]
            for k in range(K):
                for j in range(nlat_out):
                    mask = (ker_idx == k) & (row_idx == j)
                    if not mask.any():
                        continue
                    inner = (vc[mask] * vs[mask]).sum()
                    norm = (r["vals"][mask] ** 2).sum()
                    if norm > 1e-12:
                        assert abs(inner / norm) < 0.05, (
                            f"{name} cos/sin not orthogonal "
                            f"at k={k}, j={j}: "
                            f"inner/norm={inner/norm:.4f}"
                        )

    def test_same_meridian_zero_rotation(self):
        """Points on the same meridian (Δlon = 0) have frame rotation γ = 0."""
        fb = _make_filter_basis()
        nlon_in = IMG_SHAPE[1]
        r = _precompute(
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            filter_basis=fb,
            theta_cutoff=THETA_CUTOFF,
        )
        input_lon = r["idx"][2] % nlon_in
        mask = (input_lon == 0) & (r["vals"].abs() > 1e-10)
        assert mask.any(), "expected some support points at Δlon = 0"

        cos_ratio = r["cos_gamma"][mask] / r["vals"][mask]
        sin_ratio = r["sin_gamma"][mask] / r["vals"][mask]
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
        r = _precompute(
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            filter_basis=fb,
            theta_cutoff=THETA_CUTOFF,
        )
        vals_scalar = r["vals"]
        vals_cos = r["cos_gamma"]
        vals_sin = r["sin_gamma"]
        ker = r["idx"][0]
        row = r["idx"][1]
        col = r["idx"][2]
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


def _unpack_fft(result):
    """Unpack build_vector_psi_fft result into a dict."""
    return dict(
        psi_s=result[0],
        s_cg=result[1],
        s_sg=result[2],
        s_g=result[3],
        v_cp=result[4],
        v_sp=result[5],
        v_cb=result[6],
        v_sb=result[7],
        v_g=result[8],
    )


class TestBuildVectorPsiFft:
    def test_scalar_fft_matches_existing(self):
        """psi_scalar_fft matches the existing scalar-only pipeline."""
        fb = _make_filter_basis()
        vfb = VectorFilterBasis(fb, fb)
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

        f = _unpack_fft(
            build_vector_psi_fft(
                IMG_SHAPE,
                IMG_SHAPE,
                vfb,
                theta_cutoff=THETA_CUTOFF,
            )
        )
        torch.testing.assert_close(f["psi_s"], psi_ref_fft)
        torch.testing.assert_close(f["s_g"], gather_ref)

    def test_fft_shapes_equal_bases(self):
        """FFT tensor shapes when K_s == K_v."""
        fb = _make_filter_basis()
        vfb = VectorFilterBasis(fb, fb)
        f = _unpack_fft(
            build_vector_psi_fft(IMG_SHAPE, IMG_SHAPE, vfb, theta_cutoff=THETA_CUTOFF)
        )
        K = fb.kernel_size
        nlat_out = IMG_SHAPE[0]
        assert f["psi_s"].shape[0] == K
        assert f["psi_s"].shape[1] == nlat_out
        # All scalar-basis tensors share shape
        assert f["psi_s"].shape == f["s_cg"].shape == f["s_sg"].shape
        # All vector-basis tensors share shape (equal basis → same K)
        assert f["v_cp"].shape == f["v_sp"].shape
        assert f["v_cb"].shape == f["v_sb"].shape
        assert f["s_g"].shape == f["v_g"].shape

    def test_fft_shapes_different_bases(self):
        """FFT tensor shapes when K_s != K_v."""
        vfb = get_vector_filter_basis(3, 5)
        K_s = vfb.scalar_kernel_size
        K_v = vfb.vector_kernel_size
        assert K_s != K_v

        f = _unpack_fft(
            build_vector_psi_fft(IMG_SHAPE, IMG_SHAPE, vfb, theta_cutoff=THETA_CUTOFF)
        )
        # Scalar-basis tensors have K_s
        assert f["psi_s"].shape[0] == K_s
        assert f["s_cg"].shape[0] == K_s
        # Vector-basis tensors have K_v
        assert f["v_cp"].shape[0] == K_v
        assert f["v_cb"].shape[0] == K_v
        # All share nlat_out
        assert f["psi_s"].shape[1] == IMG_SHAPE[0]
        assert f["v_cp"].shape[1] == IMG_SHAPE[0]

    @pytest.mark.parametrize(
        "basis_type,kernel_shape",
        [("piecewise linear", 3), ("morlet", (3, 3))],
    )
    def test_works_with_different_basis_types(self, basis_type, kernel_shape):
        """Precomputation succeeds with different filter basis types."""
        vfb = get_vector_filter_basis(kernel_shape, kernel_shape, basis_type)
        f = _unpack_fft(
            build_vector_psi_fft(IMG_SHAPE, IMG_SHAPE, vfb, theta_cutoff=THETA_CUTOFF)
        )
        assert f["psi_s"].shape == f["s_cg"].shape == f["s_sg"].shape
        assert f["v_cp"].shape == f["v_sp"].shape
        assert f["v_cb"].shape == f["v_sb"].shape


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

    def test_forward_shape_different_ks_kv(self):
        """Output shapes correct when K_s != K_v."""
        vfb = get_vector_filter_basis(3, 5)
        conv = VectorDiscoConvS2(
            in_channels_scalar=3,
            in_channels_vector=2,
            out_channels_scalar=4,
            out_channels_vector=2,
            in_shape=IMG_SHAPE,
            out_shape=IMG_SHAPE,
            vector_filter_basis=vfb,
            theta_cutoff=THETA_CUTOFF,
        )
        B = 2
        x_s = torch.randn(B, 3, *IMG_SHAPE)
        x_v = torch.randn(B, 2, *IMG_SHAPE, 2)
        with torch.no_grad():
            y_s, y_v = conv(x_s, x_v)
        assert y_s.shape == (B, 4, *IMG_SHAPE)
        assert y_v.shape == (B, 2, *IMG_SHAPE, 2)
        # Verify weight shapes reflect K_s vs K_v
        K_s = vfb.scalar_kernel_size
        K_v = vfb.vector_kernel_size
        assert conv.W_ss.shape == (4, 3, K_s)
        assert conv.W_vv.shape == (2, 2, K_s, 2)
        assert conv.W_sv.shape == (2, 3, K_v, 2)
        assert conv.W_vs.shape == (4, 2, K_v, 2)

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
        """R_x(pi) equivariance for all four paths.

        R_x(pi) maps (theta, phi) -> (pi-theta, 2pi-phi) and negates
        both vector components. The equiangular grid maps exactly to
        itself, so no interpolation is needed.

        All four paths (ss, sv, vs, vv) are equivariant because the
        bearing-based angular factors (φ for sv, β for vs) transform
        correctly under the rotation.
        """
        torch.manual_seed(123)
        nlat, nlon = IMG_SHAPE
        lat_idx = torch.arange(nlat - 1, -1, -1)
        lon_idx = (-torch.arange(nlon)) % nlon

        def rotate_s(t):
            return t[:, :, lat_idx][:, :, :, lon_idx].contiguous()

        def rotate_v(t):
            return (-t[:, :, lat_idx][:, :, :, lon_idx]).contiguous()

        # Full module with all four paths active
        conv = _make_conv(3, 2, 3, 2)
        x_s = torch.randn(1, 3, nlat, nlon)
        x_v = torch.randn(1, 2, nlat, nlon, 2)
        with torch.no_grad():
            y_s, y_v = conv(x_s, x_v)
            y_s_rot, y_v_rot = conv(rotate_s(x_s), rotate_v(x_v))
        torch.testing.assert_close(y_s_rot, rotate_s(y_s), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y_v_rot, rotate_v(y_v), atol=1e-4, rtol=1e-4)

    def test_scalar_gradient_produces_vector(self):
        """Scalar-to-vector path produces nonzero directional output."""
        nlat, nlon = IMG_SHAPE
        conv = _make_conv(1, 0, 0, 1, bias=False)

        lon = torch.linspace(0, 2 * math.pi, nlon + 1)[:nlon]
        s = torch.cos(lon).reshape(1, 1, 1, nlon).repeat(1, 1, nlat, 1)

        # Use both gradient components
        with torch.no_grad():
            conv.W_sv.zero_()
            conv.W_sv[0, 0, :, :] = 1.0

        x_v = torch.zeros(1, 0, nlat, nlon, 2)
        with torch.no_grad():
            _, y_v = conv(s, x_v)

        # Output should vary with longitude (not be flat)
        mid = nlat // 2
        mag = (y_v[0, 0, mid, :, 0] ** 2 + y_v[0, 0, mid, :, 1] ** 2).sqrt()
        assert mag.std() > 0.001, "gradient output is flat"

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
        assert y_s[0, 0, mid].std() > 0.001, "divergence output is flat"

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
        assert y_s[0, 0, mid].std() > 0.001, "curl output is flat"
