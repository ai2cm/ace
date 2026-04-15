import math

import torch
import torch.nn as nn

from ._cache import lru_cache
from ._convolution import _normalize_convolution_tensor_s2
from ._disco_utils import _disco_s2_contraction_fft, _get_psi, _precompute_psi_banded
from ._fft import rfft
from ._filter_basis import FilterBasis, get_filter_basis
from ._quadrature import precompute_latitudes, precompute_longitudes


class VectorFilterBasis:
    """Pairs a scalar and vector radial filter basis.

    The scalar basis (isotropic filters) is used for scalar→scalar and
    vector→vector paths. The vector basis (oriented cos/sin filters) is
    used for scalar→vector and vector→scalar paths. The two bases can
    have different kernel shapes, allowing independent radial resolution
    for isotropic and oriented operations.
    """

    def __init__(
        self,
        scalar_basis: FilterBasis,
        vector_basis: FilterBasis,
    ):
        self.scalar_basis = scalar_basis
        self.vector_basis = vector_basis

    @property
    def scalar_kernel_size(self) -> int:
        return self.scalar_basis.kernel_size

    @property
    def vector_kernel_size(self) -> int:
        return self.vector_basis.kernel_size

    def __repr__(self):
        return (
            f"VectorFilterBasis("
            f"scalar={self.scalar_basis}, "
            f"vector={self.vector_basis})"
        )


def get_vector_filter_basis(
    scalar_kernel_shape: int | tuple[int, ...],
    vector_kernel_shape: int | tuple[int, ...],
    basis_type: str = "piecewise linear",
) -> VectorFilterBasis:
    """Create a VectorFilterBasis from kernel shape parameters."""
    return VectorFilterBasis(
        scalar_basis=get_filter_basis(scalar_kernel_shape, basis_type),
        vector_basis=get_filter_basis(vector_kernel_shape, basis_type),
    )


@lru_cache(typed=True, copy=True)
def _precompute_vector_convolution_tensor_s2(
    in_shape: tuple[int],
    out_shape: tuple[int],
    filter_basis: FilterBasis,
    grid_in: str = "equiangular",
    grid_out: str = "equiangular",
    theta_cutoff: float = 0.01 * math.pi,
    theta_eps: float = 1e-3,
    basis_norm_mode: str = "mean",
):
    r"""Precompute filter tensors for vector DISCO convolution.

    Computes filter basis values and three angular pairs (φ, β, γ) at
    each support point. Returns normalized values modulated by each pair.

    Returns:
    -------
    out_idx : (3, nnz) long tensor
        Indices [kernel, out_lat, in_lat*nlon_in + in_lon].
    out_vals_scalar : (nnz,) float tensor
        Normalized ψ_k(r) values with quadrature weights merged.
    out_vals_cos_gamma, out_vals_sin_gamma : (nnz,) float tensors
        ψ_k(r)·cos(γ), ψ_k(r)·sin(γ) — for frame rotation (vv path).
    out_vals_cos_phi, out_vals_sin_phi : (nnz,) float tensors
        ψ_k(r)·cos(φ), ψ_k(r)·sin(φ) — bearing at output (sv path).
    out_vals_cos_beta, out_vals_sin_beta : (nnz,) float tensors
        ψ_k(r)·cos(β), ψ_k(r)·sin(β) — bearing at input (vs path).
    """
    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size
    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, _ = precompute_latitudes(nlat_out, grid=grid_out)
    lons_in = precompute_longitudes(nlon_in)

    quad_weights = win.reshape(-1, 1) / nlon_in / 2.0
    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    collected_idx = []
    collected_vals = []
    collected_cos_gamma = []
    collected_sin_gamma = []
    collected_cos_phi = []
    collected_sin_phi = []
    collected_cos_beta = []
    collected_sin_beta = []

    # Input coordinates: beta = longitude (λ), gamma = colatitude
    beta = lons_in
    gamma = lats_in.reshape(-1, 1)

    cbeta = torch.cos(beta)
    sbeta = torch.sin(beta)
    cgamma = torch.cos(gamma)
    sgamma = torch.sin(gamma)

    for t in range(nlat_out):
        alpha = -lats_out[t]
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)

        # Euler-rotated position (Y-axis rotation by alpha)
        x = cos_alpha * cbeta * sgamma + cgamma * sin_alpha
        y = sbeta * sgamma
        z = -cbeta * sin_alpha * sgamma + cos_alpha * cgamma

        norm_xyz = torch.sqrt(x * x + y * y + z * z)
        x = x / norm_xyz
        y = y / norm_xyz
        z = z / norm_xyz

        theta = torch.arccos(z)
        phi = torch.arctan2(y, x)
        phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

        # Euler-rotated geographic north of each input point
        eN_x = -cos_alpha * cgamma * cbeta + sin_alpha * sgamma
        eN_y = -cgamma * sbeta
        eN_z = sin_alpha * cgamma * cbeta + cos_alpha * sgamma

        # Local basis vectors at the rotated position (theta, phi)
        cos_theta = z
        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta * cos_theta, min=0.0))
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # cos(β_angle) = ê_N' · (−θ̂),  sin(β_angle) = ê_N' · φ̂
        cos_beta_angle = (
            eN_x * (-cos_theta * cos_phi)
            + eN_y * (-cos_theta * sin_phi)
            + eN_z * sin_theta
        )
        sin_beta_angle = eN_x * (-sin_phi) + eN_y * cos_phi

        # γ_frame = φ − β_angle
        cos_gamma_frame = cos_phi * cos_beta_angle + sin_phi * sin_beta_angle
        sin_gamma_frame = sin_phi * cos_beta_angle - cos_phi * sin_beta_angle

        # Filter support values
        iidx, vals = filter_basis.compute_support_vals(
            theta, phi, r_cutoff=theta_cutoff_eff
        )

        idx = torch.stack(
            [
                iidx[:, 0],
                t * torch.ones_like(iidx[:, 0]),
                iidx[:, 1] * nlon_in + iidx[:, 2],
            ],
            dim=0,
        )

        si, sj = iidx[:, 1], iidx[:, 2]

        # At θ=0 (input coincides with output), the bearing direction
        # is undefined. Zero out bearing-based angles (φ, β) there;
        # the frame rotation γ remains valid (identity at θ=0).
        near_pole = theta[si, sj] < 1e-10
        s_cp = cos_phi[si, sj].clone()
        s_sp = sin_phi[si, sj].clone()
        s_cb = cos_beta_angle[si, sj].clone()
        s_sb = sin_beta_angle[si, sj].clone()
        s_cp[near_pole] = 0.0
        s_sp[near_pole] = 0.0
        s_cb[near_pole] = 0.0
        s_sb[near_pole] = 0.0

        collected_idx.append(idx)
        collected_vals.append(vals)
        collected_cos_gamma.append(cos_gamma_frame[si, sj])
        collected_sin_gamma.append(sin_gamma_frame[si, sj])
        collected_cos_phi.append(s_cp)
        collected_sin_phi.append(s_sp)
        collected_cos_beta.append(s_cb)
        collected_sin_beta.append(s_sb)

    out_idx = torch.cat(collected_idx, dim=-1)
    out_vals = torch.cat(collected_vals, dim=-1)
    cos_g = torch.cat(collected_cos_gamma, dim=-1)
    sin_g = torch.cat(collected_sin_gamma, dim=-1)
    cos_p = torch.cat(collected_cos_phi, dim=-1)
    sin_p = torch.cat(collected_sin_phi, dim=-1)
    cos_b = torch.cat(collected_cos_beta, dim=-1)
    sin_b = torch.cat(collected_sin_beta, dim=-1)

    # Normalize scalar values with quadrature weights merged
    out_vals = _normalize_convolution_tensor_s2(
        out_idx,
        out_vals,
        in_shape,
        out_shape,
        kernel_size,
        quad_weights,
        transpose_normalization=False,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=True,
    )

    # Angular-modulated values share the scalar normalization
    def _finalize(t):
        return (out_vals * t).to(dtype=torch.float32).contiguous()

    out_idx = out_idx.contiguous()
    out_vals = out_vals.to(dtype=torch.float32).contiguous()

    return (
        out_idx,
        out_vals,
        _finalize(cos_g),
        _finalize(sin_g),
        _finalize(cos_p),
        _finalize(sin_p),
        _finalize(cos_b),
        _finalize(sin_b),
    )


def _banded_fft_from_values(
    idx, vals, kernel_size, nlat_in, nlon_in, nlat_out, lat_min, max_bw
):
    """Build a banded FFT tensor from sparse (idx, vals) with known banding."""
    ker_idx = idx[0]
    row_idx = idx[1]
    col_idx = idx[2]
    input_lat = col_idx // nlon_in
    input_lon = col_idx % nlon_in

    psi_banded = torch.zeros(kernel_size, nlat_out, max_bw, nlon_in, dtype=vals.dtype)
    banded_lat = input_lat - lat_min[row_idx]
    psi_banded[ker_idx, row_idx, banded_lat, input_lon] = vals

    return rfft(psi_banded, dim=-1).conj()


def _build_fft_for_basis(
    in_shape,
    out_shape,
    filter_basis,
    grid_in,
    grid_out,
    theta_cutoff,
    basis_norm_mode,
):
    """Build all banded FFT tensors for one FilterBasis.

    Returns:
    -------
    psi_fft : isotropic filter FFT
    cos_gamma_fft, sin_gamma_fft : frame rotation (γ) modulated
    cos_phi_fft, sin_phi_fft : bearing at output (φ) modulated
    cos_beta_fft, sin_beta_fft : bearing at input (β) modulated
    gather_idx : banding gather index
    """
    (
        idx,
        vals,
        vals_cg,
        vals_sg,
        vals_cp,
        vals_sp,
        vals_cb,
        vals_sb,
    ) = _precompute_vector_convolution_tensor_s2(
        in_shape,
        out_shape,
        filter_basis,
        grid_in=grid_in,
        grid_out=grid_out,
        theta_cutoff=theta_cutoff,
        basis_norm_mode=basis_norm_mode,
    )

    K = filter_basis.kernel_size
    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    psi_sparse = _get_psi(K, idx, vals, nlat_in, nlon_in, nlat_out, nlon_out)
    psi_fft, gather_idx = _precompute_psi_banded(psi_sparse, nlat_in, nlon_in)

    lat_min = gather_idx[:, 0]
    max_bw = gather_idx.shape[1]

    def _fft(v):
        return _banded_fft_from_values(
            idx, v, K, nlat_in, nlon_in, nlat_out, lat_min, max_bw
        )

    return (
        psi_fft,
        _fft(vals_cg),
        _fft(vals_sg),
        _fft(vals_cp),
        _fft(vals_sp),
        _fft(vals_cb),
        _fft(vals_sb),
        gather_idx,
    )


def build_vector_psi_fft(
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
    vector_filter_basis: VectorFilterBasis,
    grid_in: str = "equiangular",
    grid_out: str = "equiangular",
    theta_cutoff: float = 0.01 * math.pi,
    basis_norm_mode: str = "mean",
):
    """Build banded FFT tensors for vector DISCO convolution.

    Returns:
    -------
    psi_scalar_fft : (K_s, ...) — isotropic, for ss
    psi_s_cos_γ_fft, psi_s_sin_γ_fft : (K_s, ...) — frame rotation, for vv
    scalar_gather_idx : (nlat_out, bw_s)
    psi_v_cos_φ_fft, psi_v_sin_φ_fft : (K_v, ...) — bearing at output, for sv
    psi_v_cos_β_fft, psi_v_sin_β_fft : (K_v, ...) — bearing at input, for vs
    vector_gather_idx : (nlat_out, bw_v)
    """
    common = dict(
        in_shape=in_shape,
        out_shape=out_shape,
        grid_in=grid_in,
        grid_out=grid_out,
        theta_cutoff=theta_cutoff,
        basis_norm_mode=basis_norm_mode,
    )

    (
        s_fft,
        s_cg,
        s_sg,
        _s_cp,
        _s_sp,
        _s_cb,
        _s_sb,
        s_gather,
    ) = _build_fft_for_basis(filter_basis=vector_filter_basis.scalar_basis, **common)
    (
        _v_fft,
        _v_cg,
        _v_sg,
        v_cp,
        v_sp,
        v_cb,
        v_sb,
        v_gather,
    ) = _build_fft_for_basis(filter_basis=vector_filter_basis.vector_basis, **common)

    return (
        s_fft,
        s_cg,
        s_sg,
        s_gather,
        v_cp,
        v_sp,
        v_cb,
        v_sb,
        v_gather,
    )


class VectorDiscoConvS2(nn.Module):
    """DISCO convolution on S2 with scalar and vector channels.

    Uses a VectorFilterBasis with independent scalar and vector radial
    bases. Scalar filters (K_s) are used for ss and vv paths; vector
    filters (K_v) are used for sv and vs paths.

    Can be constructed with either a VectorFilterBasis or a kernel_shape
    (which creates a VectorFilterBasis with equal scalar/vector bases).
    """

    def __init__(
        self,
        in_channels_scalar: int,
        in_channels_vector: int,
        out_channels_scalar: int,
        out_channels_vector: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        vector_filter_basis: VectorFilterBasis | None = None,
        kernel_shape: int | tuple[int, ...] | None = None,
        basis_type: str = "piecewise linear",
        basis_norm_mode: str = "mean",
        grid_in: str = "equiangular",
        grid_out: str = "equiangular",
        bias: bool = True,
        theta_cutoff: float | None = None,
    ):
        super().__init__()

        if vector_filter_basis is not None:
            vfb = vector_filter_basis
        elif kernel_shape is not None:
            vfb = get_vector_filter_basis(kernel_shape, kernel_shape, basis_type)
        else:
            raise ValueError("provide vector_filter_basis or kernel_shape")

        self.in_channels_scalar = in_channels_scalar
        self.in_channels_vector = in_channels_vector
        self.out_channels_scalar = out_channels_scalar
        self.out_channels_vector = out_channels_vector
        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        if self.nlon_in % self.nlon_out != 0:
            raise ValueError("nlon_in must be divisible by nlon_out")

        K_s = vfb.scalar_kernel_size
        K_v = vfb.vector_kernel_size
        self._scalar_kernel_size = K_s
        self._vector_kernel_size = K_v

        if theta_cutoff is None:
            theta_cutoff = torch.pi / float(self.nlat_out - 1)
        self.theta_cutoff = theta_cutoff
        if self.theta_cutoff <= 0.0:
            raise ValueError("theta_cutoff must be positive")

        (
            psi_s,
            psi_s_cg,
            psi_s_sg,
            s_gather,
            psi_v_cp,
            psi_v_sp,
            psi_v_cb,
            psi_v_sb,
            v_gather,
        ) = build_vector_psi_fft(
            in_shape,
            out_shape,
            vfb,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=self.theta_cutoff,
            basis_norm_mode=basis_norm_mode,
        )
        # Scalar basis: isotropic + frame rotation (γ)
        self.register_buffer("psi_scalar_fft", psi_s, persistent=False)
        self.register_buffer("psi_s_cos_g_fft", psi_s_cg, persistent=False)
        self.register_buffer("psi_s_sin_g_fft", psi_s_sg, persistent=False)
        # Vector basis: bearing at output (φ) and bearing at input (β)
        self.register_buffer("psi_v_cos_p_fft", psi_v_cp, persistent=False)
        self.register_buffer("psi_v_sin_p_fft", psi_v_sp, persistent=False)
        self.register_buffer("psi_v_cos_b_fft", psi_v_cb, persistent=False)
        self.register_buffer("psi_v_sin_b_fft", psi_v_sb, persistent=False)
        # Integer index tensors as plain attrs for DDP compatibility
        self.scalar_gather_idx = s_gather
        self.vector_gather_idx = v_gather

        # Weight init: scale so total output variance ≈ 1
        N_s_in = in_channels_scalar
        N_v_in = in_channels_vector
        N_s_out = out_channels_scalar
        N_v_out = out_channels_vector

        s_fan = max(1, N_s_in * K_s + 2 * N_v_in * K_v)
        v_fan = max(1, 2 * N_s_in * K_v + 2 * N_v_in * K_s)
        s_scale = 1.0 / math.sqrt(s_fan)
        v_scale = 1.0 / math.sqrt(v_fan)

        self.W_ss = nn.Parameter(s_scale * torch.randn(N_s_out, N_s_in, K_s))
        self.W_vs = nn.Parameter(s_scale * torch.randn(N_s_out, N_v_in, K_v, 2))
        self.W_sv = nn.Parameter(v_scale * torch.randn(N_v_out, N_s_in, K_v, 2))
        self.W_vv = nn.Parameter(v_scale * torch.randn(N_v_out, N_v_in, K_s, 2))

        # W_vs2: diagonal scalar-grad dot vector → scalar (advection-type).
        # Only created when N_s_in == N_s_out and both scalar/vector channels
        # are present.  Shape (N_s, N_v_in, K_v, 2) where the last dim indexes
        # {V·∇s  (d=0),  V·∇⊥s  (d=1)}.  Each output scalar channel o
        # receives only the contribution from input scalar channel c = o.
        if N_s_in == N_s_out and N_s_in > 0 and N_v_in > 0:
            self.W_vs2 = nn.Parameter(s_scale * torch.randn(N_s_in, N_v_in, K_v, 2))
        else:
            self.register_parameter("W_vs2", None)

        if bias and N_s_out > 0:
            self.bias_scalar = nn.Parameter(torch.zeros(N_s_out))
        else:
            self.bias_scalar = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.scalar_gather_idx = fn(self.scalar_gather_idx)
        self.vector_gather_idx = fn(self.vector_gather_idx)
        return self

    def extra_repr(self):
        return (
            f"in_s={self.in_channels_scalar}, "
            f"in_v={self.in_channels_vector}, "
            f"out_s={self.out_channels_scalar}, "
            f"out_v={self.out_channels_vector}, "
            f"in_shape={self.nlat_in, self.nlon_in}, "
            f"out_shape={self.nlat_out, self.nlon_out}, "
            f"K_s={self._scalar_kernel_size}, "
            f"K_v={self._vector_kernel_size}, "
            f"theta_cutoff={self.theta_cutoff}"
        )

    def forward(
        self,
        x_scalar: torch.Tensor,
        x_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_scalar: (B, N_s_in, H, W)
            x_vector: (B, N_v_in, H, W, 2) where last dim is (u, v)

        Returns:
            y_scalar: (B, N_s_out, H, W)
            y_vector: (B, N_v_out, H, W, 2)
        """
        nlon_out = self.nlon_out
        N_s_in = self.in_channels_scalar
        N_v_in = self.in_channels_vector
        B = x_scalar.shape[0]
        K_s = self._scalar_kernel_size
        K_v = self._vector_kernel_size
        H, W = self.nlat_out, self.nlon_out
        s_g = self.scalar_gather_idx
        v_g = self.vector_gather_idx

        # --- Scalar input contractions (3 calls) ---
        if N_s_in > 0:
            # Isotropic filter on scalars (for ss)
            f_s = _disco_s2_contraction_fft(
                x_scalar, self.psi_scalar_fft, s_g, nlon_out
            )
            # Bearing (φ) filter on scalars (for sv)
            f_cp = _disco_s2_contraction_fft(
                x_scalar, self.psi_v_cos_p_fft, v_g, nlon_out
            )
            f_sp = _disco_s2_contraction_fft(
                x_scalar, self.psi_v_sin_p_fft, v_g, nlon_out
            )
        else:
            f_s = x_scalar.new_zeros(B, 0, K_s, H, W)
            f_cp = x_scalar.new_zeros(B, 0, K_v, H, W)
            f_sp = f_cp

        # --- Vector input contractions (4 calls) ---
        if N_v_in > 0:
            u = x_vector[..., 0]
            v = x_vector[..., 1]
            uv = torch.cat([u, v], dim=1)

            # Frame rotation (γ) on vectors (for vv)
            cg_uv = _disco_s2_contraction_fft(uv, self.psi_s_cos_g_fft, s_g, nlon_out)
            sg_uv = _disco_s2_contraction_fft(uv, self.psi_s_sin_g_fft, s_g, nlon_out)
            conv_u = cg_uv[:, :N_v_in] - sg_uv[:, N_v_in:]
            conv_v = sg_uv[:, :N_v_in] + cg_uv[:, N_v_in:]

            # Bearing at input (β) on vectors (for vs)
            cb_uv = _disco_s2_contraction_fft(uv, self.psi_v_cos_b_fft, v_g, nlon_out)
            sb_uv = _disco_s2_contraction_fft(uv, self.psi_v_sin_b_fft, v_g, nlon_out)
            div = cb_uv[:, :N_v_in] + sb_uv[:, N_v_in:]
            curl = cb_uv[:, N_v_in:] - sb_uv[:, :N_v_in]
        else:
            conv_u = x_scalar.new_zeros(B, 0, K_s, H, W)
            conv_v = conv_u
            div = x_scalar.new_zeros(B, 0, K_v, H, W)
            curl = div

        # --- Scalar output ---
        y_s = torch.einsum("ock,bckxy->boxy", self.W_ss, f_s)
        dc = torch.stack([div, curl], dim=-1)
        y_s = y_s + torch.einsum("ockd,bckxyd->boxy", self.W_vs, dc)

        # W_vs2: V·∇s diagonal advection term.
        # grad_comp[b,c_v,c_s,k,x,y] = u*f_cp + v*f_sp  ≈ V·∇s
        # perp_comp                   = -u*f_sp + v*f_cp ≈ V·∇⊥s
        if self.W_vs2 is not None and N_v_in > 0 and N_s_in > 0:
            u_e = u[:, :, None, None, :, :]  # (B, N_v_in, 1,      1,    H, W)
            v_e = v[:, :, None, None, :, :]
            fcp = f_cp[:, None, :, :, :, :]  # (B, 1,      N_s_in, K_v, H, W)
            fsp = f_sp[:, None, :, :, :, :]
            grad_comp = u_e * fcp + v_e * fsp  # (B, N_v_in, N_s_in, K_v, H, W)
            perp_comp = v_e * fcp - u_e * fsp
            adv_dc = torch.stack([grad_comp, perp_comp], dim=-1)
            # einsum: (c, v, k, d), (B, v, c, k, H, W, d) -> (B, c, H, W)
            y_s = y_s + torch.einsum("cvkd,bvckxyd->bcxy", self.W_vs2, adv_dc)

        if self.bias_scalar is not None:
            y_s = y_s + self.bias_scalar.reshape(1, -1, 1, 1)

        # --- Vector output ---
        # scalar → vector (bearing φ filter: gradient + perp gradient)
        M_u_s = torch.stack([f_cp, -f_sp], dim=-1)
        M_v_s = torch.stack([f_sp, f_cp], dim=-1)
        y_u = torch.einsum("ockd,bckxyd->boxy", self.W_sv, M_u_s)
        y_v = torch.einsum("ockd,bckxyd->boxy", self.W_sv, M_v_s)

        # vector → vector (scalar filter: stretch + 90° rotation)
        M_u_v = torch.stack([conv_u, -conv_v], dim=-1)
        M_v_v = torch.stack([conv_v, conv_u], dim=-1)
        y_u = y_u + torch.einsum("ockd,bckxyd->boxy", self.W_vv, M_u_v)
        y_v = y_v + torch.einsum("ockd,bckxyd->boxy", self.W_vv, M_v_v)

        y_vector = torch.stack([y_u, y_v], dim=-1)
        return y_s, y_vector
