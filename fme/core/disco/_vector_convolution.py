import math

import torch

from ._cache import lru_cache
from ._convolution import _normalize_convolution_tensor_s2
from ._disco_utils import _get_psi, _precompute_psi_banded
from ._fft import rfft
from ._filter_basis import FilterBasis
from ._quadrature import precompute_latitudes, precompute_longitudes


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

    Like _precompute_convolution_tensor_s2, but additionally computes
    cos(γ) and sin(γ) (frame rotation angle) at each support point.

    Returns:
    -------
    out_idx : (3, nnz) long tensor
        Indices [kernel, out_lat, in_lat*nlon_in + in_lon].
    out_vals_scalar : (nnz,) float tensor
        Normalized ψ_k(r) values with quadrature weights merged.
    out_vals_cos : (nnz,) float tensor
        ψ_k(r)·cos(γ) values (normalized, quadrature merged).
    out_vals_sin : (nnz,) float tensor
        ψ_k(r)·sin(γ) values (normalized, quadrature merged).
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
    collected_cos = []
    collected_sin = []

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

        support_cos = cos_gamma_frame[iidx[:, 1], iidx[:, 2]]
        support_sin = sin_gamma_frame[iidx[:, 1], iidx[:, 2]]

        idx = torch.stack(
            [
                iidx[:, 0],
                t * torch.ones_like(iidx[:, 0]),
                iidx[:, 1] * nlon_in + iidx[:, 2],
            ],
            dim=0,
        )

        collected_idx.append(idx)
        collected_vals.append(vals)
        collected_cos.append(support_cos)
        collected_sin.append(support_sin)

    out_idx = torch.cat(collected_idx, dim=-1)
    out_vals = torch.cat(collected_vals, dim=-1)
    out_cos = torch.cat(collected_cos, dim=-1)
    out_sin = torch.cat(collected_sin, dim=-1)

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

    # cos/sin filter values share the scalar normalization
    out_vals_cos = out_vals * out_cos
    out_vals_sin = out_vals * out_sin

    out_idx = out_idx.contiguous()
    out_vals = out_vals.to(dtype=torch.float32).contiguous()
    out_vals_cos = out_vals_cos.to(dtype=torch.float32).contiguous()
    out_vals_sin = out_vals_sin.to(dtype=torch.float32).contiguous()

    return out_idx, out_vals, out_vals_cos, out_vals_sin


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


def build_vector_psi_fft(
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
    filter_basis: FilterBasis,
    grid_in: str = "equiangular",
    grid_out: str = "equiangular",
    theta_cutoff: float = 0.01 * math.pi,
    basis_norm_mode: str = "mean",
):
    """Build banded FFT tensors for vector DISCO convolution.

    Returns:
    -------
    psi_scalar_fft : (K, nlat_out, max_bw, nfreq) complex tensor
    psi_cos_fft : (K, nlat_out, max_bw, nfreq) complex tensor
    psi_sin_fft : (K, nlat_out, max_bw, nfreq) complex tensor
    gather_idx : (nlat_out, max_bw) long tensor
    """
    idx, vals_scalar, vals_cos, vals_sin = _precompute_vector_convolution_tensor_s2(
        in_shape,
        out_shape,
        filter_basis,
        grid_in=grid_in,
        grid_out=grid_out,
        theta_cutoff=theta_cutoff,
        basis_norm_mode=basis_norm_mode,
    )

    kernel_size = filter_basis.kernel_size
    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    # Build scalar banded FFT via the existing pipeline
    psi_scalar_sparse = _get_psi(
        kernel_size, idx, vals_scalar, nlat_in, nlon_in, nlat_out, nlon_out
    )
    psi_scalar_fft, gather_idx = _precompute_psi_banded(
        psi_scalar_sparse, nlat_in, nlon_in
    )

    # Build cos/sin banded FFTs reusing the same banding parameters
    lat_min = gather_idx[:, 0]
    max_bw = gather_idx.shape[1]

    psi_cos_fft = _banded_fft_from_values(
        idx, vals_cos, kernel_size, nlat_in, nlon_in, nlat_out, lat_min, max_bw
    )
    psi_sin_fft = _banded_fft_from_values(
        idx, vals_sin, kernel_size, nlat_in, nlon_in, nlat_out, lat_min, max_bw
    )

    return psi_scalar_fft, psi_cos_fft, psi_sin_fft, gather_idx
