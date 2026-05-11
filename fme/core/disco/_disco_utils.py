# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications: stripped down to FFT-based contraction only, removing
# CUDA kernel wrappers and torch sparse-matrix fallbacks.


import torch

from ._fft import irfft, rfft


def _get_psi(
    kernel_size: int,
    psi_idx: torch.Tensor,
    psi_vals: torch.Tensor,
    nlat_in: int,
    nlon_in: int,
    nlat_out: int,
    nlon_out: int,
    nlat_in_local: int | None = None,
    nlat_out_local: int | None = None,
    semi_transposed: bool | None = False,
):
    """Creates a sparse tensor for spherical harmonic convolution operations."""
    nlat_in_local = nlat_in_local if nlat_in_local is not None else nlat_in
    nlat_out_local = nlat_out_local if nlat_out_local is not None else nlat_out

    if semi_transposed:
        tout = psi_idx[2] // nlon_out
        pout = psi_idx[2] % nlon_out
        pout = nlon_out - 1 - pout
        tin = psi_idx[1]
        idx = torch.stack([psi_idx[0], tout, tin * nlon_out + pout], dim=0)
        psi = torch.sparse_coo_tensor(
            idx, psi_vals, size=(kernel_size, nlat_out_local, nlat_in_local * nlon_out)
        ).coalesce()
    else:
        psi = torch.sparse_coo_tensor(
            psi_idx,
            psi_vals,
            size=(kernel_size, nlat_out_local, nlat_in_local * nlon_in),
        ).coalesce()
    return psi


def _precompute_psi_banded(psi_sparse: torch.Tensor, nlat_in: int, nlon: int):
    """Build a banded dense representation of psi directly from sparse COO data.

    Instead of densifying the full (K, nlat_out, nlat_in, nlon) tensor, this
    only stores the contiguous band of input latitudes that have nonzero entries
    for each output latitude, reducing memory by ~nlat_in/band_width.

    Returns:
    -------
    psi_banded_fft_conj : (K, nlat_out, max_bw, nfreq) complex tensor
    gather_idx : (nlat_out, max_bw) long tensor of input latitude indices
    """
    K, nlat_out, _ = psi_sparse.shape
    psi = psi_sparse.coalesce()
    indices = psi.indices()  # (3, nnz)
    values = psi.values()  # (nnz,)

    ker_idx = indices[0]
    row_idx = indices[1]  # output lat
    col_idx = indices[2]  # input_lat * nlon + lon
    input_lat = col_idx // nlon
    input_lon = col_idx % nlon

    # Find min/max input lat per output lat (across all kernel indices)
    lat_min = torch.full((nlat_out,), nlat_in, dtype=torch.long)
    lat_max = torch.full((nlat_out,), 0, dtype=torch.long)
    lat_min.scatter_reduce_(0, row_idx, input_lat, reduce="amin")
    lat_max.scatter_reduce_(0, row_idx, input_lat, reduce="amax")

    # Handle empty rows
    empty = lat_min >= nlat_in
    lat_min[empty] = 0
    lat_max[empty] = 0

    max_bw = (lat_max - lat_min + 1).max().item()

    # Build banded tensor from sparse entries (no full densification)
    psi_banded = torch.zeros(K, nlat_out, max_bw, nlon, dtype=values.dtype)
    banded_lat = input_lat - lat_min[row_idx]
    psi_banded[ker_idx, row_idx, banded_lat, input_lon] = values

    # Precompute FFT and gather index
    psi_banded_fft_conj = rfft(psi_banded, dim=-1).conj()
    gather_idx = lat_min.unsqueeze(1) + torch.arange(max_bw).unsqueeze(0)
    gather_idx = gather_idx.clamp(max=nlat_in - 1)

    return psi_banded_fft_conj, gather_idx


def _disco_s2_contraction_fft(
    x: torch.Tensor,
    psi_fft_conj: torch.Tensor,
    gather_idx: torch.Tensor,
    nlon_out: int,
):
    """FFT-based DISCO S2 contraction using banded psi representation.

    Parameters
    ----------
    x : (B, C, nlat_in, nlon_in)
    psi_fft_conj : (K, nlat_out, bw, nfreq)
    gather_idx : (nlat_out, bw)
    nlon_out : int

    Returns:
    -------
    (B, C, K, nlat_out, nlon_out)
    """
    batch_size, n_chans, nlat_in, nlon_in = x.shape
    kernel_size, nlat_out, bw, nfreq = psi_fft_conj.shape
    pscale = nlon_in // nlon_out

    # FFT of input along longitude
    X_f = rfft(x.to(torch.float32), dim=-1)  # (B, C, nlat_in, nfreq)
    X_f = X_f.reshape(batch_size * n_chans, nlat_in, nfreq)

    # Gather relevant input lats for each output lat
    X_f_gathered = X_f[:, gather_idx, :]  # (B*C, nlat_out, bw, nfreq)

    # Cross-correlate: einsum over band width and frequency, then irfft
    Y_f = torch.einsum("kowf,bowf->bkof", psi_fft_conj, X_f_gathered)

    # Inverse FFT
    y = irfft(Y_f, n=nlon_in, dim=-1)  # (B*C, K, nlat_out, nlon_in)

    # Subsample for stride
    y = y[..., ::pscale]  # (B*C, K, nlat_out, nlon_out)

    y = y.reshape(batch_size, n_chans, kernel_size, nlat_out, nlon_out).contiguous()
    return y.to(x.dtype)
