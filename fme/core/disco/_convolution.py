# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications: stripped down to DiscreteContinuousConvS2 (forward only)
# using the FFT-based contraction as the sole implementation path. CUDA
# kernel and torch sparse-matrix fallbacks are removed.

import abc
import math

import torch
import torch.nn as nn
from torch_harmonics import filter_basis as _filter_basis_module
from torch_harmonics.cache import lru_cache
from torch_harmonics.filter_basis import FilterBasis
from torch_harmonics.quadrature import precompute_latitudes, precompute_longitudes

from ._disco_utils import _disco_s2_contraction_fft, _get_psi, _precompute_psi_banded


def _normalize_convolution_tensor_s2(
    psi_idx,
    psi_vals,
    in_shape,
    out_shape,
    kernel_size,
    quad_weights,
    transpose_normalization=False,
    basis_norm_mode="mean",
    merge_quadrature=False,
    eps=1e-9,
):
    """Normalizes convolution tensor values based on specified normalization
    mode.
    """
    if basis_norm_mode == "none":
        return psi_vals

    idx = torch.stack(
        [
            psi_idx[0],
            psi_idx[1],
            psi_idx[2] // in_shape[1],
            psi_idx[2] % in_shape[1],
        ],
        dim=0,
    )

    ikernel = idx[0]

    if transpose_normalization:
        ilat_out = idx[2]
        ilat_in = idx[1]
        nlat_out = in_shape[0]
        correction_factor = out_shape[1] / in_shape[1]
    else:
        ilat_out = idx[1]
        ilat_in = idx[2]
        nlat_out = out_shape[0]

    q = quad_weights[ilat_in].reshape(-1)

    vnorm = torch.zeros(kernel_size, nlat_out, device=psi_vals.device)
    support = torch.zeros(kernel_size, nlat_out, device=psi_vals.device)

    for ik in range(kernel_size):
        for ilat in range(nlat_out):
            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))
            vnorm[ik, ilat] = torch.sum(psi_vals[iidx].abs() * q[iidx])
            support[ik, ilat] = torch.sum(q[iidx])

    for ik in range(kernel_size):
        for ilat in range(nlat_out):
            iidx = torch.argwhere((ikernel == ik) & (ilat_out == ilat))

            if basis_norm_mode == "individual":
                val = vnorm[ik, ilat]
            elif basis_norm_mode == "mean":
                val = vnorm[ik, :].mean()
            elif basis_norm_mode == "support":
                val = support[ik, ilat]
            elif basis_norm_mode == "none":
                val = 1.0
            else:
                raise ValueError(f"Unknown basis normalization mode {basis_norm_mode}.")

            psi_vals[iidx] = psi_vals[iidx] / (val + eps)

            if merge_quadrature:
                psi_vals[iidx] = psi_vals[iidx] * q[iidx]

    if transpose_normalization and merge_quadrature:
        psi_vals = psi_vals / correction_factor

    return psi_vals


@lru_cache(typed=True, copy=True)
def _precompute_convolution_tensor_s2(
    in_shape: tuple[int],
    out_shape: tuple[int],
    filter_basis: FilterBasis,
    grid_in: str | None = "equiangular",
    grid_out: str | None = "equiangular",
    theta_cutoff: float | None = 0.01 * math.pi,
    theta_eps: float | None = 1e-3,
    transpose_normalization: bool | None = False,
    basis_norm_mode: str | None = "mean",
    merge_quadrature: bool | None = False,
):
    r"""Precomputes the rotated filters at positions using YZY Euler angles."""
    assert len(in_shape) == 2
    assert len(out_shape) == 2

    kernel_size = filter_basis.kernel_size

    nlat_in, nlon_in = in_shape
    nlat_out, nlon_out = out_shape

    lats_in, win = precompute_latitudes(nlat_in, grid=grid_in)
    lats_out, wout = precompute_latitudes(nlat_out, grid=grid_out)
    lons_in = precompute_longitudes(nlon_in)

    if transpose_normalization:
        quad_weights = wout.reshape(-1, 1) / nlon_in / 2.0
    else:
        quad_weights = win.reshape(-1, 1) / nlon_in / 2.0

    theta_cutoff_eff = (1.0 + theta_eps) * theta_cutoff

    out_idx = []
    out_vals = []

    beta = lons_in
    gamma = lats_in.reshape(-1, 1)

    cbeta = torch.cos(beta)
    sbeta = torch.sin(beta)
    cgamma = torch.cos(gamma)
    sgamma = torch.sin(gamma)

    out_roff = torch.zeros(nlat_out + 1, dtype=torch.int64, device=lons_in.device)
    out_roff[0] = 0
    for t in range(nlat_out):
        alpha = -lats_out[t]

        x = torch.cos(alpha) * cbeta * sgamma + cgamma * torch.sin(alpha)
        y = sbeta * sgamma
        z = -cbeta * torch.sin(alpha) * sgamma + torch.cos(alpha) * cgamma

        norm = torch.sqrt(x * x + y * y + z * z)
        x = x / norm
        y = y / norm
        z = z / norm

        theta = torch.arccos(z)
        phi = torch.arctan2(y, x)
        phi = torch.where(phi < 0.0, phi + 2 * torch.pi, phi)

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

        out_idx.append(idx)
        out_vals.append(vals)
        out_roff[t + 1] = out_roff[t] + iidx.shape[0]

    out_idx = torch.cat(out_idx, dim=-1)
    out_vals = torch.cat(out_vals, dim=-1)

    out_vals = _normalize_convolution_tensor_s2(
        out_idx,
        out_vals,
        in_shape,
        out_shape,
        kernel_size,
        quad_weights,
        transpose_normalization=transpose_normalization,
        basis_norm_mode=basis_norm_mode,
        merge_quadrature=merge_quadrature,
    )

    out_idx = out_idx.contiguous()
    out_vals = out_vals.to(dtype=torch.float32).contiguous()

    return out_idx, out_vals, out_roff


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for discrete-continuous convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: int | tuple[int, ...],
        basis_type: str = "piecewise linear",
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.kernel_shape = kernel_shape
        self.filter_basis = _filter_basis_module.get_filter_basis(
            kernel_shape=kernel_shape, basis_type=basis_type
        )

        self.groups = groups

        if in_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of input channels has to be an integer "
                "multiple of the group size"
            )
        if out_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of output channels has to be an integer "
                "multiple of the group size"
            )
        self.groupsize = in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize / self.kernel_size)
        self.weight = nn.Parameter(
            scale * torch.randn(out_channels, self.groupsize, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @property
    def kernel_size(self):
        return self.filter_basis.kernel_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConvS2(DiscreteContinuousConv):
    """Discrete-continuous (DISCO) convolution on the 2-Sphere.

    Uses an FFT-based contraction for efficient computation.  Forked from
    torch-harmonics; see Ocampo, Price, McEwen, *Scalable and equivariant
    spherical CNNs by discrete-continuous (DISCO) convolutions*, ICLR 2023.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: tuple[int, int],
        out_shape: tuple[int, int],
        kernel_shape: int | tuple[int, ...],
        basis_type: str = "piecewise linear",
        basis_norm_mode: str = "mean",
        groups: int = 1,
        grid_in: str = "equiangular",
        grid_out: str = "equiangular",
        bias: bool = True,
        theta_cutoff: float | None = None,
    ):
        super().__init__(
            in_channels, out_channels, kernel_shape, basis_type, groups, bias
        )

        self.nlat_in, self.nlon_in = in_shape
        self.nlat_out, self.nlon_out = out_shape

        assert self.nlon_in % self.nlon_out == 0

        if theta_cutoff is None:
            self.theta_cutoff = torch.pi / float(self.nlat_out - 1)
        else:
            self.theta_cutoff = theta_cutoff

        if self.theta_cutoff <= 0.0:
            raise ValueError("Error, theta_cutoff has to be positive.")

        idx, vals, _ = _precompute_convolution_tensor_s2(
            in_shape,
            out_shape,
            self.filter_basis,
            grid_in=grid_in,
            grid_out=grid_out,
            theta_cutoff=self.theta_cutoff,
            transpose_normalization=False,
            basis_norm_mode=basis_norm_mode,
            merge_quadrature=True,
        )

        ker_idx = idx[0, ...].contiguous()
        row_idx = idx[1, ...].contiguous()
        col_idx = idx[2, ...].contiguous()
        vals = vals.contiguous()

        self.register_buffer("psi_ker_idx", ker_idx, persistent=False)
        self.register_buffer("psi_row_idx", row_idx, persistent=False)
        self.register_buffer("psi_col_idx", col_idx, persistent=False)
        self.register_buffer("psi_vals", vals, persistent=False)

        # Precompute banded FFT of psi for FFT-based contraction
        psi_sparse = _get_psi(
            self.kernel_size,
            self.psi_idx,
            self.psi_vals,
            self.nlat_in,
            self.nlon_in,
            self.nlat_out,
            self.nlon_out,
        )
        psi_fft_conj, gather_idx = _precompute_psi_banded(
            psi_sparse, self.nlat_in, self.nlon_in
        )
        self.register_buffer("psi_fft_conj", psi_fft_conj, persistent=False)
        self.register_buffer("psi_gather_idx", gather_idx, persistent=False)

    def extra_repr(self):
        return (
            f"in_shape={(self.nlat_in, self.nlon_in)}, "
            f"out_shape={(self.nlat_out, self.nlon_out)}, "
            f"in_chans={self.groupsize * self.groups}, "
            f"out_chans={self.weight.shape[0]}, "
            f"filter_basis={self.filter_basis}, "
            f"kernel_shape={self.kernel_shape}, "
            f"theta_cutoff={self.theta_cutoff}, "
            f"groups={self.groups}"
        )

    @property
    def psi_idx(self):
        return torch.stack(
            [self.psi_ker_idx, self.psi_row_idx, self.psi_col_idx], dim=0
        ).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _disco_s2_contraction_fft(
            x,
            self.psi_fft_conj.to(x.device),
            self.psi_gather_idx.to(x.device),
            self.nlon_out,
        )

        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)

        out = torch.einsum(
            "bgckxy,gock->bgoxy",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        ).contiguous()
        out = out.reshape(B, -1, H, W)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
