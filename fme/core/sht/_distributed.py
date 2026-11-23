"""Distributed SHT and DISCO helpers; import from ``fme.core.sht``.

``DistributedRealSHT``/``DistributedInverseRealSHT`` fork the torch-harmonics
0.8.0 ``forward`` (BSD-3-Clause, Copyright (c) 2022 The torch-harmonics Authors)
to clone the ``view_as_complex`` result before a ``distributed_transpose``;
otherwise ``torch.compile`` on torch 2.7-2.9 raises "setStorage ... out of
bounds". The inverse zeroes DC/Nyquist via ``view_as_real``, since dynamo
cannot trace upstream's ``.imag`` setattr.

``DistributedDiscreteContinuousConvS2`` replaces upstream's sparse/CUDA-extension
contraction with fme's FFT contraction so it compiles to one graph.

Recheck on torch-harmonics pin bumps.
"""

import torch
import torch_harmonics.distributed as thd
from torch_harmonics.distributed import (
    DistributedResampleS2,
    compute_split_shapes,
    distributed_transpose_azimuth,
    distributed_transpose_polar,
    init,
    is_initialized,
)

from fme.core.disco._disco_utils import (
    _disco_s2_contraction_fft,
    _precompute_psi_banded,
)

__all__ = [
    "DistributedRealSHT",
    "DistributedInverseRealSHT",
    "DistributedDiscreteContinuousConvS2",
    "DistributedResampleS2",
    "compute_split_shapes",
    "distributed_transpose_azimuth",
    "distributed_transpose_polar",
    "init",
    "is_initialized",
]


def _view_as_complex(xs: torch.Tensor, clone: bool) -> torch.Tensor:
    """``view_as_complex(xs)``, cloned if ``clone`` so it owns its storage."""
    x = torch.view_as_complex(xs)
    return x.clone() if clone else x


class DistributedRealSHT(thd.DistributedRealSHT):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions but got {x.dim()} "
                "instead"
            )

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[-3]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (-3, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")

        # truncate
        x = x[..., : self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = thd.distributed_transpose_polar.apply(x, (-3, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # contraction
        xs = torch.einsum(
            "...kmr,mlk->...lmr", x, self.weights.to(x.dtype)
        ).contiguous()

        # cast to complex; clone before a transpose
        x = _view_as_complex(xs, clone=self.comm_size_polar > 1)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_polar)
            x = thd.distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)

        return x


class DistributedInverseRealSHT(thd.DistributedInverseRealSHT):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"Expected tensor with at least 3 dimensions but got {x.dim()} "
                "instead"
            )

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[-3]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = thd.distributed_transpose_polar.apply(x, (-3, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # einsum
        xs = torch.einsum("...lmr, mlk->...kmr", x, self.pct.to(x.dtype)).contiguous()

        # cast to complex; clone before a transpose
        x = _view_as_complex(
            xs, clone=self.comm_size_polar > 1 or self.comm_size_azimuth > 1
        )

        if self.comm_size_polar > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_polar)
            x = thd.distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (-3, -1), self.m_shapes)

        # set DCT and nyquist frequencies to 0 (view_as_real: dynamo can't trace .imag=)
        torch.view_as_real(x[..., 0])[..., 1] = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < x.shape[-1]):
            torch.view_as_real(x[..., self.nlon // 2])[..., 1] = 0.0

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)

        return x


class DistributedDiscreteContinuousConvS2(thd.DistributedDiscreteContinuousConvS2):
    """Upstream's distributed DISCO convolution on the 2-sphere, contracting
    with fme's FFT-based contraction instead of a sparse tensor or the CUDA
    extension, so that dynamo traces it to a single graph.

    Each rank holds the input latitudes of its polar shard and all output
    latitudes, as upstream does: the contraction produces partial sums over
    the local input latitudes, which are then summed across the polar group
    and split back into latitude shards.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        psi_fft_conj, gather_idx = _precompute_psi_banded(
            self.psi, self.nlat_in_local, self.nlon_in
        )
        # replaces the sparse tensor upstream contracts with
        del self.psi
        # plain attributes rather than buffers, as in fme.core.disco: DDP
        # cannot broadcast complex or integer buffers, and these are
        # deterministic given the rank's polar shard
        self.psi_fft_conj = psi_fft_conj
        self.psi_gather_idx = gather_idx

    def _apply(self, fn, recurse=True):
        # moves the plain-attribute tensors along with .to() / .cuda()
        super()._apply(fn, recurse=recurse)
        self.psi_fft_conj = fn(self.psi_fft_conj)
        self.psi_gather_idx = fn(self.psi_gather_idx)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (1, -1), self.lon_in_shapes)

        x = _disco_s2_contraction_fft(
            x,
            self.psi_fft_conj.to(x.device),
            self.psi_gather_idx.to(x.device),
            self.nlon_out,
        )

        # sum the partial contractions over the polar group, then keep this
        # rank's output latitudes
        x = thd.reduce_from_polar_region(x)
        x = thd.scatter_to_polar_region(x, -2)

        # transpose back, so that lon is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        B, C, K, H, W = x.shape
        x = x.reshape(B, self.groups, self.groupsize, K, H, W)
        out = torch.einsum(
            "bgckxy,gock->bgoxy",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        ).contiguous()
        out = out.reshape(out.shape[0], -1, H, W)

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out
