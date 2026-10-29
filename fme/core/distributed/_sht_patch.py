"""Compile-safe distributed spherical harmonic transforms.

``torch_harmonics.distributed.DistributedRealSHT`` and
``DistributedInverseRealSHT`` pass the output of ``torch.view_as_complex``
straight into a ``distributed_transpose`` (the polar transpose in the forward
transform when latitude and longitude are both split, the azimuth transpose in
the inverse transform when longitude is split and latitude is not). Under
``torch.compile``, AOT autograd regenerates that complex view of a real tensor
with the wrong storage size and raises "setStorage ... out of bounds" on the
second call, so a compiled network fails under spatial parallelism. The
forward passes below are forked from torch-harmonics 0.8.0 (BSD-3-Clause,
Copyright (c) 2022 The torch-harmonics Authors) and differ only in cloning
that view, so the transposes receive a tensor that owns its storage.
"""

import torch
import torch_harmonics.distributed as thd


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

        # cast to complex; cloned so the transpose below never receives a view
        x = torch.view_as_complex(xs).clone()

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

        # inverse FFT; cloned so the transposes below never receive a view
        x = torch.view_as_complex(xs).clone()

        if self.comm_size_polar > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_polar)
            x = thd.distributed_transpose_polar.apply(x, (-2, -3), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = thd.distributed_transpose_azimuth.apply(x, (-3, -1), self.m_shapes)

        # set DCT and nyquist frequencies to 0:
        x[..., 0].imag = 0.0
        if (self.nlon % 2 == 0) and (self.nlon // 2 < x.shape[-1]):
            x[..., self.nlon // 2].imag = 0.0

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = thd.compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = thd.distributed_transpose_azimuth.apply(x, (-1, -3), chan_shapes)

        return x
