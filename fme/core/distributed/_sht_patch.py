"""Compile-safe distributed inverse spherical harmonic transform.

``torch_harmonics.distributed.DistributedInverseRealSHT`` passes the output of
``torch.view_as_complex`` straight into its azimuth ``all_to_all`` transpose
when the polar dimension is not split. Under ``torch.compile``, AOT autograd
regenerates that complex view of a real tensor with the wrong storage size
and raises "setStorage ... out of bounds", so a compiled network fails under
spatial parallelism that splits only longitude. The forward pass below is
forked from torch-harmonics 0.8.0 (BSD-3-Clause, Copyright (c) 2022 The
torch-harmonics Authors) and differs only in cloning that view, so the
transpose receives a tensor that owns its storage.
"""

import torch
import torch_harmonics.distributed as thd


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
