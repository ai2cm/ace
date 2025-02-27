# flake8: noqa
# Copied from tag v0.1.0 https://github.com/NVIDIA/modulus-makani/tree/0218658e925a3c88c6513022a33ecc1647735c2e
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

# import FactorizedTensor from tensorly for tensorized operations
import tensorly as tl
import torch
import torch.nn as nn

tl.set_backend("pytorch")

# import convenience functions for factorized tensors
from tltorch.factorized_tensors.core import FactorizedTensor

# for the experimental module
from .contractions import _contract_rank
from .factorizations import get_contract_fun


class SpectralConv(nn.Module):
    """
    Spectral Convolution implemented via SHT or FFT. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        separable=False,
        bias=False,
        gain=1.0,
    ):
        super(SpectralConv, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)
        if hasattr(self.forward_transform, "grid"):
            self.scale_residual = self.scale_residual or (
                self.forward_transform.grid != self.inverse_transform.grid
            )

        # remember factorization details
        self.operator_type = operator_type
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        self.modes_lat_local = self.modes_lat
        self.modes_lon_local = self.modes_lon
        self.nlat_local = self.inverse_transform.nlat
        self.nlon_local = self.inverse_transform.nlon

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        # Compute scaling factor for correct initialization
        scale = math.sqrt(gain / in_channels) * torch.ones(
            self.modes_lat_local, dtype=torch.complex64
        )
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0] *= math.sqrt(2.0)
        init = scale * torch.randn(*weight_shape, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(init))

        if self.operator_type == "dhconv":
            self.weight.is_shared_mp = ["matmul", "w"]
            self.weight.sharded_dims_mp = [None for _ in weight_shape]
            self.weight.sharded_dims_mp[-1] = "h"
        else:
            self.weight.is_shared_mp = ["matmul"]
            self.weight.sharded_dims_mp = [None for _ in weight_shape]
            self.weight.sharded_dims_mp[-1] = "w"
            self.weight.sharded_dims_mp[-2] = "h"

        # get the contraction handle. This should return a pyTorch contraction
        self._contract = get_contract_fun(
            self.weight,
            implementation="factorized",
            separable=separable,
            complex=True,
            operator_type=operator_type,
        )

        if bias == "constant":
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        elif bias == "position":
            self.bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.nlat_local, self.nlon_local)
            )
            self.bias.is_shared_mp = ["matmul"]
            self.bias.sharded_dims_mp = [None, None, "h", "w"]

    def forward(self, x):
        dtype = x.dtype
        residual = x
        x = x.float()
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x).contiguous()
            if self.scale_residual:
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # approach with unpadded weights
        weight = torch.view_as_complex(self.weight)
        xp = self._contract(
            x, weight, separable=self.separable, operator_type=self.operator_type
        )
        x = xp.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.to(dtype=dtype)

        return x, residual


class FactorizedSpectralConv(nn.Module):
    """
    Factorized version of SpectralConv. Uses tensorly-torch to keep the weights factorized
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
        gain=1.0,
    ):
        super(FactorizedSpectralConv, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)
        if hasattr(self.forward_transform, "grid"):
            self.scale_residual = self.scale_residual or (
                self.forward_transform.grid != self.inverse_transform.grid
            )

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "ComplexDense"  # No factorization
        complex_weight = factorization[:7].lower() == "complex"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        self.modes_lat_local = self.modes_lat
        self.modes_lon_local = self.modes_lon

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        elif self.operator_type == "rank":
            weight_shape += [self.rank]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        # form weight tensors
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=self.rank,
            factorization=factorization,
            fixed_rank_modes=False,
            **decomposition_kwargs,
        )
        # initialization of weights
        scale = math.sqrt(gain / float(weight_shape[0]))
        self.weight.normal_(mean=0.0, std=scale)

        # get the contraction handle
        if operator_type == "rank":
            self._contract = _contract_rank
        else:
            self._contract = get_contract_fun(
                self.weight,
                implementation="reconstructed",
                separable=separable,
                complex=complex_weight,
                operator_type=operator_type,
            )

        if bias == "constant":
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        elif bias == "position":
            self.bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.nlat_local, self.nlon_local)
            )
            self.bias.is_shared_mp = ["matmul"]
            self.bias.sharded_dims_mp = [None, None, "h", "w"]

    def forward(self, x):
        dtype = x.dtype
        residual = x
        x = x.float()

        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x).contiguous()
            if self.scale_residual:
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        if self.operator_type == "rank":
            xp = self._contract(x, self.weight, self.lat_weight, self.lon_weight)
        else:
            xp = self._contract(
                x,
                self.weight,
                separable=self.separable,
                operator_type=self.operator_type,
            )
        x = xp.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x, residual
