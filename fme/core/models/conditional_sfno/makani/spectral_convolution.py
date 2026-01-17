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

import torch
import torch.nn as nn
from torch import amp

# import convenience functions for factorized tensors
from .factorizations import get_contract_fun


class SpectralConv(nn.Module):
    """
    Spectral Convolution implemented via SHT or FFT. Designed for convolutions on the
    two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports
    convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        num_groups=1,
        operator_type="dhconv",
        separable=False,
        bias=False,
        gain=1.0,
    ):
        super().__init__()

        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

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

        weight_shape = [num_groups, in_channels // num_groups]

        if not self.separable:
            weight_shape += [out_channels // num_groups]

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
        scale = math.sqrt(gain / (in_channels // num_groups)) * torch.ones(
            self.modes_lat_local, dtype=torch.complex64
        )
        # seemingly the first weight is not really complex, so we need to
        # account for that
        scale[0] *= math.sqrt(2.0)
        init = scale * torch.randn(*weight_shape, dtype=torch.complex64)
        self.weight = nn.Parameter(init)

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

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))

    def forward(self, x):
        dtype = x.dtype
        residual = x
        x = x.float()

        with amp.autocast(device_type="cuda", enabled=False):
            x = self.forward_transform(x).contiguous()
            if self.scale_residual:
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        B, C, H, W = x.shape
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)
        xp = self._contract(
            x, self.weight, separable=self.separable, operator_type=self.operator_type
        )
        x = xp.reshape(B, self.out_channels, H, W).contiguous()

        with amp.autocast(device_type="cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.to(dtype=dtype)

        return x, residual
