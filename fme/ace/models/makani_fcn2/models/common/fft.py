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
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super().__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fft_handle(x, s=(self.nlat, self.nlon), dim=(-2, -1), norm="ortho")

        if self.truncate:
            y = torch.cat(
                (
                    y[..., : self.lmax_high, : self.mmax],
                    y[..., -self.lmax_low :, : self.mmax],
                ),
                dim=-2,
            )

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):
        super().__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = torch.fft.irfft2(xt, s=(self.nlat, self.nlon), dim=(-2, -1), norm="ortho")

        return out
