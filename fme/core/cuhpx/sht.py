# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
import torch.cuda.nvtx
import torch.nn as nn

from fme.core.cuhpx.tools import (
    _precompute_legpoly,
    healpix_irfft_bluestein,
    healpix_irfft_torch,
    healpix_rfft_bluestein,
    healpix_rfft_torch,
    healpix_weights,
)
from fme.core.device import get_device


class SHT(nn.Module):
    def __init__(
        self,
        nside,
        lmax: int,
        mmax: int,
        grid="healpix",
        quad_weights="ring",
        norm="ortho",
        csphase=True,
        use_bluestein=False,
    ):
        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside
        self.quad_weights = quad_weights
        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, w = healpix_weights(nside, self.quad_weights)
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        tq = np.flip(np.arccos(cost))
        self.mmax = mmax or (self.nlon // 2 + 1)
        weights = torch.from_numpy(w)

        pct = _precompute_legpoly(
            self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase
        )
        pct = torch.from_numpy(pct)

        weights = torch.einsum("mlk,k->mlk", pct, weights)
        self.weights = weights.float().to(get_device())

    def forward(self, x: torch.Tensor):
        if torch.is_complex(x):
            raise ValueError("Input tensor must be real.")

        if self.use_bluestein:
            x = healpix_rfft_bluestein(x, self.mmax, self.nside)
        else:
            x = healpix_rfft_torch(x, self.mmax, self.nside)
        x = torch.view_as_real(x)

        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax

        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction
        xout[..., 0] = torch.einsum(
            "...km,mlk->...lm", x[..., : self.mmax, 0], self.weights
        )
        xout[..., 1] = torch.einsum(
            "...km,mlk->...lm", x[..., : self.mmax, 1], self.weights
        )
        x = torch.view_as_complex(xout)

        return x


class iSHT(nn.Module):
    def __init__(
        self,
        nside,
        lmax=None,
        mmax=None,
        grid="healpix",
        norm="ortho",
        csphase=True,
        use_bluestein=False,
    ):
        super().__init__()

        self.nside = nside
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.nlat = 4 * nside - 1
        self.nlon = 4 * nside

        self.use_bluestein = use_bluestein

        if self.grid == "healpix":
            cost, _ = healpix_weights(nside, "none")
            self.lmax = lmax or self.nlat
        else:
            raise (ValueError("Unknown quadrature mode"))

        t = np.flip(np.arccos(cost))

        self.mmax = mmax or (self.nlon // 2 + 1)
        pct = _precompute_legpoly(
            self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase
        )
        pct = torch.from_numpy(pct)

        self.pct = pct.float().to(get_device())

    def forward(self, x: torch.Tensor):
        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk->...km", x[..., 0], self.pct)
        im = torch.einsum("...lm, mlk->...km", x[..., 1], self.pct)
        xs = torch.stack((rl, im), -1)

        x = torch.view_as_complex(xs)

        if self.use_bluestein:
            x = healpix_irfft_bluestein(x, self.mmax, self.nside)
        else:
            x = healpix_irfft_torch(x, self.mmax, self.nside)

        return x
