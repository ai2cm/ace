# flake8: noqa
# fmt: off
# isort: skip_file

"""
This file contains a fix that we needed to get the SFNO to work on multiple
unroll steps in multiprocessing (e.g. multi-GPU mode.) We forked this code from
the torch harmonics sht.py file [*].

[*] https://github.com/NVIDIA/torch-harmonics/blob/17eefa53468d1a885d72087918eba905fa53e10a/torch_harmonics/sht.py
"""


# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
import torch.nn as nn
import torch.fft

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly

from fme.core.device import get_device
from fme.core.benchmark.timer import Timer, NullTimer


class RealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "healpix":
            raise(NotImplementedError("'healpix' grid not supported by InverseRealVectorSHT"))
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        pct = torch.as_tensor(_precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase))
        weights = torch.einsum('mlk,k->mlk', pct, w)

        # remember quadrature weights
        self.weights = weights.float().to(get_device())

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor, timer: Timer = NullTimer()):

        assert(x.shape[-2] == self.nlat)
        assert(x.shape[-1] == self.nlon)
        with torch.autocast("cuda", enabled=False):
            with timer.child("rfft"):
                # rfft and view_as_complex don't support BF16, see https://github.com/pytorch/pytorch/issues/117844
                x = x.float()

                # apply real fft in the longitudinal direction
                x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

            with timer.child("permute"):
                x = x.transpose(-1, -2).contiguous()  # (B, C, H, W) -> (B, C, W, H)
                x = x.contiguous()

            with timer.child("contraction"):
                # do the Legendre-Gauss quadrature
                x = torch.view_as_real(x)

                # contraction
                weights = self.weights.to(x.device).to(x.dtype)
                rl = torch.einsum('...mk,mlk->...lm', x[..., :self.mmax, :, 0], weights)
                im = torch.einsum('...mk,mlk->...lm', x[..., :self.mmax, :, 1], weights)
                xout = torch.stack((rl, im), -1)
                x = torch.view_as_complex(xout)

        return x

class InverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "healpix":
            raise(NotImplementedError("'healpix' grid not supported by RealVectorSHT"))
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = torch.flip(torch.arccos(cost), dims=(0,))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        pct = torch.as_tensor(_precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase))

        # register buffer
        self.pct = pct.float().to(get_device())

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor, timer: Timer = NullTimer()):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)

        with timer.child("permute"):
            x = x.permute(0, 1, 3, 2).contiguous()  # (B, C, H, W) -> (B, C, W, H)
            x = x.contiguous()

        with torch.autocast("cuda", enabled=False):
            with timer.child("contraction"):
                # irfft and view_as_complex don't support BF16, see https://github.com/pytorch/pytorch/issues/117844
                # Evaluate associated Legendre functions on the output nodes
                x = torch.view_as_real(x).float()

                pct = self.pct.to(x.device).to(x.dtype)
                rl = torch.einsum('...ml, mlk->...km', x[..., 0], pct )
                im = torch.einsum('...ml, mlk->...km', x[..., 1], pct )
                xs = torch.stack((rl, im), -1)

                # apply the inverse (real) FFT
                x = torch.view_as_complex(xs)
            with timer.child("irfft"):
                x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x
