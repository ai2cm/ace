# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause
#
# Subset: only the functions needed by DISCO convolution precomputation.

import math

import numpy as np
import torch

from ._cache import lru_cache


def _trapezoidal_weights(
    n: int, a: float = -1.0, b: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    xlg = torch.as_tensor(np.linspace(a, b, n, endpoint=True))
    wlg = (b - a) / n * torch.ones(n, requires_grad=False)
    wlg[0] *= 0.5
    wlg[-1] *= 0.5
    return xlg, wlg


def _legendre_gauss_weights(
    n: int, a: float = -1.0, b: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = torch.as_tensor(xlg).clone()
    wlg = torch.as_tensor(wlg).clone()
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    return xlg, wlg


def _clenshaw_curtiss_weights(
    n: int, a: float = -1.0, b: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    assert n > 1

    tcc = torch.cos(
        torch.linspace(math.pi, 0, n, dtype=torch.float64, requires_grad=False)
    )

    if n == 2:
        wcc = torch.as_tensor([1.0, 1.0], dtype=torch.float64)
    else:
        n1 = n - 1
        N = torch.arange(1, n1, 2, dtype=torch.float64)
        ll = len(N)
        m = n1 - ll

        v = torch.cat(
            [
                2 / N / (N - 2),
                1 / N[-1:],
                torch.zeros(m, dtype=torch.float64, requires_grad=False),
            ]
        )
        v = 0 - v[:-1] - torch.flip(v[1:], dims=(0,))

        g0 = -torch.ones(n1, dtype=torch.float64, requires_grad=False)
        g0[ll] = g0[ll] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1**2 - 1 + (n1 % 2))
        wcc = torch.fft.ifft(v + g).real
        wcc = torch.cat((wcc, wcc[:1]))

    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5
    return tcc, wcc


def _precompute_quadrature_weights(
    n: int, grid: str = "equiangular", a: float = -1.0, b: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    if grid == "equidistant":
        return _trapezoidal_weights(n, a=a, b=b)
    elif grid == "legendre-gauss":
        return _legendre_gauss_weights(n, a=a, b=b)
    elif grid == "equiangular":
        return _clenshaw_curtiss_weights(n, a=a, b=b)
    else:
        raise ValueError(f"Unknown grid type {grid}")


@lru_cache(typed=True, copy=True)
def precompute_longitudes(nlon: int) -> torch.Tensor:
    return torch.linspace(
        0, 2 * math.pi, nlon + 1, dtype=torch.float64, requires_grad=False
    )[:-1]


@lru_cache(typed=True, copy=True)
def precompute_latitudes(
    nlat: int, grid: str = "equiangular"
) -> tuple[torch.Tensor, torch.Tensor]:
    xlg, wlg = _precompute_quadrature_weights(nlat, grid=grid, a=-1.0, b=1.0)
    lats = torch.flip(torch.arccos(xlg), dims=(0,)).clone()
    wlg = torch.flip(wlg, dims=(0,)).clone()
    return lats, wlg
