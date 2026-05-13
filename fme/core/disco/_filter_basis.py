# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import math

import torch

from ._cache import lru_cache


def _circle_dist(x1: torch.Tensor, x2: torch.Tensor):
    return torch.minimum(
        torch.abs(x1 - x2), torch.abs(2 * math.pi - torch.abs(x1 - x2))
    )


def _log_factorial(x: torch.Tensor):
    return torch.lgamma(x + 1)


def _factorial(x: torch.Tensor):
    return torch.exp(_log_factorial(x))


class FilterBasis(metaclass=abc.ABCMeta):
    """Abstract base class for a filter basis."""

    kernel_shape: int | list[int] | tuple[int, ...]

    def __init__(self, kernel_shape: int | list[int] | tuple[int, ...]):
        self.kernel_shape = kernel_shape

    def __repr__(self):
        class_name = self.__class__.__name__
        if hasattr(self, "extra_repr"):
            return f"{class_name}({self.extra_repr()})"
        else:
            return f"{class_name}()"

    def extra_repr(self):
        return f"kernel_shape={self.kernel_shape}"

    @property
    @abc.abstractmethod
    def kernel_size(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):
        raise NotImplementedError


@lru_cache(typed=True, copy=False)
def get_filter_basis(
    kernel_shape: int | tuple[int, ...], basis_type: str
) -> FilterBasis:
    """Factory function to generate the appropriate filter basis."""
    if basis_type == "piecewise linear":
        return PiecewiseLinearFilterBasis(kernel_shape=kernel_shape)
    elif basis_type == "morlet":
        return MorletFilterBasis(kernel_shape=kernel_shape)
    elif basis_type == "isotropic morlet":
        return IsotropicMorletFilterBasis(kernel_shape=kernel_shape)
    elif basis_type == "zernike":
        return ZernikeFilterBasis(kernel_shape=kernel_shape)
    else:
        raise ValueError(f"Unknown basis_type {basis_type}")


class PiecewiseLinearFilterBasis(FilterBasis):
    """Tensor-product basis on a disk from piecewise linear basis functions."""

    kernel_shape: list[int]

    def __init__(self, kernel_shape: int | list[int] | tuple[int, ...]):
        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape]
        if len(kernel_shape) == 1:
            kernel_shape = [kernel_shape[0], 1]
        elif len(kernel_shape) != 2:
            raise ValueError(
                f"expected kernel_shape to be of length 1 or 2 "
                f"but got {kernel_shape} instead."
            )
        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):
        return (self.kernel_shape[0] // 2) * self.kernel_shape[1] + self.kernel_shape[
            0
        ] % 2

    def _compute_support_vals_isotropic(
        self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float
    ):
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)
        nr = self.kernel_shape[0]
        dr = 2 * r_cutoff / (nr + 1)

        if nr % 2 == 1:
            ir = ikernel * dr
        else:
            ir = (ikernel + 0.5) * dr

        iidx = torch.argwhere(((r - ir).abs() <= dr) & (r <= r_cutoff))
        vals = 1 - (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs() / dr
        return iidx, vals

    def _compute_support_vals_anisotropic(
        self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float
    ):
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)
        nr = self.kernel_shape[0]
        nphi = self.kernel_shape[1]
        dr = 2 * r_cutoff / (nr + 1)
        dphi = 2.0 * math.pi / nphi

        if nr % 2 == 1:
            ir = ((ikernel - 1) // nphi + 1) * dr
            iphi = ((ikernel - 1) % nphi) * dphi - math.pi
        else:
            ir = (ikernel // nphi + 0.5) * dr
            iphi = (ikernel % nphi) * dphi - math.pi

        if nr % 2 == 1:
            cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
            cond_phi = (ikernel == 0) | (_circle_dist(phi, iphi).abs() <= dphi)
            iidx = torch.argwhere(cond_r & cond_phi)
            dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phi = _circle_dist(phi[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0])
            vals = 1 - dist_r / dr
            vals *= torch.where((iidx[:, 0] > 0), (1 - dist_phi / dphi), 1.0)
        else:
            rn = -r
            phin = torch.where(phi + math.pi >= math.pi, phi - math.pi, phi + math.pi)
            cond_r = ((r - ir).abs() <= dr) & (r <= r_cutoff)
            cond_phi = _circle_dist(phi, iphi).abs() <= dphi
            cond_rn = ((rn - ir).abs() <= dr) & (rn <= r_cutoff)
            cond_phin = _circle_dist(phin, iphi) <= dphi
            iidx = torch.argwhere((cond_r & cond_phi) | (cond_rn & cond_phin))

            dist_r = (r[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phi = _circle_dist(phi[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0])
            dist_rn = (rn[iidx[:, 1], iidx[:, 2]] - ir[iidx[:, 0], 0, 0]).abs()
            dist_phin = _circle_dist(
                phin[iidx[:, 1], iidx[:, 2]], iphi[iidx[:, 0], 0, 0]
            )
            vals = cond_r[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_r / dr)
            vals *= cond_phi[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_phi / dphi)
            valsn = cond_rn[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (1 - dist_rn / dr)
            valsn *= cond_phin[iidx[:, 0], iidx[:, 1], iidx[:, 2]] * (
                1 - dist_phin / dphi
            )
            vals += valsn

        return iidx, vals

    def compute_support_vals(self, r: torch.Tensor, phi: torch.Tensor, r_cutoff: float):
        if self.kernel_shape[1] > 1:
            return self._compute_support_vals_anisotropic(r, phi, r_cutoff=r_cutoff)
        else:
            return self._compute_support_vals_isotropic(r, phi, r_cutoff=r_cutoff)


class MorletFilterBasis(FilterBasis):
    """Morlet-style filter basis on the disk."""

    kernel_shape: list[int]

    def __init__(self, kernel_shape: int | list[int] | tuple[int, ...]):
        if isinstance(kernel_shape, int):
            kernel_shape = [kernel_shape, kernel_shape]
        if len(kernel_shape) != 2:
            raise ValueError(
                f"expected kernel_shape to be of length 2 "
                f"but got {kernel_shape} instead."
            )
        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):
        return self.kernel_shape[0] * self.kernel_shape[1]

    def hann_window(self, r: torch.Tensor, width: float = 1.0):
        return torch.cos(0.5 * torch.pi * r / width) ** 2

    def compute_support_vals(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        r_cutoff: float,
        width: float = 1.0,
    ):
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)
        nkernel = ikernel % self.kernel_shape[1]
        mkernel = ikernel // self.kernel_shape[1]

        iidx = torch.argwhere(
            (r <= r_cutoff)
            & torch.full_like(ikernel, True, dtype=torch.bool, device=r.device)
        )

        r = r[iidx[:, 1], iidx[:, 2]] / r_cutoff
        phi = phi[iidx[:, 1], iidx[:, 2]]
        x = r * torch.sin(phi)
        y = r * torch.cos(phi)
        n = nkernel[iidx[:, 0], 0, 0]
        m = mkernel[iidx[:, 0], 0, 0]

        harmonic = torch.where(
            n % 2 == 1,
            torch.sin(torch.ceil(n / 2) * math.pi * x / width),
            torch.cos(torch.ceil(n / 2) * math.pi * x / width),
        )
        harmonic *= torch.where(
            m % 2 == 1,
            torch.sin(torch.ceil(m / 2) * math.pi * y / width),
            torch.cos(torch.ceil(m / 2) * math.pi * y / width),
        )

        vals = self.hann_window(r, width=width) * harmonic
        return iidx, vals


class IsotropicMorletFilterBasis(FilterBasis):
    """Morlet-style filter basis using only radial modes.

    Each basis function is a product of a Hann radial window and a 1-D
    Fourier harmonic in the normalised radial coordinate ``r / r_cutoff``.
    Because none of the basis functions depend on the azimuthal angle
    ``phi``, any learned linear combination is guaranteed to be isotropic
    (radially symmetric).

    ``kernel_shape`` is a single integer giving the number of radial modes.
    If a tuple is provided, only the first element is used.
    """

    kernel_shape: int

    def __init__(self, kernel_shape: int | list[int] | tuple[int, ...]):
        if isinstance(kernel_shape, list | tuple):
            kernel_shape = kernel_shape[0]
        if not isinstance(kernel_shape, int):
            raise ValueError(
                f"expected kernel_shape to be an integer "
                f"but got {kernel_shape} instead."
            )
        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self) -> int:
        return self.kernel_shape

    def compute_support_vals(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        r_cutoff: float,
        width: float = 1.0,
    ):
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)

        iidx = torch.argwhere(
            (r <= r_cutoff)
            & torch.full_like(ikernel, True, dtype=torch.bool, device=r.device)
        )

        r_norm = r[iidx[:, 1], iidx[:, 2]] / r_cutoff
        n = ikernel[iidx[:, 0], 0, 0]

        # Radial Fourier modes: cos/sin pattern identical to Morlet but in r.
        harmonic = torch.where(
            n % 2 == 1,
            torch.sin(torch.ceil(n / 2) * math.pi * r_norm / width),
            torch.cos(torch.ceil(n / 2) * math.pi * r_norm / width),
        )

        # Hann radial envelope
        window = torch.cos(0.5 * torch.pi * r_norm / width) ** 2
        vals = window * harmonic

        return iidx, vals


class ZernikeFilterBasis(FilterBasis):
    """Zernike polynomial basis defined on the disk."""

    kernel_shape: int

    def __init__(self, kernel_shape: int | list[int] | tuple[int, ...]):
        if isinstance(kernel_shape, list | tuple):
            kernel_shape = kernel_shape[0]
        if not isinstance(kernel_shape, int):
            raise ValueError(
                f"expected kernel_shape to be an integer "
                f"but got {kernel_shape} instead."
            )
        super().__init__(kernel_shape=kernel_shape)

    @property
    def kernel_size(self):
        return (self.kernel_shape * (self.kernel_shape + 1)) // 2

    def zernikeradial(self, r: torch.Tensor, n: torch.Tensor, m: torch.Tensor):
        out = torch.zeros_like(r)
        bound = (n - m) // 2 + 1
        max_bound = bound.max().item()

        for k in range(max_bound):
            inc = (
                (-1) ** k
                * _factorial(n - k)
                * r ** (n - 2 * k)
                / (
                    math.factorial(k)
                    * _factorial((n + m) // 2 - k)
                    * _factorial((n - m) // 2 - k)
                )
            )
            out += torch.where(k < bound, inc, 0.0)
        return out

    def zernikepoly(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        n: torch.Tensor,
        l: torch.Tensor,  # noqa: E741
    ):
        m = 2 * l - n
        return torch.where(
            m < 0,
            self.zernikeradial(r, n, -m) * torch.sin(m * phi),
            self.zernikeradial(r, n, m) * torch.cos(m * phi),
        )

    def compute_support_vals(
        self,
        r: torch.Tensor,
        phi: torch.Tensor,
        r_cutoff: float,
        width: float = 0.25,
    ):
        ikernel = torch.arange(self.kernel_size, device=r.device).reshape(-1, 1, 1)
        iidx = torch.argwhere(
            (r <= r_cutoff)
            & torch.full_like(ikernel, True, dtype=torch.bool, device=r.device)
        )

        nshifts = torch.arange(self.kernel_shape, device=r.device)
        nshifts = (nshifts + 1) * nshifts // 2
        nkernel = torch.searchsorted(nshifts, ikernel, right=True) - 1
        lkernel = ikernel - nshifts[nkernel]

        r = r[iidx[:, 1], iidx[:, 2]] / r_cutoff
        phi = phi[iidx[:, 1], iidx[:, 2]]
        n = nkernel[iidx[:, 0], 0, 0]
        l = lkernel[iidx[:, 0], 0, 0]  # noqa: E741

        vals = self.zernikepoly(r, phi, n, l)
        return iidx, vals
