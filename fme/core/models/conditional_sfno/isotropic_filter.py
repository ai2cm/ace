"""Isotropic Morlet filter basis and monkey-patch for torch-harmonics.

This module defines an isotropic variant of the Morlet filter basis where
all basis functions depend only on radial distance, not on azimuthal angle.
It monkey-patches ``torch_harmonics.filter_basis.get_filter_basis`` (and the
copy already imported into ``torch_harmonics.disco.convolution``) so that
the basis type ``"isotropic morlet"`` is available everywhere DISCO
convolutions are constructed.

This is a temporary measure until the change is contributed upstream or the
dependency is forked.
"""

import math

import torch
from torch_harmonics import filter_basis as _fb
from torch_harmonics.disco import convolution as _conv


class IsotropicMorletFilterBasis(_fb.FilterBasis):
    """Morlet-style filter basis using only radial modes.

    Each basis function is a product of a Hann radial window and a 1-D
    Fourier harmonic in the normalised radial coordinate ``r / r_cutoff``.
    Because none of the basis functions depend on the azimuthal angle
    ``phi``, any learned linear combination is guaranteed to be isotropic
    (radially symmetric).

    ``kernel_shape`` is a single integer giving the number of radial modes.
    If a tuple is provided, only the first element is used.
    """

    def __init__(
        self,
        kernel_shape: int | tuple[int] | tuple[int, int],
    ):
        if isinstance(kernel_shape, list | tuple):
            kernel_shape = kernel_shape[0]
        if not isinstance(kernel_shape, int):
            raise ValueError(
                f"expected kernel_shape to be an integer but got "
                f"{kernel_shape} instead."
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


# ---------------------------------------------------------------------------
# Monkey-patch get_filter_basis
# ---------------------------------------------------------------------------

_original_get_filter_basis = _fb.get_filter_basis


@_fb.lru_cache(typed=True, copy=False)
def _patched_get_filter_basis(
    kernel_shape: int | tuple[int] | tuple[int, int],
    basis_type: str,
) -> _fb.FilterBasis:
    if basis_type == "isotropic morlet":
        return IsotropicMorletFilterBasis(kernel_shape=kernel_shape)
    return _original_get_filter_basis(kernel_shape, basis_type)


_fb.get_filter_basis = _patched_get_filter_basis
_conv.get_filter_basis = _patched_get_filter_basis
