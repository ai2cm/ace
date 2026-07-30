# Forked from torch-harmonics (BSD-3-Clause)
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.fft as fft
import torch.nn as nn


def _pad_dim_right(
    x: torch.Tensor, dim: int, target_size: int, value: float = 0.0
) -> torch.Tensor:
    """Pad tensor along a single dimension to target_size (right-side only)."""
    ndim = x.ndim
    dim = dim if dim >= 0 else ndim + dim
    pad_amount = target_size - x.shape[dim]
    pad_spec = [0] * (2 * ndim)
    pad_spec[(ndim - 1 - dim) * 2 + 1] = pad_amount
    return nn.functional.pad(x, tuple(pad_spec), value=value)


def rfft(
    x: torch.Tensor, nmodes: int | None = None, dim: int = -1, **kwargs
) -> torch.Tensor:
    """Real FFT with correct padding/truncation of modes."""
    if "n" in kwargs:
        raise ValueError("The 'n' argument is not allowed. Use 'nmodes' instead.")

    x = fft.rfft(x, dim=dim, **kwargs)

    if nmodes is not None and nmodes > x.shape[dim]:
        x = _pad_dim_right(x, dim, nmodes, value=0.0)
    elif nmodes is not None and nmodes < x.shape[dim]:
        x = x.narrow(dim, 0, nmodes)

    return x


def irfft(
    x: torch.Tensor, n: int | None = None, dim: int = -1, **kwargs
) -> torch.Tensor:
    """Inverse real FFT with Hermitian symmetry enforcement."""
    if n is None:
        n = 2 * (x.size(dim) - 1)

    x[..., 0].imag = 0.0
    if (n % 2 == 0) and (n // 2 < x.size(dim)):
        x[..., n // 2].imag = 0.0

    x = fft.irfft(x, n=n, dim=dim, **kwargs)
    return x
