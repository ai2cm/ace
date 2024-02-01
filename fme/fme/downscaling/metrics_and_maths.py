"""Pure functions, e.g. metrics and math functions that are useful for downscaling."""

from typing import Callable, Tuple

import torch

from fme.core.typing_ import TensorMapping

from . import piq


def map_tensor_mapping(
    fun: Callable[..., torch.Tensor]
) -> Callable[..., TensorMapping]:
    """
    Closure over a list comprehension that applies the given function `fun` to
    each value in a TensorMapping, returning a TensorMapping mapping the same
    keys to the each corresponding result, e.g.

        {k: fun(v) for k, v in x.items()}

    Note that the function `fun` can take any number of arguments so this also
    works:

        {k: fun(x[k], y[k]) for k in x.keys()}

    where `x` and `y` are both TensorMappings with the same keys.

    Args:
        fun: The function to apply to each corresponding key-value pair.

    Returns:
        A function that applies the given function to each corresponding
        key-value pair of multiple TensorMappings.
    """

    def ret(*args: TensorMapping):
        if not all(args[0].keys() == named_tensor.keys() for named_tensor in args):
            raise ValueError("All NamedTensors must have the same keys")

        return {
            k: fun(*(named_tensor[k] for named_tensor in args)) for k in args[0].keys()
        }

    return ret


def min_max_normalization(
    x: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
) -> torch.Tensor:
    """
    Normalize the input tensor to the unit range [0, 1].

    Args:
        x: Input tensor.
        min_: Minimum value for normalization.
        max_: Maximum value for normalization.

    Returns:
        The normalized tensor. If the input tensor is constant, returns a tensor
        of 0.5.
    """
    if min_ == max_:
        return torch.full_like(x, fill_value=0.5)
    return (x - min_) / (max_ - min_)


def _normalize_tensors(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_ = torch.min(x.min(), y.min())
    max_ = torch.max(x.max(), y.max())
    return min_max_normalization(x, min_, max_), min_max_normalization(x, min_, max_)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize data to unit range and compute piq.psnr."""
    pred_norm, target_norm = _normalize_tensors(pred, target)
    return piq.psnr(pred_norm, target_norm)


def compute_ssim(
    pred: torch.Tensor, target: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    """Normalize data to unit range and compute piq.ssim."""
    pred_norm, target_norm = _normalize_tensors(pred, target)
    return piq.ssim(pred_norm, target_norm, *args, **kwargs)  # type: ignore


def compute_zonal_power_spectrum(
    tensor: torch.Tensor,
    lats: torch.Tensor,
    min_abs_lat: int = 30,
    max_abs_lat: int = 60,
) -> torch.Tensor:
    """
    Zonal power spectrum of a given tensor over specified latitude ranges.

    This function computes the zonal power spectrum by first performing a real
    Fourier transform along the longitudinal axis of the tensor, then
    calculating the power spectrum and averaging over the specified latitudinal
    range.

    Args:
        tensor: Tensor of shape [..., latitude, longitude].
        lats: Tensor containing latitude values corresponding to the tensor.
        min_abs_lat: Minimum latitude value for the computation.
        max_abs_lat: Maximum latitude value for the computation.

    Returns:
        torch.Tensor: Averaged zonal power spectrum over the specified latitude range.
    """
    uhat = torch.fft.rfft(tensor, dim=-1)
    power = torch.real(uhat * torch.conj(uhat))

    # Account for negative wave numbers
    ones_and_twos = torch.tensor(
        [1] + [2] * (power.shape[-1] - 1), device=tensor.device
    )
    power *= ones_and_twos

    # Apply latitude mask
    mask = (min_abs_lat <= torch.abs(lats)) & (torch.abs(lats) <= max_abs_lat)
    power = power[..., mask, :]

    return power.mean(dim=-2)
