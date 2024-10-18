"""Pure functions, e.g. metrics and math functions that are useful for downscaling."""

from typing import Callable, Collection, Tuple

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


def filter_tensor_mapping(x: TensorMapping, keys: Collection[str]) -> TensorMapping:
    """
    Filters a tensor mapping based on a set of keys.

    Args:
        x: The input tensor mapping.
        keys: The set of keys to filter the tensor mapping.
    """
    return {k: v for (k, v) in x.items() if k in keys}


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


def compute_crps(
    target: torch.Tensor,
    prediction: torch.Tensor,
    sample_dim: int = 1,
):
    """
    CRPS is defined for one-dimensional random variables as

        CRPS(F, x) = integral_z (F(z) - H(z - x))^2 dz

    where $F(x)$ is the cumulative distribution function (CDF) of the forecast
    distribution $F$ and $H(x)$ denotes the Heaviside step function, where $x$
    is an observation.

    This implementation is based on the identity:

        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    This reference (but not the implementation) is taken from the
    Proper Scoring repository [1]_. It can also be found on Wikipedia [2]_
    and is taken originally from Gneiting and Raftery (2007) [3]_ Equation 21.

    Args:
        target: The target tensor without a sample dimension
        prediction: The prediction tensor with a sample dimension
        sample_dim: The dimension of `prediction` corresponding to sample.

    .. [1] https://github.com/properscoring/properscoring/blob/master/properscoring/_crps.py
    .. [2] https://en.wikipedia.org/wiki/Scoring_rule
    .. [3] https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
    """  # noqa: E501
    sample_mae_estimate = get_sample_mae_estimate(prediction, sample_dim)
    truth_mae = torch.abs(target.unsqueeze(sample_dim) - prediction).mean(
        axis=sample_dim
    )
    return truth_mae - 0.5 * sample_mae_estimate


def get_sample_mae_estimate(prediction: torch.Tensor, sample_dim: int = 1):
    n_samples = prediction.shape[sample_dim]
    out_shape = list(prediction.shape)
    out_shape.pop(sample_dim)
    if n_samples == 1:
        return torch.full(
            out_shape,
            fill_value=torch.nan,
            device=prediction.device,
            dtype=prediction.dtype,
        )
    else:
        sample_mae_estimate = torch.zeros(
            out_shape, device=prediction.device, dtype=prediction.dtype
        )
        for i in range(1, n_samples):
            sample_mae_estimate += torch.abs(
                prediction - torch.roll(prediction, shifts=i, dims=sample_dim)
            ).mean(axis=sample_dim)
        sample_mae_estimate /= n_samples - 1
    return sample_mae_estimate


def compute_mae_error(
    target: torch.Tensor,
    prediction: torch.Tensor,
    sample_dim: int = 1,
):
    """
    Computes the following metric, which is like CRPS but goes
    to zero for a perfect forecast:

        mae_error(F, x) = E_F|X - x| - E_F|X - X'|

    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    Args:
        target: The target tensor without a sample dimension
        prediction: The prediction tensor with a sample dimension
        sample_dim: The dimension of `prediction` corresponding to sample.
    """
    sample_mae_estimate = get_sample_mae_estimate(prediction, sample_dim)
    truth_mae = torch.abs(target.unsqueeze(sample_dim) - prediction).mean(
        axis=sample_dim
    )
    return truth_mae - sample_mae_estimate


def compute_psnr(
    prediction: torch.Tensor, target: torch.Tensor, add_channel_dim: bool
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio between the predicted and target tensors.

    Args:
        prediction: tensor either of shape (batch, channel, height, width)
            or (batch, height, width)
        target: tensor of shape (batch, channel, height, width)
            or (batch, height, width)
        add_channel_dim: Add a channel dim if it is missing.
    """
    prediction_norm, target_norm = _normalize_tensors(prediction, target)
    if add_channel_dim:
        channel_dim = -3
        prediction_norm = prediction_norm.unsqueeze(channel_dim)
        target_norm = target_norm.unsqueeze(channel_dim)
    return piq.psnr(prediction_norm, target_norm)


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    add_channel_dim: bool,
    *args,
    **kwargs
) -> torch.Tensor:
    """Normalize data to unit range and compute piq.ssim.

    Args:
        prediction: tensor either of shape (batch, channel, height, width)
            or (batch, height, width)
        target: tensor of shape (batch, channel, height, width)
            or (batch, height, width)
        add_channel_dim: Add a channel dim if it is missing.
    """
    prediction_norm, target_norm = _normalize_tensors(prediction, target)
    if add_channel_dim:
        channel_dim = -3
        prediction_norm = prediction_norm.unsqueeze(channel_dim)
        target_norm = target_norm.unsqueeze(channel_dim)
    # ssim does not return a list despite the type hint
    return piq.ssim(prediction_norm, target_norm, *args, **kwargs)  # type: ignore


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


def interpolate(tensor: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        tensor,
        scale_factor=[scale_factor, scale_factor],
        mode="bicubic",
        align_corners=True,
    )
