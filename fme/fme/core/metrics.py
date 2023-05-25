from typing import Iterable, Optional, Union
from typing_extensions import TypeAlias

import torch

Dimension: TypeAlias = Union[int, Iterable[int]]
Tensor: TypeAlias = torch.Tensor


def lat_cell_centers(num_points: int, device=None) -> Tensor:
    """Returns the latitudes of the cell centers for a regular lat-lon grid.

    Args:
        num_points: Number of latitude points.
        device: Device to place the tensor on.

    Returns a tensor of shape (num_points,) with the latitudes of the cell centers.
    The order is from negative to positive latitudes, e.g. [-89, -87, ..., 87, 89].
    """
    offset = (180.0 / num_points) / 2.0
    pole_center = 90.0 - offset
    start, stop = -pole_center, pole_center
    return torch.linspace(start, stop, num_points, device=device)


def spherical_area_weights(num_lat: int, num_lon: int, device=None) -> Tensor:
    """Computes the spherical area weights (unitless) for a regular lat-lon grid.

    Args:
        num_lat: Number of latitude points.
        num_lon: Number of longitude points.
        device: Device to place the tensor on.

    Returns a tensor of shape (num_lat, num_lon).
    """
    lats = lat_cell_centers(num_lat, device=device)
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def weighted_mean(
    tensor: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.

    Returns a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim)

    return (tensor * weights).sum(dim=dim) / weights.expand(tensor.shape).sum(dim=dim)


def weighted_mean_bias(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """Computes the mean bias across the specified list of dimensions assuming
    that the weights are applied to the last dimensions, e.g. the spatial dimensions.

    Args:
        truth: Tensor
        predicted: Tensor
        dim: Dimensions to compute the mean over.
        weights: Weights to apply to the mean.

    Returns a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert (
        truth.shape == predicted.shape
    ), "Truth and predicted should have the same shape."
    bias = predicted - truth
    return weighted_mean(bias, weights=weights, dim=dim)


def root_mean_squared_error(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """
    Computes the weighted global RMSE over all variables. Namely, for each variable:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    If you want to compute the RMSE over the time dimension, then pass in
    `truth.mean(time_dim)` and `predicted.mean(time_dim)` and specify `dims=space_dims`.

    Args:
        truth: Tensor whose last dimensions are to be weighted
        predicted: Tensor whose last dimensions are to be weighted
        weights: Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    assert (
        truth.shape == predicted.shape
    ), "Truth and predicted should have the same shape."
    sq_bias = torch.square(predicted - truth)
    return weighted_mean(sq_bias, weights=weights, dim=dim).sqrt()


def gradient_magnitude(tensor: Tensor, dim: Dimension = ()) -> Tensor:
    """Compute the magnitude of gradient across the specified dimensions."""
    gradients = torch.gradient(tensor, dim=dim)
    return torch.sqrt(sum([g**2 for g in gradients]))


def weighted_mean_gradient_magnitude(
    tensor: Tensor, weights: Optional[Tensor] = None, dim: Dimension = ()
) -> Tensor:
    """Compute weighted mean of gradient magnitude across the specified dimensions."""
    return weighted_mean(gradient_magnitude(tensor, dim), weights=weights, dim=dim)


def gradient_magnitude_percent_diff(
    truth: Tensor,
    predicted: Tensor,
    weights: Optional[Tensor] = None,
    dim: Dimension = (),
) -> Tensor:
    """Compute the percent difference of the weighted mean gradient magnitude across
    the specified dimensions."""
    truth_grad_mag = weighted_mean_gradient_magnitude(truth, weights, dim)
    predicted_grad_mag = weighted_mean_gradient_magnitude(predicted, weights, dim)
    return 100 * (predicted_grad_mag - truth_grad_mag) / truth_grad_mag


def rmse_of_time_mean(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    time_dim: Dimension = 0,
    spatial_dims: Dimension = (-2, -1),
) -> torch.Tensor:
    """Compute the RMSE of the time-average given truth and predicted.

    Args:
        truth: truth tensor
        predicted: predicted tensor
        weights: weights to use for computing spatial RMSE
        time_dim: time dimension
        spatial_dims: spatial dimensions over which RMSE is calculated

    Returns:
        The RMSE between the time-mean of the two input tensors. The time and
        spatial dims are reduced.
    """
    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)
    ret = root_mean_squared_error(
        truth_time_mean, predicted_time_mean, weights=weights, dim=spatial_dims
    )
    return ret


def time_and_global_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    time_dim: Dimension = 0,
    spatial_dims: Dimension = (-2, -1),
) -> torch.Tensor:
    """Compute the global- and time-mean bias given truth and predicted.

    Args:
        truth: truth tensor
        predicted: predicted tensor
        weights: weights to use for computing the global mean
        time_dim: time dimension
        spatial_dims: spatial dimensions over which global mean is calculated

    Returns:
        The global- and time-mean bias between the predicted and truth tensors. The
        time and spatial dims are reduced.
    """
    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)
    result = weighted_mean(
        predicted_time_mean - truth_time_mean, weights=weights, dim=spatial_dims
    )
    return result
