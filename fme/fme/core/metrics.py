from typing import Iterable, Optional, TypeAlias, Union

import torch

Dimension: TypeAlias = Union[int, Iterable[int]]
Tensor: TypeAlias = torch.Tensor


def lat_cell_centers(num_points: int) -> Tensor:
    """Returns the latitudes of the cell centers for a regular lat-lon grid.

    Args:
        num_points: Number of latitude points.

    Returns a tensor of shape (num_points,) with the latitudes of the cell centers.
    The order is from negative to positive latitudes, e.g. [-89, -87, ..., 87, 89].
    """
    offset = (180.0 / num_points) / 2.0
    pole_center = 90.0 - offset
    start, stop = -pole_center, pole_center
    return torch.linspace(start, stop, num_points)


def spherical_area_weights(num_lat: int, num_lon: int) -> Tensor:
    """Computes the spherical area weights (unitless) for a regular lat-lon grid.

    Returns a tensor of shape (num_lat, num_lon).
    """
    lats = lat_cell_centers(num_lat)
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
    return (tensor * weights).sum(dim=dim) / weights.sum(dim=dim)


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
