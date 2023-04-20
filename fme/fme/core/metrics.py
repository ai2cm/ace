from typing import List, Optional

import numpy as np
import torch

Tensor = torch.Tensor


def _create_range(start: int , stop: int, num_steps: int) -> Tensor:
    if num_steps == 1:
        raise ValueError("Range must include start and stop, e.g. num_steps > 1.")

    step = (stop - start) / (num_steps - 1)
    ret = torch.arange(0, num_steps) * step + start
    return ret


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
    return _create_range(start, stop, num_points)


def spherical_area_weights(num_lat: int, num_lon: int) -> Tensor:
    """Computes the spherical area weights for a regular lat-lon grid."""
    lats = lat_cell_centers(num_lat)
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    return weights


def per_variable_rmse(truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the per-variable root mean-squared error between the truth and predicted values.
    All the dimensions except the first are averaged over, e.g. the spatial dimensions.
    
    Namely, for each variable, compute
        ||predicted - truth||_2

    Args:
        truth: Tensor of shape (variable, ... mean dims)
        predicted: Tensor of shape (variable, ... mean dims)

    Returns a tensor of shape (variable,) with the per variable loss.
    """
    mean_dims = tuple(range(1, truth.ndim))
    ret = (predicted - truth).square().mean(dim=mean_dims).sqrt()
    return ret


def weighted_mean_bias(
        truth: Tensor, predicted: Tensor, dim: List[int], weights: Optional[Tensor] = None) -> Tensor:
    """Computes the mean bias across the specified list of dimensions.
    
    Args:
        truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted: Tensor of shape (variable, time, grid_yt, grid_xt)
        dim: Dimensions to compute the mean over.
        weights: Optional weights to apply to the mean. If None, uses `spherical_area_weights`.
        
    Returns a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert len(truth.shape) == len(predicted.shape) == 4, "Expected 4D tensors."
    _, _, num_lat, num_lon = truth.shape
    if weights is None:
        weights = spherical_area_weights(num_lat, num_lon)

    bias = predicted - truth
    means = (bias * weights).mean(dim=dim)
    return means


def weighted_global_mean_bias(truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the global mean biases across the variables.
    
    Args:
        truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted: Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable,) of the mean biases of each variable.
    """
    return weighted_mean_bias(truth, predicted, (-1, -2, -3))


def weighted_time_mean_bias(truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the time mean biases across the variables.
    
    Args:
        truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted: Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable, time) of the time mean biases of each variable.
    """
    return weighted_mean_bias(truth, predicted, (-1, -2))


def weighted_global_time_rmse(truth: Tensor, predicted: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """
    Computes the weighted global time RMSE over all variables. Namely, for each variable:
    
        (weights * ((xhat - x).mean('time')) ** 2).mean('space')
        
    Args:
        truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    assert len(truth.shape) == len(predicted.shape) == 4, "Expected 4D tensors."
    _, _, num_lat, num_lon = truth.shape
    
    if weights is None:
        weights = spherical_area_weights(num_lat, num_lon)
    
    bias = predicted - truth
    space_mean_bias = bias.mean(dim=(1,))
    weighted_mean = (weights * torch.square(space_mean_bias)).mean(dim=(-1, -2))
    print(f"{space_mean_bias=}")
    print(f"{weighted_mean=}")
    rmse = torch.sqrt(weighted_mean)
    return rmse
