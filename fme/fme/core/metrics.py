from typing import List, Optional

import numpy as np
import torch

Tensor = torch.Tensor


def _create_range(start, stop, num_steps):
    if num_steps == 1:
        raise ValueError("Range must include start and stop, e.g. num_steps > 1.")

    step = (stop - start) / (num_steps - 1)
    ret = torch.arange(0, num_steps) * step + start
    return ret


def spherical_area_weights(num_lat: int, num_lon: int) -> Tensor:
    """Computes the spherical area weights for a regular lat-lon grid."""
    lats = _create_range(89.9999, -89.9999, num_lat)  # Due to floating point issues, cos(deg2rad(90)) != 0.
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    return weights


def per_variable_fno_loss(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the per-variable Fourier Neural Operator (FNO) loss at a single time step.
    
    Namely, for each variable, compute
        ||predicted - ground_truth|| / ||ground_truth||

    Args:
        ground_truth: Tensor of shape (variable, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, grid_yt, grid_xt)

    Returns a tensor of shape (variable,) with the per variable loss.
    """
    normalizer = torch.linalg.norm(ground_truth)
    ret = torch.linalg.norm(predicted - ground_truth, dim=(-1, -2))
    return ret / normalizer 


def weighted_mean_bias(
        ground_truth: Tensor, predicted: Tensor, dim: List[int], weights: Optional[Tensor] = None) -> Tensor:
    """Computes the mean bias across the specified list of dimensions.
    
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        dim:          Dimensions to compute the mean over.
        weights:      Optional weights to apply to the mean. If None, uses `spherical_area_weights`.
        
    Returns a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert len(ground_truth.shape) == len(predicted.shape) == 4, "Expected 4D tensors."
    _, _, num_lat, num_lon = ground_truth.shape
    if weights is None:
        weights = spherical_area_weights(num_lat, num_lon)

    bias = predicted - ground_truth
    means = (bias * weights).mean(dim=dim)
    return means


def weighted_global_mean_bias(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the global mean biases across the variables.
    
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted: Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable,) of the mean biases of each variable.
    """
    return weighted_mean_bias(ground_truth, predicted, (-1, -2, -3))


def weighted_time_mean_bias(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the time mean biases across the variables.
    
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable, time) of the time mean biases of each variable.
    """
    return weighted_mean_bias(ground_truth, predicted, (-1, -2))


def weighted_global_time_rmse(ground_truth: Tensor, predicted: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """
    Computes the weighted global time RMSE over all variables. Namely, for each variable:
    
        (weights * ((xhat - x).mean('time')) ** 2).mean('space')
        
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    assert len(ground_truth.shape) == len(predicted.shape) == 4, "Expected 4D tensors."
    _, _, num_lat, num_lon = ground_truth.shape
    
    if weights is None:
        weights = spherical_area_weights(num_lat, num_lon)
    
    bias = predicted - ground_truth
    space_mean_bias = bias.mean(dim=(1,))
    weighted_mean = (weights * torch.square(space_mean_bias)).mean(dim=(-1, -2))
    print(f"{space_mean_bias=}")
    print(f"{weighted_mean=}")
    rmse = torch.sqrt(weighted_mean)
    return rmse
