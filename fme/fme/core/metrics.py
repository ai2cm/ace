from typing import List

import numpy as np
import torch

Tensor = torch.Tensor


def _create_range(start, stop, num_steps):
    step = (stop - start) / (num_steps - 1)
    ret = torch.arange(0, num_steps) * step + start
    return ret


def _spherical_area_weights(num_lat, num_lon):
    lats = _create_range(90, -90, num_lat)
    weights = np.tile(np.cos(np.deg2rad(lats)), (num_lon, 1)).T
    return weights


def per_variable_fno_loss(
        ground_truth: Tensor, predictions: Tensor, time:int = 1, dim:List =("grid_yt", "grid_xt")):
    """Computes the per-variable FNO loss for a given time step.
    
    Namely, for each variable, compute
        ||predictions - ground_truth|| / ||ground_truth||
    """
    residual = predictions.isel(time=time) - ground_truth.isel(time=time)
    normalizer = np.linalg.norm(ground_truth.isel(time=time).to_array().to_numpy())
    ret = np.sqrt(np.square(residual).sum(dim=dim))
    return ret / normalizer 


def weighted_mean_bias(ground_truth: Tensor, predicted: Tensor, dim: List[int]) -> Tensor:
    """Computes the mean bias across the specified list of dimensions.
    
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        dim:          Dimensions to compute the mean over.
        
    Returns a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    _, _, num_lat, num_lon = ground_truth.shape
    weights = _spherical_area_weights(num_lat, num_lon)
    residuals = predicted - ground_truth
    means = (residuals * weights).mean(dim=dim)
    return means


def weighted_global_mean_bias(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Computes the global mean biases across the variables.
    
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        
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


def weighted_global_time_rmse(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """
    Computes the weighted global time RMSE over all variables. Namely, for each variable:
    
        (weights * ((xhat - x).mean('time')) ** 2).mean('space')
        
    Args:
        ground_truth: Tensor of shape (variable, time, grid_yt, grid_xt)
        predicted:    Tensor of shape (variable, time, grid_yt, grid_xt)
        
    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    _, _, num_lat, num_lon = ground_truth.shape
    weights = torch.Tensor(_spherical_area_weights(num_lat, num_lon))
    
    residuals = predicted - ground_truth
    space_residuals = residuals.mean(dim=(1,))
    weighted_mean = (weights * torch.square(space_residuals)).mean(dim=(1, 2))
    rmse = torch.sqrt(weighted_mean)
    return rmse