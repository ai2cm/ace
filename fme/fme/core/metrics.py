from typing import List

import numpy as np
import xarray

Dataset = xarray.core.dataset.Dataset


def per_variable_fno_loss(
        ground_truth: Dataset, predictions: Dataset, time:int = 1, dim:List =("grid_yt", "grid_xt")):
    """Computes the per-variable FNO loss for a given time step.
    
    Namely, for each variable, compute
        ||predictions - ground_truth|| / ||ground_truth||
    """
    residual = predictions.isel(time=time) - ground_truth.isel(time=time)
    normalizer = np.linalg.norm(ground_truth.isel(time=time).to_array().to_numpy())
    ret = np.sqrt(np.square(residual).sum(dim=dim))
    return ret / normalizer 


def mean_bias(ground_truth: Dataset, predicted: Dataset, dim: List) -> Dataset:
    """Computes the mean bias across the specified list of dimensions."""
    means = (predicted - ground_truth).mean(dim=dim)
    return means


def global_mean_bias(ground_truth: Dataset, predicted: Dataset, dim=("grid_yt", "grid_xt", "time")):
    """Computes the *global* mean bias of a set of predictions for each variable."""
    return mean_bias(ground_truth, predicted, dim)


def time_mean_bias(ground_truth: Dataset, predicted: Dataset, dim=("grid_yt", "grid_xt")):
    """Computes the *time* mean bias of a set of predictions for each variable."""
    return mean_bias(ground_truth, predicted, dim)