from typing import Iterable, Optional, Tuple, Union
from typing_extensions import TypeAlias
import numpy as np

import torch

Dimension: TypeAlias = Union[int, Iterable[int]]
Array: TypeAlias = Union[np.ndarray, torch.Tensor]


def spherical_area_weights(lats: Array, num_lon: int) -> torch.Tensor:
    """Computes area weights given the latitudes of a regular lat-lon grid.

    Args:
        lats: tensor of shape (num_lat,) with the latitudes of the cell centers.
        num_lon: Number of longitude points.
        device: Device to place the tensor on.

    Returns a torch.tensor of shape (num_lat, num_lon).
    """
    if isinstance(lats, np.ndarray):
        lats = torch.from_numpy(lats)
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def weighted_mean(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.

    Returns a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim)

    return (tensor * weights).sum(dim=dim) / weights.expand(tensor.shape).sum(dim=dim)


def weighted_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the mean bias across the specified list of dimensions assuming
    that the weights are applied to the last dimensions, e.g. the spatial dimensions.

    Args:
        truth: torch.Tensor
        predicted: torch.Tensor
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
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """
    Computes the weighted global RMSE over all variables. Namely, for each variable:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    If you want to compute the RMSE over the time dimension, then pass in
    `truth.mean(time_dim)` and `predicted.mean(time_dim)` and specify `dims=space_dims`.

    Args:
        truth: torch.Tensor whose last dimensions are to be weighted
        predicted: torch.Tensor whose last dimensions are to be weighted
        weights: torch.Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns a tensor of shape (variable,) of weighted RMSEs.
    """
    assert (
        truth.shape == predicted.shape
    ), "Truth and predicted should have the same shape."
    sq_bias = torch.square(predicted - truth)
    return weighted_mean(sq_bias, weights=weights, dim=dim).sqrt()


def gradient_magnitude(tensor: torch.Tensor, dim: Dimension = ()) -> torch.Tensor:
    """Compute the magnitude of gradient across the specified dimensions."""
    gradients = torch.gradient(tensor, dim=dim)
    return torch.sqrt(sum([g**2 for g in gradients]))


def weighted_mean_gradient_magnitude(
    tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim: Dimension = ()
) -> torch.Tensor:
    """Compute weighted mean of gradient magnitude across the specified dimensions."""
    return weighted_mean(gradient_magnitude(tensor, dim), weights=weights, dim=dim)


def gradient_magnitude_percent_diff(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
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


def global_dry_air_mass(
    specific_total_water: torch.Tensor,
    surface_pressure: torch.Tensor,
    sigma_grid_offsets_ak: torch.Tensor,
    sigma_grid_offsets_bk: torch.Tensor,
    area: torch.Tensor,
    spatial_dims: Optional[Tuple[int, int]] = (0, 1),
) -> torch.Tensor:
    """Computes the global dry air mass in kg/m/s^2.

    Args:
        specific_total_water (lat, lon, vertical_level), (kg/kg)
        surface_pressure: (lat, lon), (Pa)
        sigma_grid_offsets_ak: Sorted sigma grid offsets ak, (vertical_level + 1,)
        sigma_grid_offsets_bk: Sorted sigma grid offsets bk, (vertical_level + 1,)
        area: Area weights of each grid cell (m^2), (lat, lon)
        spatial_dims: Tuple indicating the spatial (lat, lon) dimensions of the
            input tensors.

    Returns:
        Integrated global dry air mass (1,), (kg)
    """

    num_levels = len(sigma_grid_offsets_ak) - 1

    if (
        num_levels != len(sigma_grid_offsets_bk) - 1
        or num_levels != specific_total_water.shape[-1]
    ):
        raise ValueError(
            (
                "Number of vertical levels in ak, bk, and specific_total_water must"
                "be the same."
            )
        )

    ak, bk = sigma_grid_offsets_ak, sigma_grid_offsets_bk

    pressure_thickness = ((ak + (surface_pressure.unsqueeze(-1) * bk))).diff(
        dim=-1
    )  # Pa
    water_path = torch.sum(
        pressure_thickness * specific_total_water, axis=-1
    )  # Pa * kg / kg = Pa  # type: ignore

    gravity = 9.80665  # m / s^2, valued used by FV3GFS
    dry_air_mass = 1 / gravity * (surface_pressure - water_path)  # kg / m^2
    return (dry_air_mass * area).sum(axis=spatial_dims)
