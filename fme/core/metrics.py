from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import torch
import torch_harmonics

from fme.core.constants import GRAVITY, LATENT_HEAT_OF_FREEZING

Dimension: TypeAlias = int | Iterable[int]
Array: TypeAlias = np.ndarray | torch.Tensor


def spherical_area_weights(lats: Array, num_lon: int) -> torch.Tensor:
    """Computes area weights given the latitudes of a regular lat-lon grid.

    Args:
        lats: tensor of shape (..., num_lat,) with the latitudes of
            the cell centers.
        num_lon: Number of longitude points.

    Returns:
        a torch.tensor of shape (num_lat, num_lon).
    """
    if isinstance(lats, np.ndarray):
        lats = torch.from_numpy(lats)
    expand_shape = [-1 for _ in range(lats.dim())]
    weights = (
        torch.cos(torch.deg2rad(lats)).unsqueeze(-1).expand(expand_shape + [num_lon])
    )
    weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
    return weights


def weighted_sum(
    tensor: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted sum across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the sum.
        dim: Dimensions to compute the sum over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted sum averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.sum(dim=dim, keepdim=keepdim)

    expanded_weights = weights.expand(tensor.shape)

    # remove potential "expected NaNs", i.e. any NaNs with 0 weight
    tensor = tensor.where(expanded_weights != 0.0, 0.0)

    return (tensor * expanded_weights).sum(dim=dim, keepdim=keepdim)


def weighted_mean(
    tensor: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim, keepdim=keepdim)

    expanded_weights = weights.expand(tensor.shape)

    # remove potential "expected NaNs", i.e. any NaNs with 0 weight
    tensor = tensor.where(expanded_weights != 0.0, 0.0)

    return (tensor * expanded_weights).sum(
        dim=dim, keepdim=keepdim
    ) / expanded_weights.sum(dim=dim, keepdim=keepdim)


def weighted_nanmean(
    tensor: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted nanmean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.nanmean(dim=dim, keepdim=keepdim)
    denom = torch.where(torch.isnan(tensor), 0.0, weights.expand(tensor.shape)).sum(
        dim=dim, keepdim=keepdim
    )
    return (tensor * weights).nansum(dim=dim, keepdim=keepdim) / denom


def weighted_std(
    tensor: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the weighted standard deviation across the specified list of dimensions.

    Computed by first computing the weighted variance, then taking the square root.

    weighted_variance = weighted_mean((tensor - weighted_mean(tensor)) ** 2)) ** 0.5

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the variance.
        dim: Dimensions to compute the standard deviation over.

    Returns:
        a tensor of the weighted standard deviation over the
            specified dimensions `dim`.
    """
    if weights is None:
        weights = torch.tensor(1.0, device=tensor.device)

    mean = weighted_mean(tensor, weights=weights, dim=dim, keepdim=True)
    variance = weighted_mean((tensor - mean) ** 2, weights=weights, dim=dim)
    return torch.sqrt(variance)


def weighted_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the mean bias across the specified list of dimensions assuming
    that the weights are applied to the last dimensions, e.g. the spatial dimensions.

    Args:
        truth: torch.Tensor
        predicted: torch.Tensor
        dim: Dimensions to compute the mean over.
        weights: Weights to apply to the mean.

    Returns:
        a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert (
        truth.shape == predicted.shape
    ), "Truth and predicted should have the same shape."
    bias = predicted - truth
    return weighted_mean(bias, weights=weights, dim=dim)


def root_mean_squared_error(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """
    Compute a weighted root mean square error between truth and predicted.

    Namely:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    Args:
        truth: torch.Tensor whose last dimensions are to be weighted
        predicted: torch.Tensor whose last dimensions are to be weighted
        weights: torch.Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns:
        A tensor of weighted RMSEs.
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
    tensor: torch.Tensor, weights: torch.Tensor | None = None, dim: Dimension = ()
) -> torch.Tensor:
    """Compute weighted mean of gradient magnitude across the specified dimensions."""
    return weighted_nanmean(gradient_magnitude(tensor, dim), weights=weights, dim=dim)


def gradient_magnitude_percent_diff(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: torch.Tensor | None = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Compute the percent difference of the weighted mean gradient magnitude across
    the specified dimensions.
    """
    truth_grad_mag = weighted_mean_gradient_magnitude(truth, weights, dim)
    predicted_grad_mag = weighted_mean_gradient_magnitude(predicted, weights, dim)
    return 100 * (predicted_grad_mag - truth_grad_mag) / truth_grad_mag


def rmse_of_time_mean(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: torch.Tensor | None = None,
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
    weights: torch.Tensor | None = None,
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


def surface_pressure_due_to_dry_air(
    surface_pressure: torch.Tensor,
    total_water_path: torch.Tensor,
) -> torch.Tensor:
    """Computes the dry air (Pa).

    Args:
        surface_pressure: the surface preessure in Pa.
        total_water_path: the total water path in kg/m^2.

    Returns:
        The surface pressure due to dry air mass only. (Pa)
    """
    return surface_pressure - GRAVITY * total_water_path


def net_surface_energy_flux(
    lw_rad_down,
    lw_rad_up,
    sw_rad_down,
    sw_rad_up,
    latent_heat_flux,
    sensible_heat_flux,
    frozen_precipitation_rate=None,
) -> torch.Tensor:
    """
    Compute the net surface energy flux from individual terms in budget.

    Args:
        lw_rad_down: Downward longwave surface radiation in W/m^2.
        lw_rad_up: Upward longwave surface radiation in W/m^2.
        sw_rad_down: Downward shortwave surface radiation in W/m^2.
        sw_rad_up: Upward shortwave surface radiation in W/m^2.
        latent_heat_flux: Latent heat flux in W/m^2 (positive=upward).
        sensible_heat_flux: Sensible heat flux in W/m^2 (positive=upward).
        frozen_precipitation_rate (optional): Frozen precipitation rate in kg/m^2/s.

    Returns:
        Net surface energy flux in W/m^2. Positive values indicate energy flowing
        from atmosphere to surface.
    """
    if frozen_precipitation_rate is not None:
        frozen_precip_heat_flux = frozen_precipitation_rate * LATENT_HEAT_OF_FREEZING
    else:
        frozen_precip_heat_flux = 0.0
    net_surface_radiative_flux = sw_rad_down - sw_rad_up + lw_rad_down - lw_rad_up
    net_surface_turbulent_heat_flux = -latent_heat_flux - sensible_heat_flux
    return (
        net_surface_radiative_flux
        + net_surface_turbulent_heat_flux
        - frozen_precip_heat_flux
    )


def net_top_of_atmosphere_energy_flux(
    sw_rad_down, sw_rad_up, lw_rad_up
) -> torch.Tensor:
    """
    Compute the net top-of-atmosphere energy flux from individual terms in budget.

    Args:
        sw_rad_down: Downward shortwave TOA radiation in W/m^2.
        sw_rad_up: Upward longwave TOA radiation in W/m^2.
        lw_rad_up: Upward shortwave TOA radiation in W/m^2.

    Returns:
        Net TOA energy flux in W/m^2. Positive values indicate energy flowing
        into the atmosphere.
    """
    return sw_rad_down - sw_rad_up - lw_rad_up


def quantile(bins: np.ndarray, hist: np.ndarray, probability: float) -> float:
    """
    Calculate the quantile value for a given histogram, using linear
    interpolation within bins. This is the inverse of the cumulative
    distribution function (CDF).

    Args:
        bins: The bin edges of the histogram.
        hist: The histogram values.
        probability: The probability to query the quantile for.

    Returns:
        The quantile value.

    Raises:
        ValueError: If probability is not between 0 and 1.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError("Probabilities must be between 0 and 1.")

    # get the normalized CDF based on the histogram
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    # append initial zero to cdf as there are no values less than the first bin
    cdf = np.insert(cdf, 0, 0)
    # find within which bin the requested pct percentile falls
    bin_idx = np.argmax(cdf > probability) - 1
    # linearly interpolate within the bin to get the percentile value
    c0, c1 = bins[bin_idx], bins[bin_idx + 1]
    p0, p1 = cdf[bin_idx], cdf[bin_idx + 1]
    return c0 + (c1 - c0) * (probability - p0) / (p1 - p0)


def spherical_power_spectrum(
    field: torch.Tensor, sht: torch_harmonics.RealSHT
) -> torch.Tensor:
    """Compute the spherical power spectrum of a field.

    Args:
        field: The field to compute the power spectrum for. It is assumed that
            the last two dimension are latitude and longitude, respectively.
        sht: An initialized spherical harmonics transformer. Must be passed for
            performance reasons.

    Returns:
        The power spectrum of the field. Will have one fewer dimensions than the
            input field.

    Notes:
        Computed by summing over all "m" wavenumbers for each total "l" wavenumber.
    """
    field_sht = sht.forward(field)
    power_spectrum = torch.sum(abs(field_sht) ** 2, dim=-1)
    return power_spectrum
