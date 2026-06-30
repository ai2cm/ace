"""Geostrophic / ageostrophic wind diagnostics on pressure surfaces.

Geostrophic balance is defined on pressure surfaces: the geostrophic wind is
reconstructed from the horizontal gradient of the geopotential height on a
surface, and the ageostrophic wind is the residual of the actual wind from it.
The global-mean ageostrophic wind speed is a diagnostic of how dynamically
coherent the predicted pressure/wind fields are -- a low-RMSE model can still
have an incoherent wind/pressure relationship.

Horizontal gradients are computed spectrally via the spherical-harmonic
transform (pole-clean), not finite differences: the gradient of a scalar field
is its irrotational (spheroidal) part, recovered by treating the field as a
velocity potential and applying the inverse vector SHT. This is algebraically
identical to the validated vorticity/divergence inversion in
``scripts/vort_div/winds.py`` (the Laplacian and inverse-Laplacian factors
cancel), specialized to zero vorticity.
"""

import torch
from torch_harmonics.sht import InverseRealVectorSHT, RealSHT

from fme.core.constants import EARTH_RADIUS, GRAVITY, OMEGA

# The ERA5 lat/lon data is treated as an equiangular grid throughout fme (the
# SFNO ``data_grid`` default), so the diagnostic uses the same grid for a
# consistent spectral representation.
_GRID = "equiangular"

# Transforms precompute Legendre quadrature buffers, so they are cached by
# (nlat, nlon, device, dtype) and reused across timesteps/batches.
_TRANSFORM_CACHE: dict[
    tuple[int, int, str, torch.dtype], tuple[RealSHT, InverseRealVectorSHT]
] = {}


def _get_transforms(
    nlat: int, nlon: int, device: torch.device, dtype: torch.dtype
) -> tuple[RealSHT, InverseRealVectorSHT]:
    key = (nlat, nlon, str(device), dtype)
    if key not in _TRANSFORM_CACHE:
        sht = RealSHT(nlat, nlon, grid=_GRID).to(device=device, dtype=dtype)
        inverse_vector_sht = InverseRealVectorSHT(nlat, nlon, grid=_GRID).to(
            device=device, dtype=dtype
        )
        _TRANSFORM_CACHE[key] = (sht, inverse_vector_sht)
    return _TRANSFORM_CACHE[key]


def horizontal_gradient(
    scalar: torch.Tensor, radius: float = EARTH_RADIUS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectral horizontal gradient of a scalar field on the sphere.

    Args:
        scalar: field with shape (..., nlat, nlon).
        radius: sphere radius in meters used to convert the unit-sphere gradient
            to a physical (per-meter) gradient.

    Returns:
        ``(grad_east, grad_north)``: the eastward and northward components of the
        gradient, i.e. ``(1 / (a cos(phi))) d/dlambda`` and ``(1 / a) d/dphi``,
        each with shape (..., nlat, nlon) and units of ``[scalar] / m``.
    """
    nlat, nlon = scalar.shape[-2], scalar.shape[-1]
    sht, inverse_vector_sht = _get_transforms(nlat, nlon, scalar.device, scalar.dtype)
    coeffs = sht(scalar)
    # Treat the scalar as the spheroidal (velocity-potential) component with zero
    # toroidal part; the inverse vector SHT then returns its gradient field. The
    # transform returns components in the (colatitude, longitude) basis: slot 0
    # is the colatitude (southward) component, so the northward gradient is its
    # negative; slot 1 is the eastward component directly.
    zeros = torch.zeros_like(coeffs)
    spheroidal_toroidal = torch.stack([coeffs, zeros], dim=-3)
    colatitude_east = inverse_vector_sht(spheroidal_toroidal)
    grad_north = -colatitude_east[..., 0, :, :] / radius
    grad_east = colatitude_east[..., 1, :, :] / radius
    return grad_east, grad_north


def coriolis_parameter(latitude_deg: torch.Tensor) -> torch.Tensor:
    """Coriolis parameter ``f = 2 * OMEGA * sin(latitude)`` (1/s)."""
    return 2.0 * OMEGA * torch.sin(torch.deg2rad(latitude_deg))


def geostrophic_wind(
    height: torch.Tensor, latitude_deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geostrophic wind from geopotential height on a pressure surface.

    ``u_g = -(g / (f a)) dh/dphi``, ``v_g = (g / (f a cos(phi))) dh/dlambda``.

    Args:
        height: geopotential height in meters, shape (..., nlat, nlon).
        latitude_deg: latitudes in degrees, broadcastable to (..., nlat, nlon)
            (typically shape (nlat, 1)).

    Returns:
        ``(u_g, v_g)``: eastward and northward geostrophic wind components (m/s).
        Values are not meaningful near the equator where ``f -> 0``; callers
        should mask the equatorial band (see ``ageostrophic_wind_speed``).
    """
    grad_east, grad_north = horizontal_gradient(height)
    f = coriolis_parameter(latitude_deg)
    u_g = -(GRAVITY / f) * grad_north
    v_g = (GRAVITY / f) * grad_east
    return u_g, v_g


def ageostrophic_wind_speed(
    eastward_wind: torch.Tensor,
    northward_wind: torch.Tensor,
    height: torch.Tensor,
    latitude_deg: torch.Tensor,
    equatorial_mask_deg: float = 10.0,
) -> torch.Tensor:
    """Magnitude of the ageostrophic wind on a pressure surface (m/s).

    ``sqrt((u - u_g)^2 + (v - v_g)^2)``, set to NaN within
    ``|latitude| < equatorial_mask_deg`` where geostrophy does not hold
    (``f -> 0``). The NaN band is meant to be excluded by a NaN-aware area mean
    downstream rather than zero-filled.

    Args:
        eastward_wind: predicted eastward wind on the surface (m/s).
        northward_wind: predicted northward wind on the surface (m/s).
        height: predicted geopotential height on the surface (m).
        latitude_deg: latitudes in degrees, broadcastable to the field shape.
        equatorial_mask_deg: half-width of the NaN-masked equatorial band.

    Returns:
        Ageostrophic wind speed with the equatorial band set to NaN, same shape
        as the input fields.
    """
    u_g, v_g = geostrophic_wind(height, latitude_deg)
    speed = torch.sqrt((eastward_wind - u_g) ** 2 + (northward_wind - v_g) ** 2)
    equatorial = torch.abs(latitude_deg) < equatorial_mask_deg
    return torch.where(
        equatorial.broadcast_to(speed.shape),
        torch.full_like(speed, float("nan")),
        speed,
    )
