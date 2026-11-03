"""Analytic Gaussian target grids for regridding.

Ported from the era5 pipeline's F<N> grid code (scripts/era5): an F<N> grid
has 2N true Gaussian latitudes (roots of the Legendre polynomial P_{2N}) and
4N equally spaced longitudes. Cell areas come exactly from the Gaussian
quadrature weights, unlike the areas stored with earlier ocean datasets
produced by scripts/data_process/compute_ocean_dataset.py, which used
xesmf.util.cell_area with the Earth's polar radius (6356 km) and are ~0.5%
low as a result.
"""

import numpy as np
import xarray as xr
from numpy.polynomial.legendre import leggauss

# Gaussian grid specs: name -> N (grid number; nlat=2N, nlon=4N)
GAUSSIAN_GRID_N = {
    "F22.5": 22.5,
    "F90": 90,
}

MEAN_EARTH_RADIUS_M = 6371.0e3


def _cell_bounds(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Compute cell boundaries from centers, clipping to [lo, hi]."""
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[lo], midpoints, [hi]])


def _gaussian_latitudes_and_weights(n: int | float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian latitudes (degrees, south-to-north) and quadrature weights
    for grid number N (2N latitudes)."""
    nlat = round(2 * n)
    x, w = leggauss(nlat)
    lat = np.degrees(np.arcsin(x))
    order = np.argsort(lat)
    return lat[order], w[order]


def make_target_grid(name: str) -> xr.Dataset:
    """Create a Gaussian target grid dataset by name (e.g. "F90").

    Returns a dataset with 1D ``lat``/``lon`` centers and ``lat_b``/``lon_b``
    bounds (the layout xESMF expects for conservative regridding) and an
    ``areacello`` variable of exact cell areas in m^2.
    """
    n = GAUSSIAN_GRID_N[name]
    lat, weights = _gaussian_latitudes_and_weights(n)
    nlon = round(4 * n)
    dlon = 360.0 / nlon
    # Longitude centers offset by half a grid spacing, matching the era5 grids
    lon = np.linspace(dlon / 2, 360 - dlon / 2, nlon)
    lat_b = _cell_bounds(lat, -90, 90)
    lon_b = _cell_bounds(lon, 0, 360)
    # Quadrature weights integrate to 2 over sin(lat) in [-1, 1], so
    # area[j, i] = R^2 * w[j] * dlon_radians sums to 4 pi R^2 exactly.
    area = np.broadcast_to(
        (MEAN_EARTH_RADIUS_M**2 * np.deg2rad(dlon) * weights)[:, None], (len(lat), nlon)
    )
    return xr.Dataset(
        data_vars={
            "areacello": (
                ["lat", "lon"],
                area.copy(),
                {
                    "long_name": "cell area from Gaussian quadrature weights",
                    "units": "m2",
                },
            ),
        },
        coords={
            "lat": (["lat"], lat),
            "lon": (["lon"], lon),
            "lat_b": (["lat_b"], lat_b),
            "lon_b": (["lon_b"], lon_b),
        },
    )
